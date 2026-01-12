import math

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.utils import group_argsort, scatter, softmax, to_dense_batch
from torch.nn.parameter import Parameter

from configs import base_config
from generate_data import EDGE_MASK_ONE, MASK, NODE_MASK_ONE, NODE_POINTER, SPEC
from utils import from_binary_states, gumbel_softmax, node_pointer_loss, temp_by_step



class StatesEncoder(torch.nn.Module):
    def __init__(self, h, num_binary_states):
        super().__init__()
        self.emb = torch.nn.Embedding(2**num_binary_states, h)

    def forward(self, states):
        return self.emb(from_binary_states(states))


class SelectBest(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.use_select_best = config.use_select_best
        self.emb = torch.nn.Embedding(2 ** (config.num_node_states + 1), config.h)

    def forward(self, binary_states, scalars, index):
        states = 2 * from_binary_states(binary_states)
        if self.use_select_best:
            
            group_with_reciever = torch.cat( #just a stacking of the states & (batch/edge) index
                [torch.unsqueeze(states, -1), torch.unsqueeze(index, -1)], dim=1
            ) # shape [n_nodes, n_indexes(=2)], each combination is a group
            _, group_index = torch.unique( #finds the unique groups based on the combined state, throws away the batch index, returns the unique group index per node
                group_with_reciever, sorted=False, return_inverse=True, dim=0
            )

            #1 for best value inside the group, 0 else
            best_in_group = gumbel_softmax( #dunno why it uses gumbel softmax here, not just argmax    
                -scalars.squeeze(), #chooses the minimum scalar
                group_index, 
                tau=0.0, 
                use_noise=False
            )

            state_with_best = states + best_in_group
            return self.emb(state_with_best.long())

        else:
            return self.emb(states.long())

class AttentionModule(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h
        self.h = h

        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)

        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)

        self.edge_key = Linear(h, h, bias=False)
        self.edge_value = Linear(h, h, bias=False)

        self.select_best_virtual = SelectBest(config)
        self.select_best_by_reciever = SelectBest(config)

        self.use_static_fts = config.use_static_fts
        self.static_fts_encoder = StatesEncoder(h, 2)
        self.combine_fts = Linear(3 * h, h, bias=False)

        self.use_noise = config.use_noise
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        node_fts = self.select_best_from_virtual(node_states, scalars, batch)
        edge_fts = self.edge_states_encoder(edge_states)

        Q = self.lin_query(node_fts)
        K = self.lin_key(node_fts)
        V = self.lin_value(node_fts)

        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts, scalars, batch)

        message = self.compute_message(
            Q=Q,
            K=K,
            V=V,
            edge_K=edge_K,
            edge_V=edge_V,
            edge_index=batch.edge_index,
            training_step=training_step,
        )

        node_fts = node_fts + scatter(message, index=batch.edge_index[1])
        edge_fts = edge_fts + message
        return node_fts, edge_fts

    def compute_message(self, Q, K, V, edge_K, edge_V, edge_index, training_step):
        Q = Q[edge_index[1]]
        K = K[edge_index[0]] + edge_K
        V = V[edge_index[0]] + edge_V

        alpha = (Q * K).sum(dim=-1) / math.sqrt(self.h)

        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        alpha = gumbel_softmax(alpha, edge_index[1], tau=tau, use_noise=use_noise)

        return V * alpha.view(-1, 1)

    def compute_static_fts(self, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        sender_s = node_scalars[batch.edge_index[0]]
        reciever_s = node_scalars[batch.edge_index[1]]

        #relaxation that are giganticly helpful inductive biases for easier inference
        rlx = scalars < reciever_s
        rlx_d = sender_s + scalars < reciever_s

        fts = torch.cat([
            rlx if self.use_static_fts[0] else torch.zeros_like(rlx), 
            rlx_d if self.use_static_fts[1] else torch.zeros_like(rlx_d)
        ], dim=-1).long()
        return self.static_fts_encoder(fts)

    def select_best_from_virtual(self, node_states, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        return self.select_best_virtual(node_states, node_scalars, batch.batch)

    def combined_edge_KV(self, node_states, edge_fts, scalars, batch):
        select_best = self.select_best_by_reciever(
            node_states[batch.edge_index[0]], scalars, batch.edge_index[1]
        )

        static_fts = self.compute_static_fts(scalars, batch)
        combined = self.combine_fts(
            torch.cat(
                [edge_fts, edge_fts[batch.batched_reverse_idx], static_fts], dim=1
            )
        )

        edge_K = self.edge_key(select_best)
        edge_V = self.edge_value(combined)

        return edge_K, edge_V

class TopKLearnAttentionModule(torch.nn.Module):
    """
    Annealed Hard Top-K Attention.
    
    Mechanism:
    1. Predict 'k' (capacity) per node based on node state.
    2. Sort edge logits to find the value at the k-th position (the threshold).
    3. Use Sigmoid((logit - threshold) / tau) as a soft-to-hard mask.
    
    Annealing (via tau):
    - High Tau: Sigmoids are flat (~0.5), acting like a weighted global average.
    - Low Tau: Sigmoids become binary (0 or 1), acting like Hard Top-K.
    """
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.h = h
        
        # Encoders
        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)
        self.static_fts_encoder = StatesEncoder(h, 2)
        
        # Projections
        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)
        
        self.edge_key = Linear(h, h, bias=False)
        self.edge_value = Linear(h, h, bias=False)
        self.combine_fts = Linear(3 * h, h, bias=False)
        
        # Helpers
        self.select_best_virtual = SelectBest(config)
        self.select_best_by_reciever = SelectBest(config)
        
        # --- Capacity Predictor (The replacement for the Sink) ---
        # Predicts a fraction [0, 1] of neighbors to keep.
        # We initialize bias=1.0 so it starts by selecting ~73% of neighbors (sigmoid(1.0))
        # rather than 50% or 0%, ensuring healthy gradient flow at step 0.
        self.k_predictor = Sequential(
            Linear(h, h),
            ReLU(),
            Linear(h, 1)
        )
        torch.nn.init.constant_(self.k_predictor[-1].bias, 1.0)
        
        self.avg = getattr(config, 'attention', 'hard') == 'gumbel_average'

        self.use_noise = config.use_noise
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        # 1. Pre-process Features
        node_fts = self.select_best_from_virtual(node_states, scalars, batch)
        edge_fts = self.edge_states_encoder(edge_states)
        
        Q = self.lin_query(node_fts)
        K = self.lin_key(node_fts)
        V = self.lin_value(node_fts)
        
        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts, scalars, batch)
        
        # 2. Compute Aggregated Message
        aggregated_message = self.compute_message(
            Q=Q, K=K, V=V,
            edge_K=edge_K, edge_V=edge_V,
            node_fts=node_fts, # Passed for k-prediction
            edge_index=batch.edge_index,
            num_nodes=node_fts.size(0),
            training_step=training_step,
        )
        
        # 3. Update Nodes & Edges
        node_fts = node_fts + aggregated_message
        edge_message = aggregated_message[batch.edge_index[1]]
        edge_fts = edge_fts + edge_message
        
        return node_fts, edge_fts

    def compute_message(self, Q, K, V, edge_K, edge_V, node_fts, edge_index, num_nodes, training_step):
        src_idx, dst_idx = edge_index
        
        # --- 1. Edge Scores ---
        K_combined = K[src_idx] + edge_K
        V_combined = V[src_idx] + edge_V
        
        dense_K, mask = to_dense_batch(K_combined, dst_idx, batch_size=num_nodes)
        dense_V, _    = to_dense_batch(V_combined, dst_idx, batch_size=num_nodes)
        
        Q_expanded = Q.unsqueeze(1)
        # [N, Neighbors]
        edge_logits = (Q_expanded * dense_K).sum(dim=-1).squeeze(1) / math.sqrt(self.h)
        edge_logits = edge_logits.masked_fill(~mask, -1e9)
        
        # --- 2. Predict K (Capacity) ---
        # "How many neighbors should I listen to?"
        # rho \in (0, 1). Using sigmoid allows smooth gradients.
        rho = torch.sigmoid(self.k_predictor(node_fts)) 
        
        degrees = mask.sum(dim=1, keepdim=True).float()
        k_float = rho * degrees
        # Clamp to ensure we select at least 1 and at most N
        k_float = k_float.clamp(min=torch.ones_like(k_float), max=degrees - 0.01)
        
        # --- 3. Determine Threshold ---
        # We need the value of the logit at index 'k'.
        # We use simple sorting. This is efficient enough for GNN neighbor sizes.
        sorted_logits, _ = torch.sort(edge_logits, descending=True, dim=1)
        
        # Get differentiable threshold at index k_float
        threshold = self.get_threshold_at_k(sorted_logits, k_float.squeeze(1))
        
        # --- 4. Gumbel-Sigmoid Selection ---
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1
        
        if use_noise:
            # Gumbel perturbation for stochasticity
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(edge_logits) + 1e-9) + 1e-9)
            logits_to_threshold = edge_logits + gumbel_noise
        else:
            logits_to_threshold = edge_logits

        # The Magic: Sigmoid acts as a soft gate.
        # High Tau -> Sigmoid is flat (values near 0.5) -> Soft Average
        # Low Tau  -> Sigmoid is step (0 or 1)         -> Hard Top-K
        diff = logits_to_threshold - threshold.unsqueeze(1)
        soft_mask = torch.sigmoid(diff / tau)
        
        # Hard Mask for Forward Pass (STE) or Eval
        if training_step == -1:
            selection_mask = (diff >= 0).float()
        else:
            # Straight-Through Estimator
            hard_mask = (diff >= 0).float()
            selection_mask = (hard_mask - soft_mask).detach() + soft_mask

        # Apply padding mask
        selection_mask = selection_mask * mask.float()

        # --- 5. Aggregate ---
        weighted_V = selection_mask.unsqueeze(-1) * dense_V
        sum_V = weighted_V.sum(dim=1)
        
        if self.avg:
            # For Average, we divide by the sum of weights (soft count)
            normalizer = selection_mask.sum(dim=1, keepdim=True) + 1e-9
            return sum_V / normalizer
        else:
            return sum_V

    def get_threshold_at_k(self, sorted_logits, k_float):
        """
        Extracts the value at index k from sorted_logits in a differentiable way.
        Interpolates between sorted_logits[floor(k)] and sorted_logits[ceil(k)].
        """
        # k_float is 1-based count. Index is 0-based.
        # If k=1, we want index 0.
        target_idx = k_float - 1.0
        
        idx_floor = target_idx.floor().long().clamp(min=0, max=sorted_logits.size(1)-1)
        idx_ceil = (idx_floor + 1).clamp(min=0, max=sorted_logits.size(1)-1)
        
        val_floor = torch.gather(sorted_logits, 1, idx_floor.unsqueeze(1))
        val_ceil  = torch.gather(sorted_logits, 1, idx_ceil.unsqueeze(1))
        
        # Interpolation weight
        frac = (target_idx - idx_floor.float()).unsqueeze(1)
        
        # Note: sorted_logits are DESCENDING.
        # If k moves from 1.0 to 2.0, threshold moves from Val[0] down to Val[1].
        threshold = (1.0 - frac) * val_floor + frac * val_ceil
        return threshold.squeeze(1)

    # --- Standard Helpers ---
    def compute_static_fts(self, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        sender_s = node_scalars[batch.edge_index[0]]
        reciever_s = node_scalars[batch.edge_index[1]]
        rlx = scalars < reciever_s
        rlx_d = sender_s + scalars < reciever_s
        fts = torch.cat([rlx, rlx_d], dim=-1).long()
        return self.static_fts_encoder(fts)

    def select_best_from_virtual(self, node_states, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        return self.select_best_virtual(node_states, node_scalars, batch.batch)

    def combined_edge_KV(self, node_states, edge_fts, scalars, batch):
        select_best = self.select_best_by_reciever(
            node_states[batch.edge_index[0]], scalars, batch.edge_index[1]
        )
        static_fts = self.compute_static_fts(scalars, batch)
        combined = self.combine_fts(
            torch.cat(
                [edge_fts, edge_fts[batch.batched_reverse_idx], static_fts], dim=1
            )
        )
        edge_K = self.edge_key(select_best)
        edge_V = self.edge_value(combined)
        return edge_K, edge_V

class TopKGumbelAttentionModule(torch.nn.Module):
    """
    Dynamic Threshold Gumbel Attention (The "Robust Sink").
    
    Replaces explicit Top-K sorting with a learned Threshold.
    - Mechanism: Mask = Sigmoid((Edge_Logits - Threshold) / Tau)
    - Benefit: Allows dynamic 'k' (0 to N) based on neighbor quality.
    - Solves "Blind Node" problem: Node sets quality standard, not count.
    """
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.h = h
        self.avg = getattr(config, 'attention', 'hard') == 'gumbel_average'

        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)
        self.static_fts_encoder = StatesEncoder(h, 2)
        
        # Projections
        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)
        self.edge_key = Linear(h, h, bias=False)
        self.edge_value = Linear(h, h, bias=False)
        self.combine_fts = Linear(3 * h, h, bias=False)
        
        self.select_best_virtual = SelectBest(config)
        self.select_best_by_reciever = SelectBest(config)

        # --- QK Normalization (Crucial for stability) ---
        # Prevents logits from exploding, keeping Threshold meaningful
        self.norm_q = torch.nn.LayerNorm(h)
        self.norm_k = torch.nn.LayerNorm(h)

        # --- Threshold Predictor ---
        # Predicts the minimum logit score required to be accepted.
        self.threshold_gate = Sequential(
            Linear(h, h),
            ReLU(),
            Linear(h, 1)
        )
        
        # CRITICAL INIT: Start with a very LOW threshold (e.g., -5.0).
        # This ensures Mask ~= 1.0 (Keep All) at the start of training.
        # The model starts with Softmax-like behavior and learns to raise the bar.
        torch.nn.init.constant_(self.threshold_gate[-1].bias, -5.0)
        
        self.use_noise = config.use_noise
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        # 1. Standard Setup
        node_fts = self.select_best_from_virtual(node_states, scalars, batch)
        edge_fts = self.edge_states_encoder(edge_states)
        
        # Apply LayerNorm to Query/Key to stabilize dot products
        Q = self.norm_q(self.lin_query(node_fts))
        K = self.norm_k(self.lin_key(node_fts))
        V = self.lin_value(node_fts)
        
        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts, scalars, batch)
        
        # 2. Compute Message
        # Pass node_fts to predict threshold
        aggregated_message = self.compute_message(
            Q, K, V, edge_K, edge_V, node_fts, 
            batch.edge_index, node_fts.size(0), training_step
        )
        
        # 3. Update
        node_fts = node_fts + aggregated_message
        edge_fts = edge_fts + aggregated_message[batch.edge_index[1]]
        
        return node_fts, edge_fts

    def compute_message(self, Q, K, V, edge_K, edge_V, node_fts, edge_index, num_nodes, training_step):
        src_idx, dst_idx = edge_index
        
        # --- 1. Edge Logits ---
        # K is already Normalized. We assume edge_K is roughly same scale.
        K_combined = K[src_idx] + edge_K
        V_combined = V[src_idx] + edge_V
        
        dense_K, mask = to_dense_batch(K_combined, dst_idx, batch_size=num_nodes)
        dense_V, _    = to_dense_batch(V_combined, dst_idx, batch_size=num_nodes)
        
        Q_expanded = Q.unsqueeze(1)
        
        # [N, Neighbors]
        # With LayerNorm, these logits will likely stay within [-10, 10]
        edge_logits = (Q_expanded * dense_K).sum(dim=-1).squeeze(1) / math.sqrt(self.h)
        edge_logits = edge_logits.masked_fill(~mask, -1e9)
        
        # --- 2. Predict Threshold ---
        # "How strong must a neighbor be?"
        # [N, 1]
        threshold = self.threshold_gate(node_fts)
        
        # --- 3. Compute Mask (Gating) ---
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        if use_noise:
            # Gumbel-like noise to logits helps exploration
            noise = -torch.log(-torch.log(torch.rand_like(edge_logits) + 1e-9) + 1e-9)
            noisy_logits = edge_logits + noise
        else:
            noisy_logits = edge_logits
            
        # The Core Logic:
        # Distance from threshold determines probability.
        # If Logit > Threshold: Sigmoid > 0.5 (Keep)
        # If Logit < Threshold: Sigmoid < 0.5 (Drop)
        diff = noisy_logits - threshold
        
        # "Soft" mask (Sigmoid)
        soft_mask = torch.sigmoid(diff / tau)
        
        # "Hard" mask (Step function)
        if training_step == -1:
            selection_mask = (diff >= 0).float()
        else:
            # Leaky STE:
            # We mix in a small amount of Softmax gradients to prevent "Dead Edges"
            # Reuse tau as beta for leak: High Tau = High Leak
            beta = min(max(tau, 0.01), 1.0)
            
            # Calculate a dense softmax distribution for the leak
            softmax_probs = F.softmax(edge_logits, dim=-1) # Use original logits for stability
            softmax_probs = softmax_probs.masked_fill(~mask, 0.0)
            
            # STE: 
            # Forward: Hard Threshold (1.0 or 0.0)
            # Backward: (1-beta)*SigmoidGrad + beta*SoftmaxGrad
            
            hard_mask = (diff >= 0).float()
            
            # Use soft_mask for the main gradient
            grad_proxy = (1.0 - beta) * soft_mask + beta * softmax_probs
            
            selection_mask = (hard_mask - grad_proxy).detach() + grad_proxy

        # Apply padding mask
        selection_mask = selection_mask * mask.float()

        # --- 4. Aggregate ---
        weighted_V = selection_mask.unsqueeze(-1) * dense_V
        sum_V = weighted_V.sum(dim=1)
        
        if self.avg:
            # Average over SELECTED nodes only
            normalizer = selection_mask.sum(dim=1, keepdim=True) + 1e-9
            return sum_V / normalizer
        else:
            # Sum (Parallel BFS needs this!)
            return sum_V

    # Helpers (Standard)
    def compute_static_fts(self, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        sender_s = node_scalars[batch.edge_index[0]]
        reciever_s = node_scalars[batch.edge_index[1]]
        rlx = scalars < reciever_s
        rlx_d = sender_s + scalars < reciever_s
        fts = torch.cat([rlx, rlx_d], dim=-1).long()
        return self.static_fts_encoder(fts)

    def select_best_from_virtual(self, node_states, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        return self.select_best_virtual(node_states, node_scalars, batch.batch)

    def combined_edge_KV(self, node_states, edge_fts, scalars, batch):
        select_best = self.select_best_by_reciever(
            node_states[batch.edge_index[0]], scalars, batch.edge_index[1]
        )
        static_fts = self.compute_static_fts(scalars, batch)
        combined = self.combine_fts(
            torch.cat(
                [edge_fts, edge_fts[batch.batched_reverse_idx], static_fts], dim=1
            )
        )
        edge_K = self.edge_key(select_best)
        edge_V = self.edge_value(combined)
        return edge_K, edge_V


def sparsemax(logits, dim=-1):
    """
    Exact, fast Sparsemax (alpha=2.0).
    Projects logits onto the probability simplex with a linear penalty.
    """
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum_logits = torch.cumsum(sorted_logits, dim=dim)
    
    # Create range [1, 2, ... k] reshaped for broadcasting
    k_range = torch.arange(1, logits.size(dim) + 1, device=logits.device)
    view_shape = [1] * logits.dim()
    view_shape[dim] = -1
    k_range = k_range.view(*view_shape)
    
    # Support condition: 1 + k * z_k > cumsum_k
    support = (k_range * sorted_logits) > (cumsum_logits - 1.0)
    k_indices = support.sum(dim=dim, keepdim=True)
    
    # Calculate Tau
    cumsum_k = torch.gather(cumsum_logits, dim, k_indices - 1)
    tau = (cumsum_k - 1.0) / k_indices.float()
    
    return torch.relu(logits - tau)

def entmax15(logits, dim=-1):
    """
    Exact, fast Entmax (alpha=1.5).
    Projects logits onto the probability simplex with a quadratic penalty.
    """
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum_z = torch.cumsum(sorted_logits, dim=dim)
    cumsum_z2 = torch.cumsum(sorted_logits ** 2, dim=dim)
    
    k_range = torch.arange(1, logits.size(dim) + 1, device=logits.device)
    view_shape = [1] * logits.dim()
    view_shape[dim] = -1
    k_range = k_range.view(*view_shape)
    
    # Calculate Mean and Variance candidates
    mean_z = cumsum_z / k_range
    mean_z2 = cumsum_z2 / k_range
    
    # Discriminant for quadratic formula
    discr = torch.relu(mean_z ** 2 - mean_z2 + (1.0 / k_range))
    tau_candidates = mean_z - torch.sqrt(discr + 1e-8)  # Add eps for numerical stability
    
    # Support condition
    support = sorted_logits > tau_candidates
    k_indices = support.sum(dim=dim, keepdim=True)
    
    # Calculate Tau
    tau = torch.gather(tau_candidates, dim, k_indices - 1)
    
    return torch.relu(logits - tau) ** 2

# ------------------------------------------------------------------------
# ALPHA ENTMAX HARD ATTENTION MODULE
# ------------------------------------------------------------------------

class AlphaEntmaxHardAttention(torch.nn.Module):
    """
    Robust Signal-Dependent Average Hard Attention (Entmax-based).
    
    Uses standard temperature annealing to control the 'Gradient Leak'.
    - Forward: Always discrete/sparse (controlled by u).
    - Backward: Mixes Sparse Gradients with Softmax Gradients based on Tau.
    """
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.h = h
        
        self.avg = getattr(config, 'attention', 'hard') == 'entmax_average'

        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)
        self.static_fts_encoder = StatesEncoder(h, 2)
        
        # Projections
        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)
        
        self.edge_key = Linear(h, h, bias=False)
        self.edge_value = Linear(h, h, bias=False)
        self.combine_fts = Linear(3 * h, h, bias=False)

        self.norm_q = torch.nn.LayerNorm(h)
        self.norm_k = torch.nn.LayerNorm(h)
        self.norm_ke = torch.nn.LayerNorm(h)
        
        # Helpers
        self.select_best_virtual = SelectBest(config)
        self.select_best_by_reciever = SelectBest(config)
        
        # --- Sparsity Gate ---
        self.sparsity_gate = Sequential(
            Linear(h, h),
            ReLU(),
            Linear(h, 1)
        )
        
        # Initialize bias to -5.0 to force start in Softmax Mode (u=0)
        torch.nn.init.constant_(self.sparsity_gate[-1].bias, -5.0)
        
        # --- Annealing Config (Reusing Standard Tau) ---
        self.temp = (
            config.processor_upper_t,  # Start (e.g. 2.0)
            config.processor_lower_t,  # End   (e.g. 0.05)
            config.num_iterations,
            config.temp_on_eval,
        )

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        # 1. Projections
        node_fts = self.select_best_from_virtual(node_states, scalars, batch)
        edge_fts_emb = self.edge_states_encoder(edge_states)
        
        Q = self.norm_q(self.lin_query(node_fts))
        K_nodes = self.norm_k(self.lin_key(node_fts))
        V_nodes = self.lin_value(node_fts)
        
        # 2. Predict Sparsity 'u'
        sparsity_u = torch.sigmoid(self.sparsity_gate(node_fts))
        
        # 3. Dense Batching
        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts_emb, scalars, batch)
        src_idx, dst_idx = batch.edge_index
        
        K_combined = K_nodes[src_idx] + edge_K
        V_combined = V_nodes[src_idx] + edge_V
        
        dense_K, mask = to_dense_batch(K_combined, dst_idx, batch_size=node_states.size(0))
        dense_V, _    = to_dense_batch(V_combined, dst_idx, batch_size=node_states.size(0))
        
        # Add Self-Loops
        self_K = K_nodes.unsqueeze(1)
        self_V = V_nodes.unsqueeze(1)
        
        combined_K = torch.cat([self_K, dense_K], dim=1)
        combined_V = torch.cat([self_V, dense_V], dim=1)
        
        self_mask = torch.ones(node_states.size(0), 1, device=mask.device).bool()
        combined_mask = torch.cat([self_mask, mask], dim=1)
        
        # 4. Compute Logits
        Q_expanded = Q.unsqueeze(1)
        logits = (Q_expanded * combined_K).sum(dim=-1).squeeze(1) / math.sqrt(self.h)
        logits = logits.masked_fill(~combined_mask, -1e9)
        
        # 5. Compute Probabilities (The Forward Sparse Distribution)
        # We rely on 'u' to pick the sparsity level.
        sparse_probs = self.compute_interpolated_probs(logits, sparsity_u)
        
        # 6. Hard Selection for Forward Pass
        # We define "Selection" as having non-zero probability
        is_selected = (sparse_probs > 1e-4).float()
        
        if self.avg:
            num_selected = is_selected.sum(dim=1, keepdim=True)
            hard_weights = is_selected / (num_selected + 1e-9)
        else:
            hard_weights = is_selected
            
        # 7. Leaky STE with Unified Annealing
        # Use existing tau schedule
        tau = temp_by_step(training_step, *self.temp)
        
        # Map Tau to Beta (Leak Amount)
        # If tau >= 1.0, beta = 1.0 (Full Leak/Softmax)
        # If tau -> 0.0, beta -> 0.0 (Pure Sparse)
        # We clamp min to 0.01 to ensure dead edges can always resurrect slightly.
        beta = min(max(tau, 0.01), 1.0)
        
        if training_step == -1:
            # Pure Hard Eval
            attention_weights = hard_weights
        else:
            if beta > 0.01:
                # Calculate Dense Softmax for the leak
                softmax_probs = F.softmax(logits, dim=-1)
                softmax_probs = softmax_probs * combined_mask.float()
                softmax_probs = softmax_probs / (softmax_probs.sum(dim=1, keepdim=True) + 1e-9)
                
                # Mix backward gradients
                grad_probs = (1.0 - beta) * sparse_probs + beta * softmax_probs
            else:
                grad_probs = sparse_probs

            # STE Magic
            # Forward: hard_weights
            # Backward: grad_probs
            attention_weights = (hard_weights - grad_probs).detach() + grad_probs
        
        # Apply mask
        attention_weights = attention_weights * combined_mask.float()
        
        # 8. Aggregate
        aggregated_message = (attention_weights.unsqueeze(-1) * combined_V).sum(dim=1)
        
        # 9. Update
        node_fts = node_fts + aggregated_message
        edge_fts_out = edge_fts_emb + aggregated_message[dst_idx]
        
        return node_fts, edge_fts_out

    def compute_interpolated_probs(self, logits, u):
        # ... (Same as previous implementation) ...
        p_soft = F.softmax(logits, dim=-1)
        p_15 = entmax15(logits, dim=-1)
        p_sparse = sparsemax(logits, dim=-1)
        
        w_low = u * 2.0
        probs_low = (1.0 - w_low) * p_soft + w_low * p_15
        
        w_high = (u - 0.5) * 2.0
        probs_high = (1.0 - w_high) * p_15 + w_high * p_sparse
        
        return torch.where(u <= 0.5, probs_low, probs_high)

    # ... Standard Helpers ...
    def compute_static_fts(self, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        sender_s = node_scalars[batch.edge_index[0]]
        reciever_s = node_scalars[batch.edge_index[1]]
        rlx = scalars < reciever_s
        rlx_d = sender_s + scalars < reciever_s
        fts = torch.cat([rlx, rlx_d], dim=-1).long()
        return self.static_fts_encoder(fts)

    def select_best_from_virtual(self, node_states, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        return self.select_best_virtual(node_states, node_scalars, batch.batch)

    def combined_edge_KV(self, node_states, edge_fts, scalars, batch):
        select_best = self.select_best_by_reciever(
            node_states[batch.edge_index[0]], scalars, batch.edge_index[1]
        )
        static_fts = self.compute_static_fts(scalars, batch)
        combined = self.combine_fts(
            torch.cat(
                [edge_fts, edge_fts[batch.batched_reverse_idx], static_fts], dim=1
            )
        )
        edge_K = self.norm_ke(self.edge_key(select_best))
        edge_V = self.edge_value(combined)
        return edge_K, edge_V

class ScalarUpdater(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h

        self.node_states_encoder = StatesEncoder(config.h, config.num_node_states)
        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)

        self.combine_fts = Linear(2 * h, h)

        self.keep_proj = Linear(h, 2)
        self.push_proj = Linear(h, 2)
        self.push_node_proj = Linear(h, 2)
        self.increment_proj = Linear(h, 2)

        self.scalars_only_as_input = config.generate_random_numbers
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )
        self.use_noise = config.use_noise

    def forward(
        self,
        node_states,
        edge_states,
        scalars,
        batch,
        training_step,
        processor_step,
        teacher_force,
    ):
        if self.scalars_only_as_input:
            return batch.scalars[:, processor_step], 0.0

        node_fts = self.node_states_encoder(node_states)
        edge_fts = self.edge_states_encoder(edge_states)

        fts = self.combine_fts(
            torch.cat(
                [edge_fts[batch.batched_reverse_idx], node_fts[batch.edge_index[0]]],
                dim=1,
            )
        )
        index = torch.repeat_interleave(torch.arange(fts.shape[0]).to(fts.device), 2)

        increment = self.compute_increment(fts, index, training_step)
        push = self.compute_push(fts, scalars.view(-1), batch, index, training_step)
        keep = self.compute_keep(fts, scalars.view(-1), index, training_step)

        new_scalars = torch.unsqueeze(increment + keep + push, -1)

        loss = (
            ((batch.scalars[:, processor_step] - new_scalars) ** 2).mean()
            # if training_step != -1 #enable loss only during training
            # else 0.0
        )

        if teacher_force:
            new_scalars = batch.scalars[:, processor_step]

        return new_scalars, loss

    def compute_increment(self, fts, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        logits = self.increment_proj(fts).view(-1)
        increment = gumbel_softmax(logits, index=index, tau=tau, use_noise=use_noise)[
            ::2
        ]
        return 1.0 * increment

    def compute_push(self, fts, scalars, batch, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        push_without_node_logits = self.push_proj(fts).view(-1)
        push_without_node = gumbel_softmax(
            push_without_node_logits, index=index, tau=tau, use_noise=use_noise
        )[::2]

        push_with_node_logits = self.push_node_proj(fts).view(-1)
        push_with_node = gumbel_softmax(
            push_with_node_logits, index=index, tau=tau, use_noise=use_noise
        )[::2]

        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        scalars_without_node = scalars - node_scalars[batch.edge_index[1]]
        scalars_with_node = scalars_without_node + node_scalars[batch.edge_index[0]]

        edge_push_without_node = scatter(
            push_without_node * scalars_without_node, batch.edge_index[1], reduce="sum"
        )
        edge_push_with_node = scatter(
            push_with_node * scalars_with_node, batch.edge_index[1], reduce="sum"
        )

        accumulated_node = edge_push_without_node + edge_push_with_node
        edge_push = torch.zeros_like(scalars)
        edge_push[batch.edge_index[0] == batch.edge_index[1]] = accumulated_node
        return edge_push

    def compute_keep(self, fts, scalars, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        logits = self.keep_proj(fts).view(-1)
        keep = gumbel_softmax(logits, index=index, tau=tau, use_noise=use_noise)[::2]
        return scalars * keep


class StatesBottleneck(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.node_projections = ModuleList(
            [Linear(h, 1) for _ in range(config.num_node_states)]
        )
        self.edge_projections = ModuleList(
            [Linear(h, 1) for _ in range(config.num_edge_states)]
        )
        self.spec = SPEC[config.algorithm]

    def forward(
        self, node_fts, edge_fts, batch, training_step, processor_step, teacher_force
    ):
        states = []

        loss = 0.0


        #could probably be optimized for vector operations by having a group and projection dimension
        for group in range(2):# group=0: node, group=1: edge
            fts = node_fts if group == 0 else edge_fts
            stacked_fts = []

            projections = self.node_projections if group == 0 else self.edge_projections
            hints = (
                batch.node_fts[:, processor_step]
                if group == 0
                else batch.edge_fts[:, processor_step]
            )

            for idx, projection in enumerate(projections): # projections not a large matrix but n*[hx1] layers
                logits = projection(fts).squeeze()
                gt = hints[:, idx].double() #select the hint to be projected into

                # loss
                # if training_step != -1: #enable loss only during training
                if self.spec[group][idx] != MASK:
                    index = batch.batch if group == 0 else batch.edge_index[0]
                    weight = 1
                    if self.spec[group][idx] == EDGE_MASK_ONE:
                        index = batch.batch[batch.edge_index[0]]
                        num_nodes = (batch.batch == 0).sum()
                        weight = num_nodes
                    ce_loss = weight * node_pointer_loss(logits, gt, index)
                else:
                    ce_loss = binary_cross_entropy_with_logits(logits, gt)

                loss += ce_loss

                # postprocess
                if not teacher_force: #if not forced, we use the model's own predictions for the next step
                    if self.spec[group][idx] != MASK:
                        index = batch.batch if group == 0 else batch.edge_index[0]
                        if self.spec[group][idx] == EDGE_MASK_ONE:
                            index = batch.batch[batch.edge_index[0]]
                        pred = gumbel_softmax(
                            logits, index=index, tau=0.0, use_noise=False
                        )
                    else:
                        pred = 1.0 * (logits > 0.0)
                else:
                    pred = gt
                stacked_fts.append(torch.unsqueeze(pred, -1))
            states.append(torch.cat(stacked_fts, -1))

        return *states, loss


class DiscreteProcessor(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h

        if getattr(config, 'attention', 'hard') in ('gumbel_average', 'gumbel_sum'):
            self.message_passing = TopKGumbelAttentionModule(config)
        elif getattr(config, 'attention','hard') in ('entmax_average', 'entmax_sum'):
            self.message_passing = AlphaEntmaxHardAttention(config)
        elif getattr(config, 'attention', 'hard') in('learn_sum', 'learn_average'):
            self.message_passing = TopKLearnAttentionModule(config)
        else:
            self.message_passing = AttentionModule(config)


        self.node_ffn = Sequential(Linear(h, h), ReLU(), Linear(h, h), ReLU())
        self.edge_ffn = Sequential(Linear(2 * h, h), ReLU(), Linear(h, h), ReLU())

        self.states_bottleneck = StatesBottleneck(config)
        self.scalar_update = ScalarUpdater(config)

    def forward(
        self,
        node_states,
        edge_states,
        scalars,
        batch,
        training_step,
        processor_step,
        teacher_force,
    ):
        node_fts, edge_fts = self.message_passing( #also has the encoder inside it
            node_states, edge_states, scalars, batch, training_step
        )
        node_fts, edge_fts = self.ffn(node_fts, edge_fts, batch)

        node_states, edge_states, states_loss = self.states_bottleneck( #has part of the decoder inside it
            node_fts, edge_fts, batch, training_step, processor_step, teacher_force
        )
        out_scalars, scalars_loss = self.scalar_update( #has part of the decoder inside it
            node_states, #we use the discretized states for the update to be more consistent
            edge_states,
            scalars,
            batch,
            training_step,
            processor_step,
            teacher_force,
        )

        loss = scalars_loss + states_loss

        return node_states, edge_states, out_scalars, loss

    def ffn(self, node_fts, edge_fts, batch):
        node_fts = node_fts + self.node_ffn(node_fts)
        edge_fts_with_reversed = torch.cat(
            [edge_fts, edge_fts[batch.batched_reverse_idx]], dim=1
        )

        edge_fts = edge_fts + self.edge_ffn(edge_fts_with_reversed)
        return node_fts, edge_fts

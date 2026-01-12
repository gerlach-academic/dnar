import math

import torch
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits, softmax
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

class AverageAttentionModule(torch.nn.Module):
    """
    Average Hard Attention with Learnable Dynamic Top-K via Gumbel-Sink mechanism.
    Vectorized using dense batching for efficiency.
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
        
        # Sink Parameters (Learnable)
        # We project the query against this sink parameter to get the sink score
        self.sink_param = Parameter(torch.randn(1, 1, h))
        
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
        
        # Get Edge-augmented K and V
        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts, scalars, batch)
        
        # 2. Compute Aggregated Message via Top-k
        # This returns [Num_Nodes, H]
        aggregated_message = self.compute_average_message(
            Q=Q,
            K=K,
            V=V,
            edge_K=edge_K,
            edge_V=edge_V,
            edge_index=batch.edge_index,
            num_nodes=node_fts.size(0),
            training_step=training_step,
        )
        
        # 3. Update Nodes
        node_fts = node_fts + aggregated_message
        
        # 4. Update Edges
        # Since we computed a node-level average, we broadcast this update 
        # back to the incoming edges to maintain flow consistency.
        # Edges get the message of the receiver they point to.
        edge_message = aggregated_message[batch.edge_index[1]]
        edge_fts = edge_fts + edge_message
        
        return node_fts, edge_fts

    def compute_average_message(self, Q, K, V, edge_K, edge_V, edge_index, num_nodes, training_step):
        """
        Vectorized Gumbel-Top-k Average Attention.
        """
        # 1. Prepare Inputs
        src_idx, dst_idx = edge_index
        
        # Keys and Values at the edges (Node K/V + Edge K/V)
        # Note: K_base is from Source, but we group by Receiver (dst_idx)
        K_combined = K[src_idx] + edge_K
        V_combined = V[src_idx] + edge_V
        
        # 2. Dense Batching
        # Convert sparse edge list to [Num_Nodes, Max_Neighbors, H]
        # This allows us to use torch.topk / sort efficiently
        dense_K, mask = to_dense_batch(K_combined, dst_idx, batch_size=num_nodes)
        dense_V, _    = to_dense_batch(V_combined, dst_idx, batch_size=num_nodes)
        
        # Q: [N, H] -> [N, 1, H]
        Q_expanded = Q.unsqueeze(1)
        
        # 3. Compute Attention Scores (Logits)
        # [N, 1, H] * [N, Max_Deg, H] -> [N, Max_Deg]
        edge_logits = (Q_expanded * dense_K).sum(dim=-1).squeeze(1) / math.sqrt(self.h)
        
        # Mask out padding (ensure padded edges have -inf score)
        edge_logits = edge_logits.masked_fill(~mask, -1e9)
        
        # 4. Sink Logic (Determine k)
        # Sink Logit: [N, 1]
        sink_logit = (Q_expanded * self.sink_param).sum(dim=-1).squeeze(1)
        
        # 5. Gumbel Perturbation
        tau = temp_by_step(training_step, *self.temp)
        if self.use_noise and training_step != -1:
            edge_noise = -torch.log(-torch.log(torch.rand_like(edge_logits) + 1e-9) + 1e-9)
            sink_noise = -torch.log(-torch.log(torch.rand_like(sink_logit) + 1e-9) + 1e-9)
            perturbed_edge_logits = edge_logits + edge_noise
            perturbed_sink_logit = sink_logit + sink_noise
        else:
            perturbed_edge_logits = edge_logits
            perturbed_sink_logit = sink_logit
        # 6. Calculate k (Continuous)

        # Softmax over {Sink, Edges} to find P(Sink)
        all_logits = torch.cat([perturbed_sink_logit.unsqueeze(1), perturbed_edge_logits], dim=1)
        all_mask = torch.cat([torch.ones(num_nodes, 1, device=mask.device).bool(), mask], dim=1)
        all_probs = torch.softmax(all_logits.masked_fill(~all_mask, -1e9) / tau, dim=1)
        
        sink_prob = all_probs[:, 0] # [N]

        n_neighbors = mask.sum(dim=1).float()
        k_float = n_neighbors * (1.0 - sink_prob)
        # Clamp strictly so interpolation is always valid
        k_float = k_float.clamp(min=1.0, max=n_neighbors - 1e-5)
        
        # 7. Sort Logits
        sorted_logits, _ = torch.sort(perturbed_edge_logits, descending=True, dim=1)
        
        # 8. Get Threshold (STE: Hard Forward, Soft Backward)
        threshold = self.get_threshold_with_ste(sorted_logits, k_float)
        
        # 9. Generate Selection Mask
        # Standard Straight-Through Estimator for the Mask itself
        # Forward: 1 if logit >= threshold, 0 else
        # Backward: Sigmoid gradient
        
        soft_mask = torch.sigmoid((perturbed_edge_logits - threshold) / tau)
        
        if training_step == -1:
             # Pure Hard Eval
             selection_scores = (perturbed_edge_logits >= threshold).float()
        else:
             # STE Training
             hard_mask = (perturbed_edge_logits >= threshold).float()
             selection_scores = (hard_mask - soft_mask).detach() + soft_mask

        # 8. Compute Weighted Average
        # [N, Max_Deg, 1] * [N, Max_Deg, H]
        weighted_V = selection_scores.unsqueeze(-1) * dense_V
        sum_V = weighted_V.sum(dim=1) # [N, H]
        
        # Normalizer (number of selected edges)
        normalizer = selection_scores.sum(dim=1, keepdim=True) + 1e-9
        
        return sum_V / normalizer

    # --- Helper methods reused from AttentionModule ---
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
    
    def get_threshold_with_ste(self, sorted_logits, k_float):
        """
        Forward: Returns the logit at index round(k).
        Backward: Returns the gradient from the linear interpolation at index k.
        """
        # 1. Hard Index (Forward pass only)
        k_hard = k_float.round().long()
        # Clamp to valid indices [0, Max_Deg - 1]
        # We subtract 1 because k=1 means index 0
        idx_hard = (k_hard - 1).clamp(min=0, max=sorted_logits.size(1) - 1)
        
        # Gather the hard threshold
        hard_threshold = torch.gather(sorted_logits, 1, idx_hard.unsqueeze(1))
        
        # 2. Soft Index (Backward pass / Gradient approximation)
        # Target float index
        target_idx = k_float - 1.0
        idx_floor = target_idx.floor().long().clamp(min=0)
        idx_ceil = (idx_floor + 1).clamp(max=sorted_logits.size(1) - 1)
        
        # Get values at floor and ceil
        val_floor = torch.gather(sorted_logits, 1, idx_floor.unsqueeze(1))
        val_ceil  = torch.gather(sorted_logits, 1, idx_ceil.unsqueeze(1))
        
        # Interpolate
        frac = target_idx - target_idx.floor() # [N]
        # Note: sorted_logits are DESCENDING. 
        # index 2 is SMALLER than index 1.
        # If k moves 1.0 -> 1.5, we want threshold to move halfway from Val[0] to Val[1].
        soft_threshold = (1.0 - frac.unsqueeze(1)) * val_floor + frac.unsqueeze(1) * val_ceil
        
        # 3. The STE Magic
        # In forward: returns hard_threshold
        # In backward: (hard - soft).detach() is 0, so gradients flow through soft_threshold
        return (hard_threshold - soft_threshold).detach() + soft_threshold


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
    tau_candidates = mean_z - torch.sqrt(discr)
    
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
    Signal-Dependent Average Hard Attention via Sparsity Gating.
    Uses an interpolation of softmax and entmax/sparsemax to create
    a dynamic sparsity level per node.
    
    Dynamically predicts the sparsity level (Alpha) for each node based on its state.
    - Input: Node Features (containing algorithmic hints like 'in_queue').
    - Output: Alpha value per node, interpolating Softmax <-> Entmax <-> Sparsemax.
    """
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.h = h
        
        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)
        self.static_fts_encoder = StatesEncoder(h, 2)
        
        # Projections
        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)
        
        self.edge_key = Linear(h, h, bias=False)
        self.edge_value = Linear(h, h, bias=False)
        self.combine_fts = Linear(3 * h, h, bias=False)
        
        # Processor Helpers
        self.select_best_virtual = SelectBest(config)
        self.select_best_by_reciever = SelectBest(config)
        
        # --- Signal-Dependent Sparsity Gate ---
        # Predicts 'u' from node features. 
        # u -> 0 (Softmax/Global Avg), u -> 1 (Sparsemax/Top-1)
        self.sparsity_gate = Sequential(
            Linear(h, h),
            ReLU(),
            Linear(h, 1)
        )

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        # 1. Get Node Context (Includes 'SelectBest' info like masks)
        node_fts = self.select_best_from_virtual(node_states, scalars, batch)
        edge_fts_emb = self.edge_states_encoder(edge_states)
        
        # 2. Predict Per-Node Sparsity 'u'
        # [N, 1]
        sparsity_u = torch.sigmoid(self.sparsity_gate(node_fts))
        
        # 3. Standard Projections
        Q = self.lin_query(node_fts)
        K_nodes = self.lin_key(node_fts)
        V_nodes = self.lin_value(node_fts)
        
        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts_emb, scalars, batch)
        
        src_idx, dst_idx = batch.edge_index
        K_combined_edges = K_nodes[src_idx] + edge_K
        V_combined_edges = V_nodes[src_idx] + edge_V
        
        # 4. Dense Batching
        dense_K, mask = to_dense_batch(K_combined_edges, dst_idx, batch_size=node_states.size(0))
        dense_V, _    = to_dense_batch(V_combined_edges, dst_idx, batch_size=node_states.size(0))
        
        # 5. Append Self-Loops
        self_K = K_nodes.unsqueeze(1)
        self_V = V_nodes.unsqueeze(1)
        
        combined_K = torch.cat([self_K, dense_K], dim=1)
        combined_V = torch.cat([self_V, dense_V], dim=1)
        
        self_mask = torch.ones(node_states.size(0), 1, device=mask.device).bool()
        combined_mask = torch.cat([self_mask, mask], dim=1)
        
        # 6. Compute Logits
        Q_expanded = Q.unsqueeze(1)
        logits = (Q_expanded * combined_K).sum(dim=-1).squeeze(1) / math.sqrt(self.h)
        logits = logits.masked_fill(~combined_mask, -1e9)
        
        # 7. Compute Probabilities with Vectorized Sparsity 'u'
        probs = self.compute_interpolated_probs(logits, sparsity_u)
        
        # 8. Hard Average + STE
        # Note: We rely on the interpolated 'probs' to drive the learning of 'sparsity_gate'
        # via the backward pass through this STE.
        is_selected = (probs > 1e-6).float()
        num_selected = is_selected.sum(dim=1, keepdim=True)
        hard_weights = is_selected / (num_selected + 1e-9)
        
        attention_weights = (hard_weights - probs).detach() + probs
        attention_weights = attention_weights * combined_mask.float()
        
        # 9. Aggregate
        aggregated_message = (attention_weights.unsqueeze(-1) * combined_V).sum(dim=1)
        
        # 10. Update
        node_fts = node_fts + aggregated_message
        edge_fts_out = edge_fts_emb + aggregated_message[dst_idx]
        
        return node_fts, edge_fts_out

    def compute_interpolated_probs(self, logits, u):
        """
        Vectorized Piecewise Linear Interpolation.
        Args:
            logits: [N, Neighbors]
            u: [N, 1] - The sparsity gate value per node
        """
        # We must calculate components for both branches because 'u' varies per node.
        # Ideally we'd lazy-eval, but torch.where computes both. 
        # Given CLRS graph sizes, this overhead is negligible compared to Sort/TopK.
        
        p_15 = entmax15(logits, dim=-1)
        
        # Branch A: Softmax <-> Entmax (u <= 0.5)
        # Rescale u [0, 0.5] -> w [0, 1]
        w_low = u * 2.0
        p_soft = softmax(logits, dim=-1)
        probs_low = (1.0 - w_low) * p_soft + w_low * p_15
        
        # Branch B: Entmax <-> Sparsemax (u > 0.5)
        # Rescale u [0.5, 1.0] -> w [0, 1]
        w_high = (u - 0.5) * 2.0
        p_sparse = sparsemax(logits, dim=-1)
        probs_high = (1.0 - w_high) * p_15 + w_high * p_sparse
        
        # Select per node
        return torch.where(u <= 0.5, probs_low, probs_high)

    # --- Standard Helpers (Assume defined as in previous context) ---
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

        if getattr(config, 'attention', 'hard') == 'average':
            self.message_passing = AverageAttentionModule(config)
        elif getattr(config, 'attention','hard') == 'entmax':
            self.message_passing = AlphaEntmaxHardAttention(config)
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

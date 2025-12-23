import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import math

# --- Fast Closed-Form Solvers ---

def sparsemax(logits, dim=-1):
    """ Alpha = 2.0 """
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum_logits = torch.cumsum(sorted_logits, dim=dim)
    k_range = torch.arange(1, logits.size(dim) + 1, device=logits.device).view(*([1]*(logits.dim()-1)), -1)
    support = (k_range * sorted_logits) > (cumsum_logits - 1.0)
    k_indices = support.sum(dim=dim, keepdim=True)
    cumsum_k = torch.gather(cumsum_logits, dim, k_indices - 1)
    tau = (cumsum_k - 1.0) / k_indices.float()
    return torch.relu(logits - tau)

def entmax15(logits, dim=-1):
    """ Alpha = 1.5 """
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cumsum_z = torch.cumsum(sorted_logits, dim=dim)
    cumsum_z2 = torch.cumsum(sorted_logits ** 2, dim=dim)
    k_range = torch.arange(1, logits.size(dim) + 1, device=logits.device).view(*([1]*(logits.dim()-1)), -1)
    mean_z = cumsum_z / k_range
    mean_z2 = cumsum_z2 / k_range
    discr = torch.relu(mean_z ** 2 - mean_z2 + (1.0 / k_range))
    tau_candidates = mean_z - torch.sqrt(discr)
    support = sorted_logits > tau_candidates
    k_indices = support.sum(dim=dim, keepdim=True)
    tau = torch.gather(tau_candidates, dim, k_indices - 1)
    return torch.relu(logits - tau) ** 2

# --------------------------------

class MorphingHardAttention(nn.Module):
    """
    Interpolated Hard Attention.
    Learnable alpha in [1.0, 2.0] interpolates between:
    [1.0, 1.5] -> Softmax to Entmax1.5
    [1.5, 2.0] -> Entmax1.5 to Sparsemax
    """
    def __init__(self, config):
        super().__init__()
        self.h = config.h
        
        # Projections
        self.lin_query = nn.Linear(self.h, self.h, bias=False)
        self.lin_key = nn.Linear(self.h, self.h, bias=False)
        self.lin_value = nn.Linear(self.h, self.h, bias=False)
        self.edge_key = nn.Linear(self.h, self.h, bias=False)
        self.edge_value = nn.Linear(self.h, self.h, bias=False)
        
        # Learnable Alpha Control
        # initialized to 0.0 -> sigmoid(0)=0.5 -> alpha=1.5 (Entmax start)
        self.alpha_param = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, node_states, edge_states, batch, training_step):
        # 1. Prepare Inputs (Self-Loop + Neighbor Batching)
        Q = self.lin_query(node_states)
        src_idx, dst_idx = batch.edge_index
        
        # Edge K/V
        edge_K = self.edge_key(edge_states)
        edge_V = self.edge_value(edge_states)
        K_edges = self.lin_key(node_states)[src_idx] + edge_K
        V_edges = self.lin_value(node_states)[src_idx] + edge_V
        
        # Dense Batching
        dense_K, mask = to_dense_batch(K_edges, dst_idx, batch_size=node_states.size(0))
        dense_V, _    = to_dense_batch(V_edges, dst_idx, batch_size=node_states.size(0))
        
        # Self Loops
        self_K = self.lin_key(node_states).unsqueeze(1)
        self_V = self.lin_value(node_states).unsqueeze(1)
        
        # Combine
        combined_K = torch.cat([self_K, dense_K], dim=1)
        combined_V = torch.cat([self_V, dense_V], dim=1)
        combined_mask = torch.cat([torch.ones(len(node_states),1, device=mask.device).bool(), mask], dim=1)
        
        # Logits
        logits = (Q.unsqueeze(1) * combined_K).sum(-1).squeeze(1) / math.sqrt(self.h)
        logits = logits.masked_fill(~combined_mask, -1e9)
        
        # 2. Compute Projections
        # We only compute what we strictly need depending on current alpha could optimize, 
        # but computing all 3 is fast enough and safe for gradients.
        p_soft = F.softmax(logits, dim=-1)
        p_15 = entmax15(logits, dim=-1)
        p_sparse = sparsemax(logits, dim=-1)
        
        # 3. Piecewise Linear Interpolation
        # u in [0, 1] maps to alpha in [1, 2]
        u = torch.sigmoid(self.alpha_param)
        
        # Differentiable Branching using Lerp
        # If u < 0.5: Mix Softmax & Entmax1.5
        # If u > 0.5: Mix Entmax1.5 & Sparsemax
        
        # Rescale u to [0, 1] for the first interval (0 to 0.5) -> w1 = u / 0.5 = 2u
        w_low = 2.0 * u
        probs_low = (1.0 - w_low) * p_soft + w_low * p_15
        
        # Rescale u to [0, 1] for the second interval (0.5 to 1.0) -> w2 = (u - 0.5) / 0.5 = 2u - 1
        w_high = 2.0 * u - 1.0
        probs_high = (1.0 - w_high) * p_15 + w_high * p_sparse
        
        # Select correct branch
        # We use a soft selection or straight 'where' since u is scalar
        if u <= 0.5:
            probs = probs_low
        else:
            probs = probs_high
            
        # 4. Hard Average with STE
        
        # Selection Mask
        # Softmax is never 0, so if u < 0.5, we effectively average ALL neighbors (Mean Pooling).
        # Sparsemax/Entmax are 0, so if u > 0.5, we select subset.
        is_selected = (probs > 1e-6).float()
        num_selected = is_selected.sum(dim=1, keepdim=True)
        hard_weights = is_selected / num_selected
        
        # STE: Forward uses hard_weights, Backward uses interpolated 'probs'
        attention_weights = (hard_weights - probs).detach() + probs
        attention_weights = attention_weights * combined_mask.float()
        
        # 5. Aggregate
        output = (attention_weights.unsqueeze(-1) * combined_V).sum(dim=1)
        
        # Update
        node_fts = node_states + output
        edge_fts = edge_states + output[dst_idx]
        
        return node_fts, edge_fts
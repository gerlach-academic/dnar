import math
import os
import networkx as nx
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset,DataLoader
from torch_geometric.data import Data, Batch
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, List
import atexit
import weakref

from configs import base_config
import argparse

# -----------------------------------------------------------------------------
# 1. PROBLEM INSTANCE WITH PLANNING SUPPORT
# -----------------------------------------------------------------------------

class ProblemInstance:
    def __init__(self, adj, start, goal, weighted, randomness, pos=None, scalar_pos=None):
        self.adj = np.copy(adj)
        self.start = start
        self.goal = goal
        self.weighted = weighted
        self.randomness = np.copy(randomness) if randomness is not None else None
        self.edge_index = np.stack(np.nonzero(adj + np.eye(adj.shape[0])))

        self.out_nodes = [[] for _ in range(adj.shape[0])]
        for x, y in self.edge_index[:, self.edge_index[0] != self.edge_index[1]].T:
            self.out_nodes[x].append(y)
        
        # Store coordinates. 
        # If standard CLRS (ER graph), this is just random noise 1D.
        # If Planning (Grid/Roadmap), this is 2D coords (y,x).
        if pos is not None:
            self.pos = pos
        else:
            n = adj.shape[0]
            random_pos = np.random.uniform(0.0, 1.0, (n,))
            self.pos = random_pos[np.argsort(random_pos)]
        
        # Separate scalar positions for BFS/DFS/Dijkstra/MST hints
        # These must be 1D sorted values for consistent learning across graph types
        if scalar_pos is not None:
            self.scalar_pos = scalar_pos
        elif self.pos.ndim == 1:
            self.scalar_pos = self.pos  # Already 1D sorted (ER graphs)
        else:
            # 2D positions (Grid/Roadmap): generate fresh sorted 1D scalars
            # This ensures BFS layer.sort() correlates with scalar values
            n = adj.shape[0]
            random_1d = np.random.uniform(0.0, 1.0, (n,))
            self.scalar_pos = random_1d[np.argsort(random_1d)]

def push_states(
    node_states, edge_states, scalars, cur_step_nodes, cur_step_edges, cur_step_scalars, edge_index=None
):
    """Push algorithm states for one timestep.
    
    MEMORY OPTIMIZATION: If edge_index is provided, immediately extract edge values
    from n×n matrices instead of storing full matrices. This reduces memory from
    O(n²) to O(E) per timestep, critical for large graphs.
    """
    node_states.append(np.stack(cur_step_nodes, axis=-1))
    
    if edge_index is not None:
        # Extract edge values immediately - don't store full n×n matrices!
        # This is the key optimization: O(E) instead of O(n²) per step
        edge_vals = [m[edge_index[0], edge_index[1]] for m in cur_step_edges]
        edge_states.append(np.stack(edge_vals, axis=-1))
    else:
        # Legacy path (shouldn't be used for large graphs)
        edge_states.append(np.stack(cur_step_edges, axis=-1))
    
    scalars.append(np.stack(cur_step_scalars, axis=-1))

# Helper to keep legacy algorithms (BFS/DFS) from crashing on 2D inputs
def get_scalar_pos_for_legacy(instance):
    """Get 1D sorted scalar positions for BFS/DFS/Dijkstra/MST hints.
    
    This ensures all graph types provide the same scalar hint structure:
    sorted 1D values where lower indices have lower scalar values.
    This matches how ER graphs work and enables cross-graph-type generalization.
    """
    return instance.scalar_pos

# -----------------------------------------------------------------------------
# 2. ALGORITHMS
# -----------------------------------------------------------------------------

def bfs(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    visited = np.zeros(n, dtype=np.int16)
    pointers = np.eye(n, dtype=np.int16)
    self_loops = np.eye(n, dtype=np.int16)

    # Use helper for 2D compatibility
    cur_scalars = get_scalar_pos_for_legacy(instance)[instance.edge_index[0]]

    visited[instance.start] = 1

    push_states(
        node_states, edge_states, scalars,
        (visited,), (pointers, self_loops), (cur_scalars,),
        edge_index=instance.edge_index,
    )

    layer = [instance.start]

    while layer:
        next_layer = []
        layer.sort()
        for node in layer:
            for out in instance.out_nodes[node]:
                if visited[out] == 0:
                    visited[out] = 1
                    next_layer.append(out)
                    assert pointers[out][out] == 1
                    pointers[out][out] = 0
                    pointers[out][node] = 1
        layer = next_layer
        push_states(
            node_states, edge_states, scalars,
            (visited,), (pointers, self_loops), (cur_scalars,),
            edge_index=instance.edge_index,
        )

    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (visited,), (pointers, self_loops), (cur_scalars,),
            edge_index=instance.edge_index,
        )
    return np.array(node_states), np.array(edge_states), np.array(scalars)


def dfs(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    not_in_the_stack = np.ones(n, dtype=np.int32)
    top_of_the_stack = np.zeros(n, dtype=np.int32)
    in_the_stack = np.zeros(n, dtype=np.int32)
    pre_end = np.zeros(n, dtype=np.int32)

    pointers = np.eye(n, dtype=np.int32)
    stack_update = np.zeros((n, n), dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    cur_scalars = get_scalar_pos_for_legacy(instance)[instance.edge_index[0]]

    top_of_the_stack[instance.start] = 1
    not_in_the_stack[instance.start] = 0

    push_states(
        node_states, edge_states, scalars,
        (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
        (pointers, stack_update, self_loops),
        (cur_scalars,),
        edge_index=instance.edge_index,
    )

    # Iterative DFS Stack: (current_node, parent_node, child_iterator_index)
    # We use explicit index tracking to know where to resume in the children list
    stack = []
    stack.append([instance.start, -1, 0]) # Mutable list for in-place index update

    while stack:
        current_node, prev_node, child_idx = stack[-1]
        children = instance.out_nodes[current_node]

        if child_idx < len(children):
            # Advance index for next time
            out = children[child_idx]
            stack[-1][2] += 1
            
            if not_in_the_stack[out]:
                # === 1. PRE-RECURSION (Arrival at out) ===
                in_the_stack[current_node] = 1
                stack_update[current_node][out] = 1
                push_states(
                    node_states, edge_states, scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops), (cur_scalars,),
                    edge_index=instance.edge_index,
                )
                stack_update[current_node][out] = 0
                
                top_of_the_stack[current_node] = 0
                top_of_the_stack[out] = 1
                not_in_the_stack[out] = 0
                pointers[out][current_node] = 1
                pointers[out][out] = 0
                stack_update[out][current_node] = 1
                push_states(
                    node_states, edge_states, scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops), (cur_scalars,),
                    edge_index=instance.edge_index,
                )
                stack_update[out][current_node] = 0
                
                # Push child
                stack.append([out, current_node, 0])
        else:
            # All children processed, return from node
            stack.pop()
            
            # === 3. FINAL NODE UPDATE (Equivalent to end of recurse function) ===
            pre_end[current_node] = 1
            stack_update[current_node][current_node] = 1
            push_states(
                node_states, edge_states, scalars,
                (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                (pointers, stack_update, self_loops), (cur_scalars,),
                edge_index=instance.edge_index,
            )
            stack_update[current_node][current_node] = 0
            
            # === 2. POST-RECURSION (Return to parent) ===
            # We do this AFTER popping child, affecting the PARENT
            if prev_node != -1:
                # Map vars: current_node -> prev_node (parent), out -> current_node (child)
                parent = prev_node
                child = current_node
                
                top_of_the_stack[parent] = 1
                top_of_the_stack[child] = 0
                in_the_stack[parent] = 0
                pre_end[child] = 0
                stack_update[parent][child] = 1
                push_states(
                    node_states, edge_states, scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops), (cur_scalars,),
                    edge_index=instance.edge_index,
                )
                stack_update[parent][child] = 0

    return np.array(node_states), np.array(edge_states), np.array(scalars)


def mst(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    in_queue = np.zeros(n, dtype=np.int32)
    in_tree = np.zeros(n, dtype=np.int32)
    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    # MST uses pos values as the "key", so we copy them.
    # Note: MST on 2D coordinates usually implies Euclidean distance as weights, 
    # but here we follow the CLRS 'Prim' logic where keys are scalars.
    node_scalars = np.copy(get_scalar_pos_for_legacy(instance))

    def compute_current_scalars(node_scalars):
        s = instance.adj[instance.edge_index[0], instance.edge_index[1]]
        s[instance.edge_index[0] == instance.edge_index[1]] = node_scalars
        return s

    in_queue[instance.start] = 1
    node_scalars[instance.start] = 0.0

    push_states(
        node_states, edge_states, scalars,
        (in_queue, in_tree), (pointers, self_loops),
        (compute_current_scalars(node_scalars),),
        edge_index=instance.edge_index,
    )

    for _ in range(1, n):
        # 1e3 penalty pushes non-queue nodes to the end (Simulating Infinity)
        node = np.argsort(node_scalars + (1.0 - in_queue) * 1e3)[0]
        assert in_queue[node] == 1
        in_tree[node] = 1
        in_queue[node] = 0

        for out in instance.out_nodes[node]:
            # Prim's update
            if in_tree[out] == 0 and (
                in_queue[out] == 0 or instance.adj[node][out] < node_scalars[out]
            ):
                pointers[out] = np.zeros(n, dtype=np.int32)
                pointers[out][node] = 1
                node_scalars[out] = instance.adj[node][out]
                in_queue[out] = 1

        push_states(
            node_states, edge_states, scalars,
            (in_queue, in_tree), (pointers, self_loops),
            (compute_current_scalars(node_scalars),),
            edge_index=instance.edge_index,
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)

def mis(instance: ProblemInstance):
    # MIS uses randomness, logic mostly independent of position
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    alive = np.ones(n, dtype=np.int32)
    in_mis = np.zeros(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    def compute_current_scalars():
        # Safeguard index for batching
        idx = min(len(node_states) // 2, len(instance.randomness) - 1)
        random_numbers = instance.randomness[idx]
        return random_numbers[instance.edge_index[0]]

    push_states(
        node_states, edge_states, scalars,
        (in_mis, alive), (self_loops,), (compute_current_scalars(),),
        edge_index=instance.edge_index,
    )
    
    while np.any(alive):
        idx = min(len(node_states) // 2, len(instance.randomness) - 1)
        random_numbers = instance.randomness[idx]

        for node in filter(lambda x: alive[x], range(n)):
            neighbors = instance.adj[node].astype(bool)
            neigh_alive = np.logical_and(neighbors, alive)
            
            if np.any(neigh_alive):
                min_neigh = random_numbers[neigh_alive].min()
            else:
                min_neigh = 2.0
            
            if random_numbers[node] < min_neigh:
                in_mis[node] = 1
            else:
                in_mis[node] = 0

        push_states(
            node_states, edge_states, scalars,
            (in_mis, alive), (self_loops,), (compute_current_scalars(),),
            edge_index=instance.edge_index,
        )

        new_alive = np.copy(alive)
        for node in filter(lambda x: alive[x], range(n)):
            neighbors = instance.adj[node].astype(bool)
            # If self or any neighbor is in MIS, node dies
            if in_mis[node] or np.any(in_mis[np.logical_and(neighbors, in_mis.astype(bool))]):
                new_alive[node] = 0
            else:
                new_alive[node] = 1

        alive = new_alive
        push_states(
            node_states, edge_states, scalars,
            (in_mis, alive), (self_loops,), (compute_current_scalars(),),
            edge_index=instance.edge_index,
        )

    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (in_mis, alive), (self_loops,), (compute_current_scalars(),),
            edge_index=instance.edge_index,
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)

def dijkstra(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    in_queue = np.zeros(n, dtype=np.int32)
    in_tree = np.zeros(n, dtype=np.int32)

    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    # FIX: For geometric graphs, pos is 2D but we need 1D scalars for g-scores
    # Use a flat copy for the actual g-score tracking
    if instance.pos.ndim == 1:
        node_scalars = np.copy(instance.pos)  # 1D: use directly
    else:
        # 2D: initialize with x-coordinates (will be overwritten anyway)
        node_scalars = np.copy(instance.pos[:, 0])

    def compute_current_scalars(node_scalars):
        scalars = instance.adj[instance.edge_index[0], instance.edge_index[1]]
        scalars[instance.edge_index[0] == instance.edge_index[1]] = node_scalars
        return scalars

    in_queue[instance.start] = 1
    node_scalars[instance.start] = 0

    push_states(
        node_states,
        edge_states,
        scalars,
        (in_queue, in_tree),
        (pointers, self_loops),
        (compute_current_scalars(node_scalars),),
        edge_index=instance.edge_index,
    )

    for _ in range(1, n):
        node = np.argsort(node_scalars + (1.0 - in_queue) * 1e3)[0]
        assert in_queue[node] == 1

        in_tree[node] = 1
        in_queue[node] = 0

        for out in instance.out_nodes[node]:
            if in_tree[out] == 0 and (
                in_queue[out] == 0
                or node_scalars[node] + instance.adj[node][out] < node_scalars[out]
            ):
                pointers[out] = np.zeros(n, dtype=np.int32)
                pointers[out][node] = 1
                node_scalars[out] = node_scalars[node] + instance.adj[node][out]
                in_queue[out] = 1

        push_states(
            node_states,
            edge_states,
            scalars,
            (in_queue, in_tree),
            (pointers, self_loops),
            (compute_current_scalars(node_scalars),),
            edge_index=instance.edge_index,
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)

def a_star(instance: ProblemInstance, build_full_tree: bool = True, pad_len:Optional[int]=None):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    target_len = pad_len if pad_len is not None else n

    # --- 1. Masks & Pointers ---
    in_open = np.zeros(n, dtype=np.int32)
    in_closed = np.zeros(n, dtype=np.int32)
    is_goal = np.zeros(n, dtype=np.int32)
    is_goal[instance.goal] = 1
    
    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    
    # --- 2. Scaling Factor Calculation ---
    edge_weights = instance.adj[instance.edge_index[0], instance.edge_index[1]]
    avg_weight = np.mean(edge_weights)
    scale_factor = 1.0
    if avg_weight > 0.5: 
        scale_factor = 1.0 / math.sqrt(n)

    # --- 3. Heuristics ---
    if instance.pos.ndim == 2:
        h_vals = np.linalg.norm(instance.pos - instance.pos[instance.goal], axis=1)
    else:
        h_vals = np.abs(instance.pos - instance.pos[instance.goal])
    
    h_start = h_vals[instance.start]

    # --- 4. Initialization ---
    g_score = np.zeros(n, dtype=np.float32)
    discovered = np.zeros(n, dtype=bool)
    
    # Random initialization for undiscovered nodes
    random_pos = np.random.uniform(0.0, 1.0, (n,))
    init_random = random_pos[np.argsort(random_pos)].astype(np.float32)
    
    # --- 4. HINT PHYSICS SETUP ---
    h_src = h_vals[instance.edge_index[0]]
    h_dst = h_vals[instance.edge_index[1]]
    w_uv = instance.adj[instance.edge_index[0], instance.edge_index[1]]
    
    edge_hint_val = (w_uv - h_src + h_dst) * scale_factor

    # --- 6. NODE SCALAR COMPUTATION ---
    def compute_current_scalars(g_curr, disc_mask):
        s = np.copy(edge_hint_val)
        mask_loops = instance.edge_index[0] == instance.edge_index[1]
        
        if mask_loops.sum() != n:
             raise ValueError(f"Graph Error: Found {mask_loops.sum()} self-loops, expected {n}")

        # For DISCOVERED nodes: d'(n) = g(n) + h(n) - h(start)
        # For UNDISCOVERED nodes: use random init (like Dijkstra)
        d_prime_discovered = (g_curr + h_vals - h_start) * scale_factor
        
        # Blend: discovered nodes get proper d', undiscovered get random
        node_scalars = np.where(disc_mask, d_prime_discovered, init_random)
        
        s[mask_loops] = node_scalars[instance.edge_index[0][mask_loops]]
        return s

    # Setup Start Node
    g_score[instance.start] = 0
    discovered[instance.start] = True
    in_open[instance.start] = 1

    # Initial Push
    push_states(
        node_states, edge_states, scalars,
        (in_open, in_closed, is_goal), (pointers, self_loops),
        (compute_current_scalars(g_score, discovered),),
        edge_index=instance.edge_index,
    )

    for _ in range(n):
        # --- 7. SELECTION LOGIC: f-score primary, node index secondary ---
        f_scores = g_score + h_vals
        
        # Create selection key: (f_score, node_index)
        # Nodes not in open set get infinite f-score
        open_mask = in_open == 1
        
        if not np.any(open_mask):
            break
        
        # Among open nodes, find minimum f-score
        f_open = np.where(open_mask, f_scores, np.inf)
        min_f = np.min(f_open)
        
        # Among nodes with minimum f-score, pick lowest index
        # This is the key change: deterministic tie-breaking by index
        candidates = np.where((f_open == min_f) & open_mask)[0]
        current = candidates[0]  # Lowest index among tied nodes
        
        # --- 8. TOGGLE LOGIC ---
        if not build_full_tree and current == instance.goal:
            in_open[current] = 0
            in_closed[current] = 1
            push_states(
                node_states, edge_states, scalars,
                (in_open, in_closed, is_goal), (pointers, self_loops),
                (compute_current_scalars(g_score, discovered),),
                edge_index=instance.edge_index,
            )
            break

        in_open[current] = 0
        in_closed[current] = 1

        for neighbor in instance.out_nodes[current]:
            if in_closed[neighbor]: continue

            tentative_g = g_score[current] + instance.adj[current][neighbor]
            
            if in_open[neighbor] == 0 or tentative_g < g_score[neighbor]:
                pointers[neighbor] = np.zeros(n, dtype=np.int32)
                pointers[neighbor][current] = 1
                g_score[neighbor] = tentative_g
                discovered[neighbor] = True
                in_open[neighbor] = 1
        
        push_states(
            node_states, edge_states, scalars,
            (in_open, in_closed, is_goal), (pointers, self_loops),
            (compute_current_scalars(g_score, discovered),),
            edge_index=instance.edge_index,
        )

    # Pad to target length
    while len(node_states) < target_len:
        push_states(
            node_states, edge_states, scalars,
            (in_open, in_closed, is_goal), (pointers, self_loops),
            (compute_current_scalars(g_score, discovered),),
            edge_index=instance.edge_index,
        )
        
    return np.array(node_states), np.array(edge_states), np.array(scalars)

def eccentricity(instance: ProblemInstance):
    """
    Compute the eccentricity of the source node using a flood-echo algorithm.
    
    Eccentricity is the maximum shortest path distance from the source to any other node.
    This implementation follows SALSA-CLRS exactly:
    - Flood phase: BFS from source, building a tree and propagating distances
    - Echo phase: Leaves propagate their max distance back up the tree to source
    
    Node states (matching SALSA-CLRS hints):
        - visited: 1 if node has been visited during flood phase
        - msg_phase: 1 if node is in echo phase (propagating max distance back)
        - flood_state: distance from source (received during flood)
        - echo_state: max distance seen (propagated during echo)
        - leaf: 1 if node is a leaf (has sent echo message)
        
    Edge states:
        - tree: tree edges built during flood phase
        - self_loops: self-loop markers
        
    Scalars:
        - eccentricity estimate (echo_state at source) on self-loops

    NOTE: Instead of random scalar positions for the index order, we really solely on node indices already present, the model learns to use those as IDs.  
    """
    n = instance.adj.shape[0]
    A = instance.adj  # Adjacency matrix
    source = instance.start
    
    node_states = []
    edge_states = []
    scalars = []
    
    # Node states (matching SALSA-CLRS)
    flood_state = np.zeros(n, dtype=np.int32)
    echo_state = np.zeros(n, dtype=np.int32)
    msg_phase = np.zeros(n, dtype=np.int32)  # 0 = flood, 1 = echo
    tree = np.zeros((n, n), dtype=np.int32)  # tree[parent, child] = 1
    visited = np.zeros(n, dtype=np.int32)
    node_is_leaf = np.zeros(n, dtype=np.int32)
    
    # For edge states representation
    self_loops = np.eye(n, dtype=np.int32)
    
    def tree_to_pointers():
        """Convert tree[parent, child] to pointer representation for edge_states."""
        # Each node points to its parent (or self if root/unvisited)
        pointers = np.eye(n, dtype=np.int32)
        for child in range(n):
            for parent in range(n):
                if tree[parent, child]:
                    pointers[child] = np.zeros(n, dtype=np.int32)
                    pointers[child][parent] = 1
                    break
        return pointers
    
    def compute_current_scalars():
        # Align with CLRS hints: floodstate_h and echostate_h
        # We combine them into the single scalar channel via max to provide local supervision
        s = np.zeros(len(instance.edge_index[0]), dtype=np.float32)
        node_vals = np.maximum(flood_state, echo_state)
        for i, (src, dst) in enumerate(zip(instance.edge_index[0], instance.edge_index[1])):
            if src == dst:
                s[i] = node_vals[src]
        return s
    
    def send_flood_msg(node, msg, messages, next_tree):
        """Send flood message to unvisited neighbors, returns True if node is a leaf."""
        is_leaf = True
        for neighbor in range(n):
            if A[node, neighbor] and not visited[neighbor]:
                messages[neighbor] = max(msg, messages[neighbor])
                next_tree[node, neighbor] = 1
                is_leaf = False
        return is_leaf
    
    def send_echo_msg(node, msg, next_echo_state, next_tree, next_msg_phase):
        """Send echo message back to parent."""
        for neighbor in range(n):
            if tree[neighbor, node] and visited[neighbor]:
                next_echo_state[neighbor] = max(msg, next_echo_state[neighbor])
                next_tree[neighbor, node] = 0  # Remove tree edge after echo
                next_msg_phase[neighbor] = 1  # Parent enters echo phase
    
    # Initial state
    push_states(
        node_states, edge_states, scalars,
        (visited, msg_phase), (tree_to_pointers(), self_loops),
        (compute_current_scalars(),),
        edge_index=instance.edge_index,
    )
    
    done = False
    while not done:
        next_visited = visited.copy()
        next_msg_phase = msg_phase.copy()
        next_tree = tree.copy()
        messages = np.zeros(n, dtype=np.int32)
        next_echo_state = echo_state.copy()
        
        for node in range(n):
            if not msg_phase[node]:
                is_leaf = False
                
                # Flood start from source
                if node == source and not visited[node]:
                    next_visited[node] = 1
                    is_leaf = send_flood_msg(node, 1, messages, next_tree)
                
                # Flood propagation
                if flood_state[node] > 0 and not visited[node]:
                    next_visited[node] = 1
                    is_leaf = send_flood_msg(node, flood_state[node] + 1, messages, next_tree)
                
                # If leaf, switch to echo phase
                if is_leaf:
                    node_is_leaf[node] = 1
                    next_echo_state[node] = flood_state[node]
                    next_msg_phase[node] = 1
                    send_echo_msg(node, flood_state[node], next_echo_state, next_tree, next_msg_phase)
            else:
                if node_is_leaf[node]:
                    continue
                    
                # Echo phase: check if all children have echoed back
                is_leaf = True
                for neighbor in range(n):
                    # Check if there's still an outgoing tree edge (child hasn't echoed)
                    if tree[node, neighbor] and not (tree[neighbor, node] and tree[node, neighbor]):
                        is_leaf = False
                
                if is_leaf:
                    if node == source:
                        done = True
                        break
                    node_is_leaf[node] = 1
                    # Send echo back to parent
                    send_echo_msg(node, echo_state[node], next_echo_state, next_tree, next_msg_phase)
        
        visited = next_visited
        msg_phase = next_msg_phase
        tree = next_tree
        
        # Receive flood messages
        for node in range(n):
            if messages[node] > 0:
                if not visited[node]:
                    flood_state[node] = messages[node]
                else:
                    # Node already visited but received message - check if now a leaf
                    is_leaf = True
                    for neighbor in range(n):
                        if tree[node, neighbor] and not (tree[neighbor, node] and tree[node, neighbor]):
                            is_leaf = False
                    if is_leaf:
                        node_is_leaf[node] = 1
                        next_echo_state[node] = flood_state[node]
                        next_msg_phase[node] = 1
                        send_echo_msg(node, flood_state[node], next_echo_state, next_tree, next_msg_phase)
        
        echo_state = next_echo_state
        
        push_states(
            node_states, edge_states, scalars,
            (visited, msg_phase), (tree_to_pointers(), self_loops),
            (compute_current_scalars(),),
            edge_index=instance.edge_index,
        )
    
    # Pad to n steps
    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (visited, msg_phase), (tree_to_pointers(), self_loops),
            (compute_current_scalars(),),
            edge_index=instance.edge_index,
        )
    
    return np.array(node_states), np.array(edge_states), np.array(scalars)


# -----------------------------------------------------------------------------
# 3. SAMPLERS
# -----------------------------------------------------------------------------

def embed_positions_from_weights(adj):
    # Shortest-path Distanzen berechnen (Floyd-Warshall approximation)
    G = nx.from_numpy_array(adj)
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(G))
    
    n = len(adj)
    dist_full = np.full((n, n), np.inf)
    for i in dist_dict:
        for j in dist_dict[i]:
            dist_full[i, j] = dist_dict[i][j]
    
    # MDS: Finde 2D-Positionen die diese Distanzen approximieren
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos = mds.fit_transform(dist_full)
    
    return pos

def er_probabilities(n):
    base = math.log(n) / n
    return (base, base * 3)

class ErdosRenyiGraphSampler:
    def __init__(self, config):
        self.weighted = config.edge_weights
        self.generate_random_numbers = config.generate_random_numbers

        self.for_astar = (config.algorithm == 'a_star')

    def __call__(self, num_nodes):
        p_segment = er_probabilities(num_nodes)
        p = p_segment[0] + np.random.rand() * (p_segment[1] - p_segment[0])
        
        start = np.random.randint(0, num_nodes)
        
        while True:
            # 1. Generate Topology
            adj = np.triu(np.random.binomial(1, p, size=(num_nodes, num_nodes)), k=1)
            adj += adj.T

            # 2. Generate Positions (Needed for A* Heuristic consistency)
            # For ER, we simulate a 1D line embedding to keep it simple but consistent
            random_pos = np.random.uniform(0.0, 1.0, (num_nodes,))
            pos = random_pos[np.argsort(random_pos)] # Sort to make 1D structure coherent

            if self.weighted:
                if self.for_astar:
                    # FIX FOR A*: Force geometric weights based on 1D position.
                    # w(u,v) = |pos(u) - pos(v)|
                    # This guarantees triangle inequality and non-negative hints.
                    dist_matrix = np.abs(pos[:, None] - pos[None, :])
                    w = np.triu(dist_matrix * (adj > 0), k=1)
                    adj = w + w.T
                    pos = embed_positions_from_weights(adj) # Re-embed in 2D for A*
                else:
                    # Standard Random Weights (Fine for Dijkstra/MST/BFS)
                    w = np.triu(np.random.uniform(0.0, 1.0, (num_nodes, num_nodes)), k=1)
                    w *= adj
                    adj = w + w.T
            
            random_numbers = None
            if self.generate_random_numbers:
                random_numbers = np.random.rand(num_nodes, num_nodes)

            goal = (start + 1) % num_nodes # Dummy
            
            # Connectivity check (Scipy) - Faster than BFS simulation
            n_components = connected_components(csr_matrix(adj), directed=False, return_labels=False)
            if n_components == 1:
                return ProblemInstance(adj, start, goal, self.weighted, random_numbers, pos=pos)

class GeometricGraphSampler:
    """
    Generates Random Geometric Graphs (RGG) with 3NN connectivity.
    Nodes are placed randomly in 2D space [0,1]^2.
    Edges are created between nodes closer than a radius 'r'.
    Edge weights are the Euclidean distance.
    """
    def __init__(self, config):
        self.weighted = True  # A* requires weighted edges
        self.generate_random_numbers = config.generate_random_numbers
        
    def __call__(self, num_nodes):
        # 1. Generate random coordinates in [0, 1]
        pos = np.random.rand(num_nodes, 2)
        
        # 2. Calculate Distance Matrix
        dist_mat = distance_matrix(pos, pos)
        
        # 3. Connect nodes based on a radius threshold
        # r = sqrt(2 * log(n) / n) is the theoretical threshold for connectivity
        radius = math.sqrt(2 * math.log(num_nodes) / num_nodes)
        
        # Create adjacency matrix: 0 if far, dist if close
        adj = np.zeros((num_nodes, num_nodes))
        mask = (dist_mat < radius) & (dist_mat > 0)
        adj[mask] = dist_mat[mask]
        
        # 4. Enforce Connectivity (k-NN fallback)
        # Random radius graphs often have isolated nodes. 
        # We connect every node to its 'k' nearest neighbors to ensure 
        # the graph is navigable for A*.
        k = 3
        # Get indices of k nearest neighbors (excluding self at col 0)
        knn_indices = np.argsort(dist_mat, axis=1)[:, 1:k+1]
        
        for i in range(num_nodes):
            for neighbor in knn_indices[i]:
                d = dist_mat[i, neighbor]
                adj[i, neighbor] = d
                adj[neighbor, i] = d # Undirected

        # 5. Extract Largest Component
        # Even with k-NN, we might have disjoint islands.
        G = nx.from_numpy_array(adj)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
        # Relabel nodes to 0..N_sub-1
        G = nx.convert_node_labels_to_integers(G)
        final_n = G.number_of_nodes()
        
        # If the graph ended up too small (e.g., < 50% of requested size), retry
        if final_n < num_nodes * 0.5:
            return self.__call__(num_nodes)

        # 6. Extract final data
        final_adj = nx.to_numpy_array(G)


        #==========
        # NOISE BREAKING: NO LONGER NEEDED, A_STAR NOW BREAKS TIES VIA NODE_INDICES
        #==========
        # #add tiny bit of noise to break symmetries
        # if final_n > 1:
        #     noise = np.random.uniform(1.0001, 1.001, size=final_adj.shape) #use additive noise, otherwise weights can become zero in the original
        #     final_adj = final_adj * noise
        #     final_adj = (final_adj + final_adj.T) / 2 # Ensure symmetry
        
        # We need to map the new indices back to the original positions
        # Since we just used convert_node_labels_to_integers on a subgraph,
        # we need to be careful. The easiest way is to pass 'pos' into 
        # the graph attributes before sub-graphing.
        node_idx_map = list(largest_cc) # The original indices
        final_pos = pos[node_idx_map]
        
        # 7. Select Start and Goal
        # For A* training to be effective, start and goal should be reasonably far apart
        start = np.random.randint(0, final_n)
        goal = np.random.randint(0, final_n)
        
        # Ensure start != goal and maybe ensure they aren't directly connected (optional)
        while start == goal:
            goal = np.random.randint(0, final_n)

        random_numbers = None
        if self.generate_random_numbers:
            random_numbers = np.random.rand(final_n, final_n)

        # Return the ProblemInstance compatible with your code
        # Pass final_pos so the A* function can calculate h(n)
        return ProblemInstance(
            adj=final_adj, 
            start=start, 
            goal=goal, 
            weighted=True, 
            randomness=random_numbers, 
            pos=final_pos
        )

class GridGraphSampler:
    def __init__(self, config):
        self.weighted = True 
        self.generate_random_numbers = config.generate_random_numbers
        
    def __call__(self, num_nodes):
        side = int(math.sqrt(num_nodes))
        G = nx.grid_2d_graph(side, side)
        
        # Remove ~20% nodes
        nodes_to_remove = [n for n in G.nodes() if np.random.rand() < 0.2]
        G.remove_nodes_from(nodes_to_remove)
        
        if len(G) == 0: return self.__call__(num_nodes)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
        G = nx.convert_node_labels_to_integers(G, label_attribute='pos_tuple')
        N = G.number_of_nodes()
        
        pos_dict = nx.get_node_attributes(G, 'pos_tuple')
        pos_arr = np.array([pos_dict[i] for i in range(N)], dtype=np.float32)
        pos_arr = pos_arr / np.max(pos_arr) # Normalize to [0,1]

        #scale this so that training is stable
        scale = 1.0 / side
        adj = nx.to_numpy_array(G, weight=None) 
        adj[adj > 0] = scale 

        #==========
        # NOISE BREAKING: NO LONGER NEEDED, A_STAR NOW BREAKS T
        #==========
        # #add tiny bit of noise to break symmetries
        # if N > 1:
        #     noise = np.random.uniform(1.0001, 1.001, size=adj.shape)
        #     adj = adj * noise
        #     adj = (adj + adj.T) / 2 # Ensure symmetry

        start = np.random.randint(0, N)
        goal = np.random.randint(0, N)
        while start == goal:
             goal = np.random.randint(0, N)

        random_numbers = None
        if self.generate_random_numbers:
            random_numbers = np.random.rand(N, N)

        return ProblemInstance(
            adj, start, goal, True, 
            random_numbers,
            pos=pos_arr
        )

class RoadmapGraphSampler:
    def __init__(self, config):
        self.weighted = True
        self.generate_random_numbers = config.generate_random_numbers
        
    def __call__(self, num_nodes):
        # Geometric Graph
        radius = math.sqrt(2 * math.log(num_nodes) / num_nodes)
        pos = np.random.rand(num_nodes, 2)
        
        dist = distance_matrix(pos, pos)
        adj = np.zeros_like(dist)
        mask = (dist < radius) & (dist > 0)
        adj[mask] = dist[mask]
        
        G = nx.from_numpy_array(adj)
        if nx.is_connected(G):
             final_pos = pos
        else:
             largest_cc = max(nx.connected_components(G), key=len)
             final_G = G.subgraph(largest_cc).copy()
             final_G = nx.convert_node_labels_to_integers(final_G)
             # Retry if too small
             if final_G.number_of_nodes() < num_nodes * 0.5:
                 return self.__call__(num_nodes)
             
             old_indices = list(largest_cc)
             final_pos = pos[old_indices]
             adj = nx.to_numpy_array(final_G)
             
        N = adj.shape[0]

        #==========
        # NOISE BREAKING: NO LONGER NEEDED, A_STAR NOW BREAKS T
        #==========
        # Add tiny noise to break symmetries
        # if N > 1:
        #     noise = np.random.uniform(1.00001, 1.001, size=adj.shape)
        #     adj = adj * noise
        #     adj = (adj + adj.T) / 2 # Ensure symmetry

        start = np.random.randint(0, N)
        goal = np.random.randint(0, N)
        while start == goal: goal = np.random.randint(0, N)

        random_numbers = None
        if self.generate_random_numbers:
            random_numbers = np.random.rand(N, N)

        return ProblemInstance(adj, start, goal, True, random_numbers, pos=final_pos)


# -----------------------------------------------------------------------------
# 4. CONFIG & SPECS
# -----------------------------------------------------------------------------

MASK = 0
NODE_POINTER = 1
EDGE_MASK_ONE = 2
NODE_MASK_ONE = 3

#THE TYPES OF HINTS TO PREDICT: (NODE_HINTS, EDGE_HINTS)=(NODE_MASKS, EDGE_MASKS/POINTERS)
SPEC = {}
SPEC["bfs"] = ((MASK,), (NODE_POINTER, NODE_POINTER))
SPEC["dfs"] = (
    (MASK, NODE_MASK_ONE, MASK, MASK),
    (NODE_POINTER, EDGE_MASK_ONE, NODE_POINTER),
)
SPEC["mst"] = ((MASK, MASK), (NODE_POINTER, NODE_POINTER))
SPEC["dijkstra"] = ((MASK, MASK), (NODE_POINTER, NODE_POINTER))
SPEC["mis"] = ((MASK, MASK, MASK, MASK), (NODE_POINTER,))
# A*: in_open, in_closed, is_goal (3 node masks) -> Pred, Self-loop (2 edge pointers/masks)
SPEC["a_star"] = ((MASK, MASK, MASK), (NODE_POINTER, NODE_POINTER))
# Eccentricity: visited, msg_phase (2 node masks) -> tree pointers, self-loops (2 edge pointers)
SPEC["eccentricity"] = ((MASK, MASK), (NODE_POINTER, NODE_POINTER))

MAX_NODE_STATES = max(len(s[0]) for s in SPEC.values())
MAX_EDGE_STATES = max(len(s[1]) for s in SPEC.values())

ALGORITHMS = {
    "bfs": bfs, #==breadth first search
    "dfs": dfs, #==depth first search
    "mst": mst, #==PRIM as that is the algorithm used
    "dijkstra": dijkstra, #==SP as that is the algorithm usedf
    "mis": mis, #==maximum independent set
    "a_star": a_star, #TOOD: for a_star we would need to implement edge based reasoning so it can properly compare edges? no relaxation already possible
    "eccentricity": eccentricity, #==eccentricity of source node (max shorttest distance to reach any other node)
}
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
from threading import Thread, Lock
import os

def _pad_features(tensor: torch.Tensor, target_features: int) -> torch.Tensor:
    """Pad the feature dimension (last dim) of a tensor with zeros."""
    current_features = tensor.shape[-1]
    if current_features >= target_features:
        return tensor
    pad_size = target_features - current_features
    padding = torch.zeros(*tensor.shape[:-1], pad_size, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-1)

# Pure Python function that processes a CHUNK of samples
def _worker_logic_chunk(args):
    """Process a chunk of samples in a single worker."""
    try:
        base_seed, config_dict, split, target_node_states, target_edge_states, num_samples_in_chunk = args
        
        # 1. Reconstruct Config
        class SimpleConfig:
            def __init__(self, d): self.__dict__ = d
        config = SimpleConfig(config_dict)

        # 2. Initialize Sampler (once per chunk)
        graph_type = getattr(config, 'graph_type', 'er')
        if graph_type == 'grid':
            sampler = GridGraphSampler(config)
        elif graph_type == 'roadmap':
            sampler = RoadmapGraphSampler(config)
        elif graph_type == 'geometric':
            sampler = GeometricGraphSampler(config)
        else:
            sampler = ErdosRenyiGraphSampler(config)
        
        algorithm = ALGORITHMS[config.algorithm]
        
        results = []
        for i in range(num_samples_in_chunk):
            try:
                # Unique seed per sample
                np.random.seed(base_seed + i)

                # Generate Instance
                instance = sampler(config.problem_size[split])

                # Run Algorithm
                if config.algorithm == 'a_star':
                    node_fts, edge_fts, scalars = algorithm(
                        instance, 
                        build_full_tree=getattr(config, "full_tree", True), 
                        pad_len=config.problem_size[split]
                    )
                else:
                    node_fts, edge_fts, scalars = algorithm(instance)

                # Format Data (Numpy-side)
                # edge_fts is now already edge-indexed (T, E, F) thanks to optimized push_states
                edge_index = instance.edge_index
                node_fts = np.transpose(node_fts, (1, 0, 2))  # (N, T, F)
                edge_fts_selected = np.transpose(edge_fts, (1, 0, 2))  # Already (E, T, F)
                scalars = np.transpose(scalars, (1, 0, 2))

                results.append({
                    'node_fts': node_fts,
                    'edge_fts': edge_fts_selected,
                    'scalars': scalars,
                    'edge_index': edge_index,
                    'pos': instance.pos,
                    'goal': instance.goal,
                    'start': instance.start,
                    'output_type': config.output_type,
                    'output_idx': config.output_idx,
                    't_node': target_node_states,
                    't_edge': target_edge_states
                })
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
    except Exception as e:
        return [{'error': str(e)}]


def create_dataloader_distributed(config, split: str, seed: int, num_workers: int = None):
    """
    Generates dataset using multiprocessing with chunked processing for efficiency.
    Each worker processes multiple samples to reduce overhead.
    """
    num_samples = config.num_samples[split]
    
    # Determine number of workers (default: use 1/3 of CPUs to be cluster-friendly)
    if num_workers is None:
        num_workers = max(1, os.cpu_count() // 4)
    else:
        num_workers = min(num_workers, os.cpu_count()//4)

    target_node_states = max(config.num_node_states, MAX_NODE_STATES)
    target_edge_states = max(config.num_edge_states, MAX_EDGE_STATES)
    config_dict = vars(config) if hasattr(config, '__dict__') else config.__dict__
    num_workers=1
    if num_workers>1:
        # Adaptive chunk sizing based on graph size to limit memory per worker
        # Each sample at size n uses ~(n * n * n * 4 * 2 / 1e9) GB for edge_fts before optimization
        # After optimization: ~(n * E * 2 * 4 / 1e9) GB where E ≈ n * log(n) * 3
        problem_size = config.problem_size[split]
        estimated_edges = int(problem_size * math.log(problem_size) * 3)
        bytes_per_sample = (
            problem_size * problem_size * 4 * 4 +  # node_fts: (N, T, F)
            estimated_edges * problem_size * 2 * 4  # edge_fts: (E, T, F)
        )
        gb_per_sample = bytes_per_sample / 1e9
        
        # Target max ~500MB per worker to stay safe
        max_samples_per_chunk = max(1, int(0.5 / gb_per_sample))
        
        # Calculate chunks needed
        chunk_size = min(max_samples_per_chunk, (num_samples + num_workers - 1) // num_workers)
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        
        # Build task arguments
        print(f"Generating {num_samples} samples with {num_workers} workers")
        print(f"  Graph size: {problem_size}, ~{gb_per_sample:.2f} GB/sample -> max {max_samples_per_chunk} samples/chunk")
        print(f"  Using {num_chunks} chunks of {chunk_size} samples each...")
        
        task_args_list = []
        samples_assigned = 0
        for chunk_idx in range(num_chunks):
            samples_in_this_chunk = min(chunk_size, num_samples - samples_assigned)
            if samples_in_this_chunk <= 0:
                break
            # Use seed * large_stride + offset to create non-overlapping seed chains per dataset
            chunk_base_seed = seed * 100000 + samples_assigned
            
            task_args = (chunk_base_seed, config_dict, split, target_node_states, target_edge_states, samples_in_this_chunk)
            task_args_list.append(task_args)
            samples_assigned += samples_in_this_chunk

        # Use ProcessPoolExecutor with fork context for efficiency (no GIL, shared memory for read-only data)
        datapoints = []
        
        # Use 'fork' on Linux for faster startup (avoids re-importing modules)
        import multiprocessing as mp_module
        ctx = mp_module.get_context('fork')
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(_worker_logic_chunk, args): i for i, args in enumerate(task_args_list)}
            
            with tqdm.tqdm(total=num_samples, desc="Generating") as pbar:
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        for result in chunk_results:
                            data = _process_worker_result(result)
                            if data:
                                datapoints.append(data)
                            pbar.update(1)
                    except Exception as e:
                        print(f"Chunk failure: {e}")
    else: #sequential
        print(f"Generating {num_samples} samples sequentially...")
        datapoints = []
        with tqdm.tqdm(total=num_samples, desc="Generating") as pbar:
            task_args = (seed * 100000, config_dict, split, target_node_states, target_edge_states, num_samples)
            chunk_results = _worker_logic_chunk(task_args)
            for result in chunk_results:
                data = _process_worker_result(result)
                if data:
                    datapoints.append(data)
                pbar.update(1)

    return datapoints

def _process_worker_result(result):
    """Helper to convert worker output to PyG Data."""
    if 'error' in result:
        # print(f"Sample failed: {result['error']}") 
        return None

    node_t = torch.tensor(result['node_fts'], dtype=torch.float32)
    edge_t = torch.tensor(result['edge_fts'], dtype=torch.float32)
    
    node_t = _pad_features(node_t, result['t_node'])
    edge_t = _pad_features(edge_t, result['t_edge'])
    
    scalars_t = torch.tensor(result['scalars'], dtype=torch.float32)
    edge_index_t = torch.tensor(result['edge_index'], dtype=torch.long).contiguous()
    
    # For scalar output type, y is the final scalar on self-loops
    if result['output_type'] == "scalar":
        self_loop_mask = edge_index_t[0] == edge_index_t[1]
        y = scalars_t[self_loop_mask, -1, 0].clone().detach()  # Final timestep scalars on self-loops
    else:
        output_fts = edge_t if result['output_type'] == "pointer" else node_t
        y = output_fts[:, -1, result['output_idx']].clone().detach().long()
    
    return Data(
        node_fts=node_t,
        edge_fts=edge_t,
        scalars=scalars_t,
        edge_index=edge_index_t,
        y=y,
        pos=torch.tensor(result['pos'], dtype=torch.float),
        goal=torch.tensor(result['goal'], dtype=torch.long),
        start=torch.tensor(result['start'], dtype=torch.long)
    )

def create_dataloader(config: base_config.Config, split: str, seed: int, device, num_workers: int = None):
    # 1. LAZY LOADING (force for large graphs to avoid memory issues)
    problem_size = config.problem_size[split]
    force_lazy = problem_size >= 500  # Auto-enable lazy for large graphs
    
    if config.use_lazy_dataset or force_lazy:
        if num_workers is None: num_workers = 0
        if force_lazy:
            print(f"Auto-enabling LazyDataset for large graph (n={problem_size}) to avoid memory issues")
        return create_lazy_dataloader(config, split, seed, device, num_workers=num_workers)

    # 2. CACHED
    use_cache = getattr(config, 'use_dataset_cache', True)
    cache_path = get_cache_path(config, split, seed)

    if use_cache:
        datapoints = load_dataset_from_cache(cache_path, device, expected_seed=seed)
        if datapoints is not None:
            return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True, collate_fn=pad_and_collate)

    # 3. GENERATE FRESH
    start_time = time.time()
    datapoints = create_dataloader_distributed(config, split, seed, num_workers=num_workers)
    
    duration = time.time() - start_time
    if len(datapoints) > 0:
        print(f"Generation finished in {duration:.2f}s")
    
    if use_cache and len(datapoints) > 0:
        save_dataset_to_cache(datapoints, cache_path, seed)
    
    return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True, collate_fn=pad_and_collate)


def _lazy_generate_sample(args):
    """
    Standalone function for generating a sample - can be pickled for multiprocessing.
    Takes a tuple: (idx, problem_size, algorithm_name, sampler_type, sampler_config, num_node_states, num_edge_states, output_type, output_idx, seed)
    """
    if len(args) == 10:
        (idx, problem_size, algorithm_name, graph_type, config_dict, 
         num_node_states, num_edge_states, output_type, output_idx, seed) = args
    else:
        # Backward compatibility or fallback
        (idx, problem_size, algorithm_name, graph_type, config_dict, 
         num_node_states, num_edge_states, output_type, output_idx) = args
        seed = None
    
    # print(f"    [Worker {idx}] Starting generation...")
    # Set seed for reproducibility
    # If explicit seed provided (e.g. from eval loop), use it as base.
    if seed is not None:
        np.random.seed(seed + idx)
    else:
        np.random.seed(idx + hash(algorithm_name) % 2**31)
    
    # Reconstruct sampler (can't pickle samplers directly)
    class SimpleConfig:
        def __init__(self, d): self.__dict__ = d
    config = SimpleConfig(config_dict)
    
    if graph_type == 'grid':
        sampler = GridGraphSampler(config)
    elif graph_type == 'roadmap':
        sampler = RoadmapGraphSampler(config)
    elif graph_type == 'geometric':
        sampler = GeometricGraphSampler(config)
    else:
        sampler = ErdosRenyiGraphSampler(config)
    
    algorithm = ALGORITHMS[algorithm_name]
    
    # 1. Generate instance
    instance = sampler(problem_size)

    # 2. Run algorithm (NumPy outputs)
    if algorithm_name == 'a_star':
        node_fts, edge_fts, scalars = algorithm(instance, build_full_tree=getattr(config, "full_tree", True), pad_len=problem_size)
    else:
        node_fts, edge_fts, scalars = algorithm(instance)

    # 3. Prepare raw CPU arrays ONLY
    edge_index = instance.edge_index.astype(np.int64)

    node_fts = np.transpose(node_fts, (1, 0, 2))  # (N, T, F)
    edge_fts = np.transpose(edge_fts, (1, 0, 2))  # (E, T, F)
    scalars = np.transpose(scalars, (1, 0, 2))

    # 4. Pad features (multitask safety)
    node_fts = _pad_features(torch.from_numpy(node_fts), num_node_states).numpy()
    edge_fts = _pad_features(torch.from_numpy(edge_fts), num_edge_states).numpy()

    # Compute y based on output type
    if output_type == "scalar":
        # For scalar output, y is the final scalar on self-loops
        self_loop_mask = edge_index[0] == edge_index[1]
        y = scalars[self_loop_mask, -1, 0]  # (N,) - one value per node
    else:
        output_fts = edge_fts if output_type == "pointer" else node_fts
        y = output_fts[:, -1, output_idx]

    print(f"    [Worker {idx}] Completed generation")
    return {
        "node_fts": node_fts,
        "edge_fts": edge_fts,
        "scalars": scalars,
        "edge_index": edge_index,
        "y": y,
        "pos": instance.pos,
        "goal": instance.goal,
        "start": instance.start,
        "sample_idx": idx # Return the index for tracking!
    }



# Global registry for cleanup
_active_lazydatasets = weakref.WeakSet()

def _cleanup_lazydatasets():
    for ds in _active_lazydatasets:
        ds.shutdown()

atexit.register(_cleanup_lazydatasets)

class LazyDataset(Dataset):
    """
    LazyDataset with simple prefetching using ProcessPoolExecutor.
    
    Refactored to support robust buffered streaming with variable batch sizes.
    It separates generation from consumption order (yields whatever is ready).
    """
    def __init__(self, config, split, sampler, algorithm, num_prefetch_workers=0, start_idx=0, num_samples_override=None, seed=None, preloaded_buffer=None):
        super().__init__()
        _active_lazydatasets.add(self)
        self.config = config
        self.split = split
        self.sampler = sampler
        self.algorithm = algorithm
        self.num_samples = num_samples_override if num_samples_override is not None else config.num_samples[split]
        self.problem_size = config.problem_size[split]
        self.start_idx = start_idx
        self.seed = seed
        self.preloaded_buffer = preloaded_buffer if preloaded_buffer is not None else []
        
        # Store config as dict for pickling to worker processes
        self.graph_type = getattr(config, 'graph_type', 'er')
        self.config_dict = vars(config) if hasattr(config, '__dict__') else config.__dict__
        
        # Prefetching setup
        # Respect user's explicit num_prefetch_workers value (including 0 for no multiprocessing)
        self.num_prefetch_workers = num_prefetch_workers
        self.executor = None
        
        if self.num_prefetch_workers > 0:
            print(f"LazyDataset: Using {self.num_prefetch_workers} prefetch workers (samples {start_idx}..{start_idx+self.num_samples-1})")
            import multiprocessing as mp_module
            # Use 'fork' context on Linux for efficiency
            # Fallback to spawn if needed, but fork is faster for read-only config sharing
            #DEBUG: try spawn for now
            try:
                ctx = mp_module.get_context('fork')
                self.executor = ProcessPoolExecutor(max_workers=self.num_prefetch_workers, mp_context=ctx)
            except ValueError:
                self.executor = ProcessPoolExecutor(max_workers=self.num_prefetch_workers)

    def shutdown(self):
        """Force shutdown of the executor to free memory."""
        if self.executor:
            # print("Shutting down LazyDataset executor...")
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def __del__(self):
        self.shutdown()

    def _make_worker_args(self, idx):
        """Create picklable args tuple for _lazy_generate_sample.
        
        idx is the logical sample index within this dataset's num_samples range.
        For preloaded samples (idx < len(preloaded_buffer)), we don't call this.
        For new samples (idx >= len(preloaded_buffer)), we generate at:
            start_idx + idx
        This correctly handles resumption where preloaded samples were already
        at indices [start_idx, start_idx + len(preloaded_buffer)).
        """
        return (
            self.start_idx + idx,
            self.problem_size,
            self.config.algorithm,
            self.graph_type,
            self.config_dict,
            self.config.num_node_states,
            self.config.num_edge_states,
            self.config.output_type,
            self.config.output_idx,
            self.seed  # Pass seed explicitly
        )

    def _to_data(self, c):
        """Convert result dict to PyG Data object."""
        # Note: goal must be 1-d tensor for PyG collation to work properly
        goal_val = c["goal"]
        if isinstance(goal_val, (int, float)):
            goal_tensor = torch.tensor([goal_val], dtype=torch.long)
        else:
            goal_tensor = torch.tensor(goal_val, dtype=torch.long)
            if goal_tensor.dim() == 0:
                goal_tensor = goal_tensor.unsqueeze(0)
        
        # Handle start node similarly
        start_val = c.get("start", 0)
        if isinstance(start_val, (int, float)):
            start_tensor = torch.tensor(start_val, dtype=torch.long)
        else:
            start_tensor = torch.tensor(start_val, dtype=torch.long)
            if start_tensor.dim() == 0:
                start_tensor = start_tensor.unsqueeze(0)
        
        return Data(
            node_fts=torch.tensor(c["node_fts"], dtype=torch.float32),
            edge_fts=torch.tensor(c["edge_fts"], dtype=torch.float32),
            scalars=torch.tensor(c["scalars"], dtype=torch.float32),
            edge_index=torch.tensor(c["edge_index"], dtype=torch.long),
            y=torch.tensor(c["y"], dtype=torch.long if c["y"].dtype != np.float32 else torch.float32),
            pos=torch.tensor(c["pos"], dtype=torch.float32),
            goal=goal_tensor,
            start=start_tensor,
        )
    
    def iter_batches_as_ready(self, max_batch_size=None, min_batch_size=1):
        """
        Stream batches of ready samples.
        
        Design:
        - Keeps a pool of active futures.
        - Yields batches as soon as `min_batch_size` items are ready.
        - Respects `max_batch_size` to avoid GPU OOM.
        - Automatically manages prefetching to keep workers busy.
        - Order of samples is NOT preserved (yields whatever finishes first).
        """
        # Sequential fallback with preloaded buffer handling
        if self.executor is None:
            batch = []
            
            # 1. Yield preloaded
            if self.preloaded_buffer:
                for item_dict in self.preloaded_buffer:
                    batch.append(self._to_data(item_dict))
                    if max_batch_size and len(batch) >= max_batch_size:
                        yield pad_and_collate(batch)
                        batch = []
            
            # 2. Yield new samples
            # Adjust range to skip preloaded count
            start = len(self.preloaded_buffer)
            for idx in range(start, self.num_samples):
                # Memory check before generation
                # if (idx - start) % 1 == 0:  # Check every sample
                #     try:
                #         import psutil
                #         import gc
                #         process = psutil.Process()
                #         mem_gb = process.memory_info().rss / 1024**3
                #         print(f"[LazyDataset] Before generating sample {idx+1}/{self.num_samples}: {mem_gb:.2f} GB, batch_size={len(batch)}")
                #     except:
                #         pass
                
                c = _lazy_generate_sample(self._make_worker_args(idx))
                data_obj = self._to_data(c)
                
                # Explicitly delete the raw result dict to free memory
                del c
                
                # Memory check after generation
                # if (idx - start) % 1 == 0:  # Check every sample
                #     try:
                #         import psutil
                #         import gc
                #         process = psutil.Process()
                #         mem_gb = process.memory_info().rss / 1024**3
                #         print(f"[LazyDataset] After generating sample {idx+1}/{self.num_samples}: {mem_gb:.2f} GB")
                #     except:
                #         pass
                
                batch.append(data_obj)
                
                # CRITICAL: For large graphs, yield immediately (don't accumulate in batch)
                # DFS on size=800 creates ~0.5-0.8 GB per sample, so we can't hold many
                should_yield = False
                if max_batch_size and len(batch) >= max_batch_size:
                    should_yield = True
                # elif self.problem_size >= 600 and len(batch) >= 1:
                #     # For very large graphs, yield after EVERY sample to avoid OOM
                #     should_yield = True
                
                if should_yield:
                    yield pad_and_collate(batch)
                    del batch
                    batch = []
                    # gc.collect()  # Force cleanup
            
            if batch:
                yield pad_and_collate(batch)
                del batch
                # gc.collect()
            return

        # Parallel Streaming Logic
        import concurrent.futures
        
        # State
        futures = set()
        self.buffer = [] # ADDED Instance var for access during timeout save
        self.next_submit_idx = 0  # Track next index to submit (for checkpoint resume)
        
        # === Inject Preloaded Buffer ===
        # Preloaded samples are already generated - they count toward num_samples
        # We need to generate (num_samples - len(preloaded_buffer)) NEW samples
        if self.preloaded_buffer:
             print(f"LazyDataset: Injecting {len(self.preloaded_buffer)} preloaded samples into buffer.")
             for item in self.preloaded_buffer:
                 # Convert dicts to Data objects (preferred path from checkpoint)
                 if isinstance(item, dict):
                    self.buffer.append(self._to_data(item))
                 else:
                    # Legacy: Data object directly - convert to dict and back to ensure consistent format
                    item_dict = {
                        'node_fts': item.node_fts.numpy() if hasattr(item.node_fts, 'numpy') else item.node_fts,
                        'edge_fts': item.edge_fts.numpy() if hasattr(item.edge_fts, 'numpy') else item.edge_fts,
                        'scalars': item.scalars.numpy() if hasattr(item.scalars, 'numpy') else item.scalars,
                        'edge_index': item.edge_index.numpy() if hasattr(item.edge_index, 'numpy') else item.edge_index,
                        'y': item.y.numpy() if hasattr(item.y, 'numpy') else item.y,
                        'pos': item.pos.numpy() if hasattr(item.pos, 'numpy') else item.pos,
                        'goal': int(item.goal.item()) if item.goal.dim() == 0 else int(item.goal[0].item()),
                        'start': int(item.start.item()) if item.start.dim() == 0 else int(item.start[0].item()),
                    }
                    self.buffer.append(self._to_data(item_dict))
        
        # next_submit_idx is RELATIVE to start_idx, indicating how many we've submitted
        # Preloaded samples count as already submitted
        self.next_submit_idx = len(self.preloaded_buffer)
        returned_count = 0
        
        print(f"LazyDataset: Will yield {self.num_samples} samples ({len(self.preloaded_buffer)} preloaded, {self.num_samples - len(self.preloaded_buffer)} to generate)")
        
        # Target number of pending tasks (2x workers to hide latency)
        target_in_flight = self.num_prefetch_workers * 2
        
        # How many NEW samples to generate (excluding preloaded)
        samples_to_generate = self.num_samples - len(self.preloaded_buffer)
        
        def submit_tasks():
            # Only submit if total in flight (running + buffered) is below limit
            # This prevents memory explosion if consumer is slow
            # next_submit_idx counts from 0 to samples_to_generate (not including preloaded)
            while (len(futures) + len(self.buffer)) < target_in_flight and self.next_submit_idx < self.num_samples:
                # The actual sample index is: start_idx + preloaded_count + (next_submit_idx - preloaded_count)
                # Simplified: start_idx + next_submit_idx - preloaded_count + preloaded_count = start_idx + next_submit_idx - preloaded_count + preloaded_count
                # Actually we want: new samples start at index (start_idx + len(preloaded_buffer))
                actual_idx = self.next_submit_idx
                args = self._make_worker_args(actual_idx)
                f = self.executor.submit(_lazy_generate_sample, args)
                futures.add(f)
                self.next_submit_idx += 1

        # Initial fill
        submit_tasks()
        
        while returned_count < self.num_samples:
            # 1. Check for completed futures (non-blocking initially, then short wait)
            # Use a short timeout to allow responsive checking without busy-waiting
            wait_time = 0 if self.buffer else 0.1  # Don't wait if buffer has data to yield
            done, not_done = concurrent.futures.wait(futures, timeout=wait_time, return_when=concurrent.futures.FIRST_COMPLETED if futures else concurrent.futures.ALL_COMPLETED)
            
            # 2. Process completed tasks immediately - add to buffer right away
            for f in done:
                futures.remove(f)
                try:
                    res = f.result()
                    self.buffer.append(self._to_data(res))
                    
                    # Memory check after adding to buffer
                    # if len(self.buffer) % 2 == 0:  # Check every 2 samples
                    #     try:
                    #         import psutil
                    #         process = psutil.Process()
                    #         mem_gb = process.memory_info().rss / 1024**3
                    #         print(f"  [LazyDataset] Buffer size: {len(self.buffer)}, Memory: {mem_gb:.2f} GB")
                    #     except:
                    #         pass
                except Exception as e:
                    print(f"Generation error: {e}")
                    # In case of error, we still count it as 'processed' to ensure termination
                    returned_count += 1
            
            # 3. Yield batches if we have enough data (or if we are finishing up)
            # Logic: Yield if >= min_batch_size OR (no more futures coming AND buffer has data)
            is_finishing = (self.next_submit_idx >= self.num_samples and not futures)
            threshold = 1 if is_finishing else min_batch_size
            
            while len(self.buffer) >= threshold:
                # Determine batch size
                take_n = len(self.buffer)
                if max_batch_size:
                    take_n = min(take_n, max_batch_size)
                
                # Extract batch (copy the items, don't slice-assign)
                batch_data = self.buffer[:take_n]
                
                # Remove yielded items from buffer IN-PLACE *BEFORE* YIELDING
                # This ensures that if the consumer manipulates the buffer (e.g. for checkpointing),
                # we don't interfere with their state when we resume.
                del self.buffer[:take_n]
                
                # Debug: log batch sizes to diagnose collation issues
                if len(batch_data) > 0:
                    sample_shapes = {
                        'node_fts': batch_data[0].node_fts.shape,
                        'edge_fts': batch_data[0].edge_fts.shape,
                    }
                    # Memory check
                    # try:
                    #     import psutil
                    #     process = psutil.Process()
                    #     mem_gb = process.memory_info().rss / 1024**3
                    #     print(f"  [LazyDataset] Before collation: {mem_gb:.2f} GB, Collating batch of {take_n} samples (shapes: N={sample_shapes['node_fts']}, E={sample_shapes['edge_fts']})...")
                    # except:
                    #     pass
                    print(f"  [LazyDataset] Collating batch of {take_n} samples (shapes: N={sample_shapes['node_fts']}, E={sample_shapes['edge_fts']})...")
                
                # Yield
                collated_batch = pad_and_collate(batch_data)
                
                # Memory check after collation
                # try:
                #     import psutil
                #     process = psutil.Process()
                #     mem_gb = process.memory_info().rss / 1024**3
                #     print(f"  [LazyDataset] After collation: {mem_gb:.2f} GB\")")
                # except:
                #     pass
                
                yield collated_batch
                returned_count += len(batch_data)
                
                # Clean up references
                del batch_data
                del collated_batch
                
                # If we emptied buffer below next threshold, stop yielding
                if len(self.buffer) < threshold:
                    break
            
            # 4. Refill pipe
            submit_tasks()
            
            # Verify termination condition
            if is_finishing and not self.buffer:
                break
    
    def iter_as_ready(self):
        """Deprecated: Use iter_batches_as_ready instead."""
        for batch in self.iter_batches_as_ready(max_batch_size=1, min_batch_size=1):
            yield batch.get_example(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.executor is not None:
             print(f"Warning: Random access __getitem__({idx}) called on LazyDataset. "
                   "This bypasses the streaming optimization and may match existing futures poorly.")
             
             # Simple fallback: generate locally or via submit -> wait
             # Since we are refactoring for eval, let's keep it simple: blocking generate
             c = _lazy_generate_sample(self._make_worker_args(idx))
             return self._to_data(c)
        else:
            # No prefetching - generate directly
            c = _lazy_generate_sample(self._make_worker_args(idx))
            return self._to_data(c)
    
    def __del__(self):
        """Clean up executor on deletion."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)

def create_lazy_dataloader(config, split, seed, device, num_workers=0):
    np.random.seed(seed)

    graph_type = getattr(config, 'graph_type', 'er')
    if graph_type == 'grid':
        sampler = GridGraphSampler(config)
    elif graph_type == 'roadmap':
        sampler = RoadmapGraphSampler(config)
    elif graph_type == 'geometric':
        # Use the Geometric sampler for A*
        sampler = GeometricGraphSampler(config) 
    else:
        sampler = ErdosRenyiGraphSampler(config)
        
    algorithm = ALGORITHMS[config.algorithm]

    # For large graphs, use prefetch workers for parallel generation
    # This keeps at most num_prefetch_workers * 2 samples in RAM (~48 * 500MB = 24GB max)
    problem_size = config.problem_size[split]
    if problem_size >= 400:
        # Use num_workers for prefetching, DataLoader workers=0 (no redundant parallelism)
        num_prefetch_workers = num_workers if num_workers > 0 else max(1, (os.cpu_count() or 1) // 4)
        dataset = LazyDataset(config, split, sampler, algorithm, num_prefetch_workers=num_prefetch_workers)
        dataloader_workers = 0  # Prefetching handles parallelism
    else:
        # Small graphs: use standard DataLoader workers
        dataset = LazyDataset(config, split, sampler, algorithm, num_prefetch_workers=0)
        dataloader_workers = num_workers

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        # also seed torch for full determinism in workers
        torch.manual_seed(worker_seed)

    # Use torch.utils.data.DataLoader + collate that builds a PyG Batch
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        worker_init_fn=seed_worker if dataloader_workers > 0 else None,
        collate_fn=pad_and_collate,
    )

    
    return loader



# ---- DATA CACHING ----
def get_dataset_cache_key(config, split: str, seed: int) -> str:
    """
    Generate a unique cache key for a dataset based on config parameters.
    
    Only includes parameters that affect data generation, not training.
    """
    cache_dict = {
        'algorithm': config.algorithm,
        'graph_type': getattr(config, 'graph_type', 'er'),
        'problem_size': config.problem_size[split],
        'num_samples': config.num_samples[split],
        'edge_weights': config.edge_weights,
        'generate_random_numbers': config.generate_random_numbers,
        'split': split,
        'seed': seed,
    }
    
    # Create deterministic hash
    json_str = json.dumps(cache_dict, sort_keys=True)
    hash_obj = hashlib.md5(json_str.encode())
    return hash_obj.hexdigest()


def get_cache_path(config, split: str, seed: int) -> Path:
    """Get the file path for cached dataset."""
    cache_dir = Path(getattr(config, 'cache_directory', 'data_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize by algorithm and graph type
    algo_dir = cache_dir / config.algorithm / getattr(config, 'graph_type', 'er')
    algo_dir.mkdir(parents=True, exist_ok=True)
    
    # Include seed explicitly in filename for clarity
    # Format: split_seed<N>_hash.pkl (e.g., train_seed40_a1b2c3d4.pkl)
    cache_key = get_dataset_cache_key(config, split, seed)
    return algo_dir / f"{split}_seed{seed}_{cache_key}.pkl"


def save_dataset_to_cache(datapoints: List[Data], cache_path: Path, seed: int):
    """Save generated dataset to disk with metadata and exclusive locking."""
    import fcntl
    
    try:
        # Move data to CPU before saving to avoid GPU memory issues
        cpu_datapoints = [data.cpu() for data in datapoints]
        
        # Include metadata for verification
        cache_data = {
            'datapoints': cpu_datapoints,
            'seed': seed,
            'num_samples': len(cpu_datapoints),
            'timestamp': time.time(),
        }
        
        # Use exclusive lock during write to prevent corruption
        with open(cache_path, 'wb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        print(f"Dataset cached to: {cache_path} (seed={seed}, n={len(cpu_datapoints)})")
    except Exception as e:
        print(f"Warning: Failed to cache dataset: {e}")


def load_dataset_from_cache(cache_path: Path, device, expected_seed: int, timeout: float = 30.0) -> Optional[List[Data]]:
    """Load dataset from disk cache with seed verification and lock detection."""
    import fcntl
    import errno
    
    if not cache_path.exists():
        return None
    
    try:
        f = open(cache_path, 'rb')
        
        # Try to acquire a shared (read) lock with timeout
        # If another process holds an exclusive lock, this will fail
        start = time.time()
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                break  # Got the lock
            except IOError as e:
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    if time.time() - start > timeout:
                        f.close()
                        raise TimeoutError(f"Cache file '{cache_path}' is locked by another process. "
                                           f"Try 'pkill -9 -u $USER -f python' to kill stale processes.")
                    time.sleep(0.5)
                else:
                    raise
        
        try:
            cache_data = pickle.load(f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()
        
        # Handle both old format (just list) and new format (dict with metadata)
        if isinstance(cache_data, dict):
            datapoints = cache_data['datapoints']
            cached_seed = cache_data.get('seed')
            
            # Verify seed matches
            if cached_seed is not None and cached_seed != expected_seed:
                print(f"Warning: Cached seed {cached_seed} != expected {expected_seed}. Regenerating...")
                return None
            
            print(f"Dataset loaded from cache: {cache_path} (seed={cached_seed}, n={len(datapoints)})")
        else:
            # Old format - just the datapoints
            datapoints = cache_data
            print(f"Dataset loaded from cache: {cache_path} (legacy format)")
        
        #not moving to device her, as we handle it in the traning loop
        return datapoints
    except TimeoutError:
        raise  # Re-raise lock timeout to caller
    except Exception as e:
        print(f"Warning: Failed to load cached dataset: {e}")
        return None


def pad_and_collate(batch):
    """
    Pads time and node dimensions so PyG can batch variable-sized graph trajectories.
    Fully vectorized implementation for maximum speed.
    
    Handles:
    1. Variable Time Steps (T) -> Pads to max_T
    2. Variable Node Counts (N) -> Adds Ghost Nodes
    3. Hybrid Targets (y) -> Detects if y is Node-level or Edge-level
    """
    if not batch:
        return Batch.from_data_list([])
    
    B = len(batch)
    
    # Fast path: single sample, no padding needed
    if B == 1:
        return Batch.from_data_list(batch)

    # =====================================================================
    # PHASE 1: Gather dimensions (vectorized)
    # =====================================================================
    Ns = [d.node_fts.shape[0] for d in batch]
    Ts = [d.node_fts.shape[1] for d in batch]
    Es = [d.edge_fts.shape[0] for d in batch]
    
    max_N = max(Ns)
    max_T = max(Ts)
    F = batch[0].node_fts.shape[2]
    EF = batch[0].edge_fts.shape[2]
    S_dim = batch[0].scalars.shape[2]
    
    # Check if all samples have same dimensions (no padding needed)
    all_same_N = all(n == max_N for n in Ns)
    all_same_T = all(t == max_T for t in Ts)
    
    if all_same_N and all_same_T:
        # No padding needed - fast path
        return Batch.from_data_list(batch)
    
    # =====================================================================
    # PHASE 2: Pre-allocate output tensors for entire batch
    # =====================================================================
    # Pre-compute total edges including ghosts
    ghost_counts = [max_N - n for n in Ns]
    total_edges = [e + g for e, g in zip(Es, ghost_counts)]
    
    # Pre-allocate lists for padded data
    padded_node_fts = []
    padded_edge_fts = []
    padded_scalars = []
    padded_edge_indices = []
    padded_ys = []
    
    # =====================================================================
    # PHASE 3: Batch padding with minimal allocations
    # =====================================================================
    # Pre-compute ghost indices for each sample (reusable)
    ghost_indices_cache = {}
    for num_ghosts in set(ghost_counts):
        if num_ghosts > 0:
            start_idx = max_N - num_ghosts
            ghost_indices_cache[num_ghosts] = torch.arange(start_idx, max_N, dtype=torch.long)
    
    for i, d in enumerate(batch):
        N, T = Ns[i], Ts[i]
        E_orig = Es[i]
        num_ghosts = ghost_counts[i]
        
        # -----------------------------------------------------------------
        # Node features: use F.pad (optimized C++ implementation)
        # -----------------------------------------------------------------
        if N < max_N or T < max_T:
            node_pad = torch.nn.functional.pad(
                d.node_fts, 
                (0, 0, 0, max_T - T, 0, max_N - N),
                mode='constant', value=0
            )
        else:
            node_pad = d.node_fts
        padded_node_fts.append(node_pad)
        
        # -----------------------------------------------------------------
        # Edge features + Scalars: handle together to minimize allocations
        # -----------------------------------------------------------------
        if num_ghosts == 0 and T == max_T:
            # No changes needed
            padded_edge_fts.append(d.edge_fts)
            padded_scalars.append(d.scalars)
            padded_edge_indices.append(d.edge_index)
        else:
            # Pre-allocate combined tensor (original edges + ghost edges)
            total_E = E_orig + num_ghosts
            
            # Allocate with zeros (ghosts will be 0)
            edge_pad = torch.zeros((total_E, max_T, EF), dtype=d.edge_fts.dtype)
            scalars_pad = torch.zeros((total_E, max_T, S_dim), dtype=d.scalars.dtype)
            
            # Copy original data in one operation
            edge_pad[:E_orig, :T, :] = d.edge_fts
            scalars_pad[:E_orig, :T, :] = d.scalars
            
            padded_edge_fts.append(edge_pad)
            padded_scalars.append(scalars_pad)
            
            # Edge index: append ghost self-loops
            if num_ghosts > 0:
                ghost_idx = ghost_indices_cache[num_ghosts]
                ghost_loops = torch.stack([ghost_idx, ghost_idx], dim=0)
                padded_edge_indices.append(torch.cat([d.edge_index, ghost_loops], dim=1))
            else:
                padded_edge_indices.append(d.edge_index)
        
        # -----------------------------------------------------------------
        # Y labels: handle padding
        # -----------------------------------------------------------------
        if d.y is not None and num_ghosts > 0:
            if d.y.shape[0] == N:
                # Node-level labels
                y_pad = torch.zeros((max_N,), dtype=d.y.dtype)
                y_pad[:N] = d.y
                y_pad[N:] = ghost_indices_cache[num_ghosts]
                padded_ys.append(y_pad)
            elif d.y.shape[0] == E_orig:
                # Edge-level labels - ghosts point to self (1)
                ghost_labels = torch.ones((num_ghosts,), dtype=d.y.dtype)
                padded_ys.append(torch.cat([d.y, ghost_labels], dim=0))
            else:
                padded_ys.append(d.y)
        else:
            padded_ys.append(d.y)
    
    # =====================================================================
    # PHASE 4: Create Data objects and batch
    # =====================================================================
    padded_list = [
        Data(
            node_fts=padded_node_fts[i],
            edge_fts=padded_edge_fts[i],
            scalars=padded_scalars[i],
            edge_index=padded_edge_indices[i],
            y=padded_ys[i], 
            pos=batch[i].pos,
            goal=batch[i].goal,
            start=batch[i].start,
            num_nodes=max_N, 
        )
        for i in range(B)
    ]

    return Batch.from_data_list(padded_list)

def create_dataloader_with_cache(config, split: str, seed: int, device):
    """
    Create dataloader with caching support. 
    
    Checks cache first, generates and saves if not found. Device irrelevant for caching.
    """
    # Check if caching is enabled (default: True)
    use_cache = getattr(config, 'use_dataset_cache', True)
    
    if use_cache:
        cache_path = get_cache_path(config, split, seed)
        
        # Try to load from cache
        datapoints = load_dataset_from_cache(cache_path, device, expected_seed=seed)
        
        if datapoints is not None:
            return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True, collate_fn=pad_and_collate)
    

    target_node_states = max(config.num_node_states, MAX_NODE_STATES)
    target_edge_states = max(config.num_edge_states, MAX_EDGE_STATES)

    # Generate data (original logic)
    np.random.seed(seed)
    datapoints = []
    
    # Choose Sampler
    graph_type = getattr(config, 'graph_type', 'er')
    if graph_type == 'grid':
        sampler = GridGraphSampler(config)
    elif graph_type == 'roadmap':
        sampler = RoadmapGraphSampler(config)
    elif graph_type == 'geometric':
        sampler = GeometricGraphSampler(config)
    else:
        sampler = ErdosRenyiGraphSampler(config)

    for _ in tqdm.tqdm(
        range(config.num_samples[split]), f"Generate samples for {split} (seed={seed})"
    ):
        instance = sampler(config.problem_size[split])
        
        algorithm = ALGORITHMS[config.algorithm]

        if config.algorithm == 'a_star':
            # A* needs pos for heuristic
            node_fts, edge_fts, scalars = algorithm(instance, build_full_tree=getattr(config, "full_tree", True), pad_len = config.problem_size[split])
        else:
            node_fts, edge_fts, scalars = algorithm(instance)

        edge_index = torch.tensor(instance.edge_index).contiguous()

        # Reshape to (Batch/Time, Nodes, Feats)
        # edge_fts is now already edge-indexed (T, E, F) thanks to optimized push_states
        node_fts = torch.transpose(torch.tensor(node_fts), 0, 1)
        edge_fts = torch.transpose(torch.tensor(edge_fts), 0, 1)  # Already (E, T, F)
        scalars = torch.transpose(torch.tensor(scalars), 0, 1)
        
        # Pad features to expected dimensions
        node_fts = _pad_features(node_fts, target_node_states)
        edge_fts = _pad_features(edge_fts, target_edge_states)

        # For scalar output type, y is the final scalar on self-loops
        if config.output_type == "scalar":
            self_loop_mask = edge_index[0] == edge_index[1]
            y = scalars[self_loop_mask, -1, 0].clone().detach()
        else:
            output_fts = edge_fts if config.output_type == "pointer" else node_fts
            y = output_fts[:, -1, config.output_idx].clone().detach()

        datapoints.append(
            Data(
                node_fts=node_fts,
                edge_fts=edge_fts,
                scalars=scalars,
                edge_index=edge_index,
                y=y,
                pos=torch.tensor(instance.pos, dtype=torch.float),
                goal=torch.tensor(instance.goal),
                start=torch.tensor(instance.start, dtype=torch.long)
            )
        )
    
    # Save to cache
    if use_cache:
        save_dataset_to_cache(datapoints, cache_path, seed)
    
    return DataLoader(
        datapoints, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=pad_and_collate
    )

def clear_cache(algorithm: Optional[str] = None, graph_type: Optional[str] = None):
    """
    Clear cached datasets.
    
    Args:
        algorithm: If specified, only clear this algorithm's cache
        graph_type: If specified, only clear this graph type's cache
    """
    #load the config
    config = base_config.Config()

    cache_dir = Path(getattr(config, 'cache_directory', 'data_cache'))
    
    if not cache_dir.exists():
        print("No cache directory found.")
        return
    
    if algorithm and graph_type:
        target = cache_dir / algorithm / graph_type
        if target.exists():
            import shutil
            shutil.rmtree(target)
            print(f"Cleared cache for {algorithm}/{graph_type}")
    elif algorithm:
        target = cache_dir / algorithm
        if target.exists():
            import shutil
            shutil.rmtree(target)
            print(f"Cleared cache for {algorithm}")
    else:
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("Cleared entire cache")


def get_cache_info(config):
    """Print information about cached datasets."""
    cache_dir = Path(getattr(config, 'cache_directory', 'data_cache'))
    
    if not cache_dir.exists():
        print("No cache directory found.")
        return
    
    total_size = 0
    cache_files = []
    
    for cache_file in cache_dir.rglob('*.pkl'):
        size = cache_file.stat().st_size
        total_size += size
        
        # Parse path: data_cache/algorithm/graph_type/split_seed40_hash.pkl
        parts = cache_file.parts
        if len(parts) >= 4:
            algorithm = parts[-3]
            graph_type = parts[-2]
            filename = parts[-1]
            
            # Extract split and seed from filename
            parts_name = filename.replace('.pkl', '').split('_')
            split = parts_name[0]
            seed = None
            if len(parts_name) > 1 and parts_name[1].startswith('seed'):
                seed = parts_name[1].replace('seed', '')
            
            # Try to load metadata
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    metadata_seed = data.get('seed')
                    num_samples = data.get('num_samples')
                else:
                    metadata_seed = None
                    num_samples = len(data)
            except:
                metadata_seed = None
                num_samples = '?'
            
            cache_files.append({
                'algorithm': algorithm,
                'graph_type': graph_type,
                'split': split,
                'seed': seed or metadata_seed or '?',
                'num_samples': num_samples,
                'size_mb': size / (1024 * 1024),
                'path': str(cache_file)
            })
    
    if not cache_files:
        print("No cached datasets found.")
        return
    
    print(f"\nCached Datasets ({len(cache_files)} files, {total_size / (1024**2):.2f} MB total)")
    print("=" * 90)
    
    # Group by algorithm and graph_type
    from collections import defaultdict
    by_algo = defaultdict(list)
    
    for cf in cache_files:
        key = f"{cf['algorithm']}/{cf['graph_type']}"
        by_algo[key].append(cf)
    
    for key, files in sorted(by_algo.items()):
        total = sum(f['size_mb'] for f in files)
        splits = sorted(set(f['split'] for f in files))
        seeds = sorted(set(str(f['seed']) for f in files))
        
        print(f"{key:30s} | {len(files):2d} files | {total:6.2f} MB | "
              f"splits: {', '.join(splits)} | seeds: {', '.join(seeds)}")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/mst.yaml",
        help="Path to the config file.",
    )
    argparser.add_argument(
        "--seed", "-s",
        type=int,
        default=42**2,
        help="Random seed for data generation.",
    )
    argparser.add_argument(
        "--size", "-sz",
        type=int, 
        default=-1,
        help="The size of the graphs to generate. Overrides the config!",
    )

    argparser.add_argument('--clear', action='store_true', help='Clear cache')
    argparser.add_argument('--algorithm', type=str, help='Target algorithm')
    argparser.add_argument('--graph_type', type=str, help='Target graph type')
    argparser.add_argument('--info', action='store_true', help='Show cache info')

    args = argparser.parse_args()

    #check if its a valid config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")

    config = base_config.read_config(args.config)

    if args.info and (not args.clear):
        get_cache_info(config)
        exit()
    
    if args.clear:
        clear_cache(args.algorithm, args.graph_type)
        exit()

    config = base_config.read_config(args.config)
    config.problem_size = {"test": args.size}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_lazy_dataset:
        data = create_lazy_dataloader(config, "test", seed=args.seed, device=device) #remember to put this onto the device later after getting the batch
    else:
        #this will also cache the data
        data = create_dataloader(config, "test", seed=args.seed, device=device)

    for batch in data:
        print(batch.node_fts[:, -1:, 0].sum() / 32)
        break

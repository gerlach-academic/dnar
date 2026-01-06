import math
import os
import networkx as nx
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset,DataLoader
from torch_geometric.data import Data, Batch
from scipy.spatial import distance_matrix
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, List

from configs import base_config
import argparse

# -----------------------------------------------------------------------------
# 1. PROBLEM INSTANCE WITH PLANNING SUPPORT
# -----------------------------------------------------------------------------

class ProblemInstance:
    def __init__(self, adj, start, goal, weighted, randomness, pos=None):
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
            random_pos = np.random.uniform(0.0, 1.0, (n,))
            self.pos = random_pos[np.argsort(random_pos)]

def push_states(
    node_states, edge_states, scalars, cur_step_nodes, cur_step_edges, cur_step_scalars
):
    node_states.append(np.stack(cur_step_nodes, axis=-1))
    edge_states.append(np.stack(cur_step_edges, axis=-1))
    scalars.append(np.stack(cur_step_scalars, axis=-1))

# Helper to keep legacy algorithms (BFS/DFS) from crashing on 2D inputs
def get_scalar_pos_for_legacy(instance):
    # If pos is 2D (planning map), just take the first coord for the 'scalar' hint
    # This keeps dimensions compatible with standard CLRS models.
    if instance.pos.ndim > 1:
        return instance.pos[:, 0]
    return instance.pos

# -----------------------------------------------------------------------------
# 2. ALGORITHMS
# -----------------------------------------------------------------------------

def bfs(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    visited = np.zeros(n, dtype=np.int32)
    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    # Use helper for 2D compatibility
    cur_scalars = get_scalar_pos_for_legacy(instance)[instance.edge_index[0]]

    visited[instance.start] = 1

    push_states(
        node_states, edge_states, scalars,
        (visited,), (pointers, self_loops), (cur_scalars,),
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
        )

    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (visited,), (pointers, self_loops), (cur_scalars,),
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
    )

    def rec_dfs(current_node, prev_node=-1):
        for out in instance.out_nodes[current_node]:
            if not_in_the_stack[out]:
                in_the_stack[current_node] = 1
                stack_update[current_node][out] = 1
                push_states(
                    node_states, edge_states, scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops), (cur_scalars,),
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
                )
                stack_update[out][current_node] = 0
                rec_dfs(out, current_node)
                top_of_the_stack[current_node] = 1
                top_of_the_stack[out] = 0
                in_the_stack[current_node] = 0
                pre_end[out] = 0
                stack_update[current_node][out] = 1
                push_states(
                    node_states, edge_states, scalars,
                    (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
                    (pointers, stack_update, self_loops), (cur_scalars,),
                )
                stack_update[current_node][out] = 0
        pre_end[current_node] = 1
        stack_update[current_node][current_node] = 1
        push_states(
            node_states, edge_states, scalars,
            (not_in_the_stack, top_of_the_stack, in_the_stack, pre_end),
            (pointers, stack_update, self_loops), (cur_scalars,),
        )
        stack_update[current_node][current_node] = 0

    rec_dfs(instance.start)
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
        )

    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (in_mis, alive), (self_loops,), (compute_current_scalars(),),
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

    node_scalars = instance.pos

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

    # --- 4. Initialization ---
    g_score = np.zeros(n, dtype=np.float32)
    f_score_raw = np.copy(h_vals)
    
    # --- 5. HINT PHYSICS SETUP ---
    # Edge Hint: Potential Function w' = w - h(u) + h(v)
    h_src = h_vals[instance.edge_index[0]]
    h_dst = h_vals[instance.edge_index[1]]
    w_uv = instance.adj[instance.edge_index[0], instance.edge_index[1]]
    
    edge_hint_val = (w_uv - h_src + h_dst) * scale_factor

    # --- 6. TIE-BREAKING HELPER (Crucial for Geometric Graphs) ---
    # Preference: Lower f. If f equal, Higher g (closer to goal).
    # Rank = f - (epsilon * g)
    EPS = 1e-4 

    def compute_current_scalars(g_curr, f_curr):
        s = np.copy(edge_hint_val)
        mask_loops = instance.edge_index[0] == instance.edge_index[1]
        
        # This catches the specific graph causing your crash
        if mask_loops.sum() != n:
             raise ValueError(f"Graph Error: Found {mask_loops.sum()} self-loops, expected {n}")

        # Calculate Rank for hints
        # This tells the network: "Even if f is same, this node is numerically smaller/better"
        ranks = (f_curr - EPS * g_curr) * scale_factor
        
        s[mask_loops] = ranks[instance.edge_index[0][mask_loops]]
        return s

    # Setup Start Node
    g_score[instance.start] = 0
    f_score_raw[instance.start] = h_vals[instance.start]
    in_open[instance.start] = 1

    # Initial Push
    push_states(
        node_states, edge_states, scalars,
        (in_open, in_closed, is_goal), (pointers, self_loops),
        (compute_current_scalars(g_score, f_score_raw),), #returns one score which is 
    )

    for _ in range(n):
        # --- 7. SELECTION LOGIC WITH TIE-BREAKING ---
        # We perform argmin on the RANK, not just raw F.
        # This ensures Python selection matches the Network's gradient signal.
        current_ranks = f_score_raw - (EPS * g_score)
        
        candidates = current_ranks + (1.0 - in_open) * 1e9 
        
        if np.min(candidates) >= 1e9: 
            break 
        
        current = np.argmin(candidates)
        
        # --- 8. TOGGLE LOGIC ---
        if not build_full_tree and current == instance.goal:
            in_open[current] = 0
            in_closed[current] = 1
            push_states(
                node_states, edge_states, scalars,
                (in_open, in_closed, is_goal), (pointers, self_loops),
                (compute_current_scalars(g_score, f_score_raw),),
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
                f_score_raw[neighbor] = g_score[neighbor] + h_vals[neighbor]
                in_open[neighbor] = 1
        
        push_states(
            node_states, edge_states, scalars,
            (in_open, in_closed, is_goal), (pointers, self_loops),
            (compute_current_scalars(g_score, f_score_raw),),
        )

    # if build_full_tree:
    #fill the rest of the tree with the last state so the model basically stops updating on goal reached
    while len(node_states) < target_len:
        push_states(
            node_states, edge_states, scalars,
            (in_open, in_closed, is_goal), (pointers, self_loops),
            (compute_current_scalars(g_score, f_score_raw),),
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
        # Self-loops get the current eccentricity estimate (echo_state at source)
        s = np.zeros(len(instance.edge_index[0]), dtype=np.float32)
        for i, (src, dst) in enumerate(zip(instance.edge_index[0], instance.edge_index[1])):
            if src == dst:
                s[i] = echo_state[source]
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
        )
    
    # Pad to n steps
    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (visited, msg_phase), (tree_to_pointers(), self_loops),
            (compute_current_scalars(),),
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
            
            # Pass the 1D positions so A* calculates correct h(n)
            instance = ProblemInstance(adj, start, goal, self.weighted, random_numbers, pos=pos)
            
            # Connectivity check (BFS)
            trace, _, _ = bfs(instance)
            if np.all(trace[-1, :, 0] == 1):
                return instance

class GeometricGraphSampler:
    """
    Generates Random Geometric Graphs (RGG).
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

        #add tiny bit of noise to break symmetries
        if final_n > 1:
            noise = np.random.uniform(1.00001, 1.0001, size=final_adj.shape) #use additive noise, otherwise weights can become zero in the original
            final_adj = final_adj * noise
            final_adj = (final_adj + final_adj.T) / 2 # Ensure symmetry
        
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

        #add tiny bit of noise to break symmetries
        if N > 1:
            noise = np.random.uniform(1.00001, 1.0001, size=adj.shape)
            adj = adj * noise
            adj = (adj + adj.T) / 2 # Ensure symmetry

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

        # Add tiny noise to break symmetries
        if N > 1:
            noise = np.random.uniform(1.00001, 1.0001, size=adj.shape)
            adj = adj * noise
            adj = (adj + adj.T) / 2 # Ensure symmetry

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
    "dijkstra": dijkstra, #==SP as that is the algorithm used
    "mis": mis, #==maximum independent set
    "a_star": a_star, #TOOD: for a_star we would need to implement edge based reasoning so it can properly compare edges? no relaxation already possible
    "eccentricity": eccentricity, #==eccentricity of source node (max shorttest distance to reach any other node)
}
import sys
import ray

def _pad_features(tensor: torch.Tensor, target_features: int) -> torch.Tensor:
    """Pad the feature dimension (last dim) of a tensor with zeros."""
    current_features = tensor.shape[-1]
    if current_features >= target_features:
        return tensor
    pad_size = target_features - current_features
    padding = torch.zeros(*tensor.shape[:-1], pad_size, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-1)

# Pure Python function that processes a CHUNK of samples (no decorators yet)
def _worker_logic_chunk(args):
    """Process a chunk of samples in a single worker to reduce Ray overhead."""
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
                        build_full_tree=False, 
                        pad_len=config.problem_size[split]
                    )
                else:
                    node_fts, edge_fts, scalars = algorithm(instance)

                # Format Data (Numpy-side)
                edge_index = instance.edge_index
                node_fts = np.transpose(node_fts, (1, 0, 2)) 
                edge_fts_selected = np.transpose(edge_fts[:, edge_index[0], edge_index[1]], (1, 0, 2))
                scalars = np.transpose(scalars, (1, 0, 2))

                results.append({
                    'node_fts': node_fts,
                    'edge_fts': edge_fts_selected,
                    'scalars': scalars,
                    'edge_index': edge_index,
                    'pos': instance.pos,
                    'goal': instance.goal,
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

# Define Ray Remote Function for chunked processing
@ray.remote
def _ray_worker_chunk(args):
    return _worker_logic_chunk(args)

def create_dataloader_distributed(config, split: str, seed: int, num_workers: int = None):
    """
    Generates dataset using RAY with chunked processing for efficiency.
    Each Ray task processes multiple samples to reduce scheduling overhead.
    """
    # 1. Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    num_samples = config.num_samples[split]
    
    # Determine number of workers = number of chunks (one chunk per core)
    if num_workers is None:
        num_workers = max(1, int(ray.available_resources().get('CPU', 4)))
    
    # Number of chunks equals number of cores for optimal parallelism
    num_chunks = num_workers
    chunk_size = (num_samples + num_chunks - 1) // num_chunks
    
    target_node_states = max(config.num_node_states, MAX_NODE_STATES)
    target_edge_states = max(config.num_edge_states, MAX_EDGE_STATES)
    config_dict = vars(config) if hasattr(config, '__dict__') else config.__dict__
    
    # 2. Launch Chunked Tasks
    print(f"Generating {num_samples} samples with Ray ({num_chunks} chunks of ~{chunk_size} samples each)...")
    
    futures = []
    samples_assigned = 0
    for chunk_idx in range(num_chunks):
        samples_in_this_chunk = min(chunk_size, num_samples - samples_assigned)
        if samples_in_this_chunk <= 0:
            break
        # Use seed * large_stride + offset to create non-overlapping seed chains per dataset
        # This ensures seed=40 and seed=41 don't overlap even with many samples
        chunk_base_seed = seed * 100000 + samples_assigned
        
        task_args = (chunk_base_seed, config_dict, split, target_node_states, target_edge_states, samples_in_this_chunk)
        futures.append(_ray_worker_chunk.remote(task_args))
        samples_assigned += samples_in_this_chunk

    # 3. Collect Results with Progress Bar
    datapoints = []
    unfinished = futures
    samples_processed = 0
    
    with tqdm.tqdm(total=num_samples, desc="Ray Generation") as pbar:
        while unfinished:
            # Wait for chunks to complete (batch of results at once)
            done, unfinished = ray.wait(unfinished, num_returns=1, timeout=None)
            
            for future in done:
                try:
                    chunk_results = ray.get(future)
                    for result in chunk_results:
                        data = _process_worker_result(result)
                        if data:
                            datapoints.append(data)
                        samples_processed += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"Chunk failure: {e}")

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
    
    output_fts = edge_t if result['output_type'] == "pointer" else node_t
    y = output_fts[:, -1, result['output_idx']].clone().detach().long()
    
    return Data(
        node_fts=node_t,
        edge_fts=edge_t,
        scalars=scalars_t,
        edge_index=edge_index_t,
        y=y,
        pos=torch.tensor(result['pos'], dtype=torch.float),
        goal=torch.tensor(result['goal'], dtype=torch.long)
    )

def create_dataloader(config: base_config.Config, split: str, seed: int, device, num_workers: int = None):
    # 1. LAZY LOADING
    if config.use_lazy_dataset:
        if num_workers is None: num_workers = 0 
        return create_lazy_dataloader(config, split, seed, device, num_workers=num_workers)

    # 2. CACHED
    use_cache = getattr(config, 'use_dataset_cache', True)
    cache_path = get_cache_path(config, split, seed)

    if use_cache:
        datapoints = load_dataset_from_cache(cache_path, device, expected_seed=seed)
        if datapoints is not None:
            return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True, collate_fn=pad_and_collate)

    # 3. GENERATE (Using Ray)
    start_time = time.time()
    datapoints = create_dataloader_distributed(config, split, seed, num_workers=num_workers)
    
    duration = time.time() - start_time
    if len(datapoints) > 0:
        print(f"Generation finished in {duration:.2f}s")
    
    if use_cache and len(datapoints) > 0:
        save_dataset_to_cache(datapoints, cache_path, seed)
    
    return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True, collate_fn=pad_and_collate)


class LazyDataset(Dataset):
    def __init__(self, config, split, sampler, algorithm):
        super().__init__()
        self.config = config
        self.split = split
        self.sampler = sampler
        self.algorithm = algorithm
        self.num_samples = config.num_samples[split]
        self.problem_size = config.problem_size[split]
        self.cache = {}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx not in self.cache:
            # 1. Generate instance
            instance = self.sampler(self.problem_size)

            # 2. Run algorithm (NumPy outputs)
            node_fts, edge_fts, scalars = self.algorithm(instance)

            # 3. Prepare raw CPU arrays ONLY
            edge_index = instance.edge_index.astype(np.int64)

            node_fts = np.transpose(node_fts, (1, 0, 2))  # (T, N, F)
            edge_fts = np.transpose(
                edge_fts[:, edge_index[0], edge_index[1]], (1, 0, 2)
            )
            scalars = np.transpose(scalars, (1, 0, 2))

            # 4. Pad features (multitask safety)
            node_fts = _pad_features(torch.from_numpy(node_fts), self.config.num_node_states).numpy()
            edge_fts = _pad_features(torch.from_numpy(edge_fts), self.config.num_edge_states).numpy()

            output_fts = edge_fts if self.config.output_type == "pointer" else node_fts
            y = output_fts[:, -1, self.config.output_idx]

            # 5. Cache RAW DATA ONLY
            self.cache[idx] = {
                "node_fts": node_fts,
                "edge_fts": edge_fts,
                "scalars": scalars,
                "edge_index": edge_index,
                "y": y,
                "pos": instance.pos,
                "goal": instance.goal,
            }

        c = self.cache[idx]

        # 6. Rebuild PyG Data (fresh, CPU, safe)
        return Data(
            node_fts=torch.tensor(c["node_fts"], dtype=torch.float32),
            edge_fts=torch.tensor(c["edge_fts"], dtype=torch.float32),
            scalars=torch.tensor(c["scalars"], dtype=torch.float32),
            edge_index=torch.tensor(c["edge_index"], dtype=torch.long),
            y=torch.tensor(c["y"], dtype=torch.long),
            pos=torch.tensor(c["pos"], dtype=torch.float32),
            goal=torch.tensor(c["goal"], dtype=torch.long),
        )

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

    dataset = LazyDataset(config, split, sampler, algorithm)

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
        num_workers=num_workers,
        worker_init_fn=seed_worker,
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
    """Save generated dataset to disk with metadata."""
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
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Dataset cached to: {cache_path} (seed={seed}, n={len(cpu_datapoints)})")
    except Exception as e:
        print(f"Warning: Failed to cache dataset: {e}")


def load_dataset_from_cache(cache_path: Path, device, expected_seed: int) -> Optional[List[Data]]:
    """Load dataset from disk cache with seed verification."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
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
    except Exception as e:
        print(f"Warning: Failed to load cached dataset: {e}")
        return None


def pad_and_collate(batch):
    """
    Pads time and node dimensions so PyG can batch variable-sized graph trajectories.
    Robustly handles:
    1. Variable Time Steps (T)
    2. Variable Node Counts (N) -> Adds Ghost Nodes
    3. Hybrid Targets (y) -> Detects if y is Node-level (Index) or Edge-level (Mask) and pads accordingly.
    """
    if not batch:
        return Batch.from_data_list([])

    # Batch shape: (Nodes, Time, Feats)
    max_N = max(d.node_fts.shape[0] for d in batch)
    max_T = max(d.node_fts.shape[1] for d in batch)

    padded_list = []

    for d in batch:
        N, T, F = d.node_fts.shape
        E_orig = d.edge_fts.shape[0]
        _, _, EF = d.edge_fts.shape
        _, _, S_dim = d.scalars.shape

        # 1. Pad node_fts: (N, T, F) -> (max_N, max_T, F)
        node_pad = torch.zeros((max_N, max_T, F), dtype=d.node_fts.dtype)
        node_pad[:N, :T, :] = d.node_fts

        # 2. Pad edge_fts: (E, T, F) -> (E, max_T, F) (Time padding only)
        edge_pad = torch.zeros((E_orig, max_T, EF), dtype=d.edge_fts.dtype)
        edge_pad[:, :T, :] = d.edge_fts

        # 3. Pad scalars: (E, T, S) -> (E, max_T, S)
        scalars_pad = torch.zeros((E_orig, max_T, S_dim), dtype=d.scalars.dtype)
        scalars_pad[:, :T, :] = d.scalars
        
        current_edge_index = d.edge_index
        current_y = d.y

        # --- FIX: GHOST NODES INJECTION ---
        if N < max_N:
            num_ghosts = max_N - N
            
            # A. Create Ghost Topology (Self-Loops)
            ghost_indices = torch.arange(N, max_N, dtype=torch.long)
            ghost_loops = torch.stack([ghost_indices, ghost_indices], dim=0)
            
            current_edge_index = torch.cat([d.edge_index, ghost_loops], dim=1)
            
            # B. Create Ghost Features (Zeros)
            ghost_edge_attrs = torch.zeros((num_ghosts, max_T, EF), dtype=d.edge_fts.dtype)
            ghost_scalars = torch.zeros((num_ghosts, max_T, S_dim), dtype=d.scalars.dtype)
            
            edge_pad = torch.cat([edge_pad, ghost_edge_attrs], dim=0)
            scalars_pad = torch.cat([scalars_pad, ghost_scalars], dim=0)

            # C. Pad Ground Truth 'y'
            # Check if y is Node-Level (size N) or Edge-Level (size E)
            if d.y is not None:
                if d.y.shape[0] == N: 
                    # Node-Level: Pad with Self-Indices
                    y_pad = torch.zeros((max_N,), dtype=d.y.dtype)
                    y_pad[:N] = d.y
                    y_pad[N:] = ghost_indices # Ghosts point to self
                    current_y = y_pad
                elif d.y.shape[0] == E_orig:
                    # Edge-Level: Append labels for the new ghost loops
                    # Ghosts point to self, so the self-loop edge is TRUE (1)
                    ghost_labels = torch.ones((num_ghosts,), dtype=d.y.dtype)
                    current_y = torch.cat([d.y, ghost_labels], dim=0)
                else:
                    # Fallback (e.g. Graph-level label): No padding needed usually
                    pass

        padded_list.append(
            Data(
                node_fts=node_pad,
                edge_fts=edge_pad,
                scalars=scalars_pad,
                edge_index=current_edge_index,
                y=current_y, 
                pos=d.pos,
                goal=d.goal,
                num_nodes=max_N, 
            )
        )

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
            node_fts, edge_fts, scalars = algorithm(instance, build_full_tree=False, pad_len = config.problem_size[split])
        else:
            node_fts, edge_fts, scalars = algorithm(instance)

        edge_index = torch.tensor(instance.edge_index).contiguous()

        # Reshape to (Batch/Time, Nodes, Feats)
        node_fts = torch.transpose(torch.tensor(node_fts), 0, 1)
        edge_fts = torch.transpose(
            torch.tensor(edge_fts)[:, edge_index[0], edge_index[1]], 0, 1
        )
        scalars = torch.transpose(torch.tensor(scalars), 0, 1)
        
        # Pad features to expected dimensions
        node_fts = _pad_features(node_fts, target_node_states)
        edge_fts = _pad_features(edge_fts, target_edge_states)

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
                goal=torch.tensor(instance.goal)
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


def get_cache_info():
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
        default=42,
        help="Random seed for data generation.",
    )

    argparser.add_argument('--clear', action='store_true', help='Clear cache')
    argparser.add_argument('--algorithm', type=str, help='Target algorithm')
    argparser.add_argument('--graph_type', type=str, help='Target graph type')
    argparser.add_argument('--info', action='store_true', help='Show cache info')

    args = argparser.parse_args()

    if args.info or (not args.clear):
        get_cache_info()
        exit()
    
    if args.clear:
        clear_cache(args.algorithm, args.graph_type)
        exit()

    #check if its a valid config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    

    config = base_config.read_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_lazy_dataset:
        data = create_lazy_dataloader(config, "val", seed=args.seed, device=device) #remember to put this onto the device later after getting the batch
    else:
        data = create_dataloader(config, "val", seed=args.seed, device=device)

    for batch in data:
        print(batch.node_fts[:, -1:, 0].sum() / 32)
        break

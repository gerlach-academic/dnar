import math
import os
import networkx as nx
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import distance_matrix

from configs import base_config
import argparse

# -----------------------------------------------------------------------------
# 1. PROBLEM INSTANCE WITH PLANNING SUPPORT
# -----------------------------------------------------------------------------

class ProblemInstance:
    def __init__(self, adj, start, goal, weighted, randomness, pos=None):
        self.adj = np.copy(adj)
        self.start = start
        self.goal = goal  # NEW: Goal node for planning
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
            random_pos = np.random.uniform(0.0, 1.0, (adj.shape[0],))
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


def dijkstra(instance: ProblemInstance):
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    in_queue = np.zeros(n, dtype=np.int32)
    in_tree = np.zeros(n, dtype=np.int32)
    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    node_dist = np.zeros(n, dtype=np.float32) #changed from position to zero, as was not informative anyways

    def compute_current_scalars(dist_vals):
        # Edge features = Weights. Self-loops = Distance Estimates.
        s = instance.adj[instance.edge_index[0], instance.edge_index[1]]
        s[instance.edge_index[0] == instance.edge_index[1]] = dist_vals
        return s

    in_queue[instance.start] = 1
    # node_dist[start] is already 0, which is correct for the start node.
    
    push_states(
        node_states, edge_states, scalars,
        (in_queue, in_tree), (pointers, self_loops),
        (compute_current_scalars(node_dist),),
    )

    for _ in range(n):
        # Priority Queue selection:
        # Add 1e9 to nodes NOT in queue. This makes them "Infinite" to argsort.
        candidates = node_dist + (1.0 - in_queue) * 1e9
        
        # If min is >= 1e9, queue is empty.
        if np.min(candidates) >= 1e9: 
            break
        
        node = np.argmin(candidates)
        
        in_tree[node] = 1
        in_queue[node] = 0

        for out in instance.out_nodes[node]:
            new_dist = node_dist[node] + instance.adj[node][out]
            
            # Relax edge
            # If out is not in tree AND (not in queue OR found a shorter path)
            if in_tree[out] == 0 and (in_queue[out] == 0 or new_dist < node_dist[out]):
                pointers[out] = np.zeros(n, dtype=np.int32)
                pointers[out][node] = 1
                node_dist[out] = new_dist
                in_queue[out] = 1

        push_states(
            node_states, edge_states, scalars,
            (in_queue, in_tree), (pointers, self_loops),
            (compute_current_scalars(node_dist),),
        )

    # Pad trajectory
    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (in_queue, in_tree), (pointers, self_loops),
            (compute_current_scalars(node_dist),),
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


def a_star(instance: ProblemInstance):
    """
    NEW: A* Search
    Uses instance.pos (coordinates) and instance.goal for heuristics.
    Hints: pred (pointer), f_score (scalar), in_open, in_closed, is_goal (masks).
    """
    n = instance.adj.shape[0]
    node_states = []
    edge_states = []
    scalars = []

    in_open = np.zeros(n, dtype=np.int32)
    in_closed = np.zeros(n, dtype=np.int32)
    is_goal = np.zeros(n, dtype=np.int32)
    is_goal[instance.goal] = 1

    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    # Heuristic: Euclidean
    if instance.pos.ndim == 2:
        h_vals = np.linalg.norm(instance.pos - instance.pos[instance.goal], axis=1)
    else:
        h_vals = np.abs(instance.pos - instance.pos[instance.goal])

    # Initialize f_score to 0 (garbage), will be masked by in_open
    g_score = np.zeros(n)
    f_score = np.zeros(n)

    def compute_current_scalars(f_vals):
        # Scalar Hint: f_score on self loops, weights on edges
        s = instance.adj[instance.edge_index[0], instance.edge_index[1]] # Weights
        mask_loops = instance.edge_index[0] == instance.edge_index[1]
        s[mask_loops] = f_vals
        return s

    g_score[instance.start] = 0
    f_score[instance.start] = h_vals[instance.start]
    in_open[instance.start] = 1

    push_states(
        node_states, edge_states, scalars,
        (in_open, in_closed, is_goal), 
        (pointers, self_loops),
        (compute_current_scalars(f_score),),
    )

    for _ in range(n):
        # Priority: f_score. Unvisited nodes (in_open=0) get +1e9 penalty
        candidates = f_score + (1.0 - in_open) * 1e9
        
        if np.min(candidates) >= 1e9: break 
        
        current = np.argmin(candidates)
        
        if current == instance.goal:
            in_open[current] = 0
            in_closed[current] = 1
            push_states(
                node_states, edge_states, scalars,
                (in_open, in_closed, is_goal), (pointers, self_loops),
                (compute_current_scalars(f_score),),
            )
            break

        in_open[current] = 0
        in_closed[current] = 1

        for neighbor in instance.out_nodes[current]:
            if in_closed[neighbor]: continue

            tentative_g = g_score[current] + instance.adj[current][neighbor]
            
            # If neighbor not in open, or we found a better path
            if in_open[neighbor] == 0 or tentative_g < g_score[neighbor]:
                pointers[neighbor] = np.zeros(n, dtype=np.int32)
                pointers[neighbor][current] = 1
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + h_vals[neighbor]
                in_open[neighbor] = 1
        
        push_states(
            node_states, edge_states, scalars,
            (in_open, in_closed, is_goal), (pointers, self_loops),
            (compute_current_scalars(f_score),),
        )

    while len(node_states) < n:
        push_states(
            node_states, edge_states, scalars,
            (in_open, in_closed, is_goal), (pointers, self_loops),
            (compute_current_scalars(f_score),),
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

def er_probabilities(n):
    base = math.log(n) / n
    return (base, base * 3)

class ErdosRenyiGraphSampler:
    def __init__(self, config):
        self.weighted = config.edge_weights
        self.generate_random_numbers = config.generate_random_numbers

    def __call__(self, num_nodes):
        p_segment = er_probabilities(num_nodes)
        p = p_segment[0] + np.random.rand() * (p_segment[1] - p_segment[0])
        
        start = np.random.randint(0, num_nodes)
        
        while True:
            adj = np.triu(np.random.binomial(1, p, size=(num_nodes, num_nodes)), k=1)
            adj += adj.T

            if self.weighted: #weights only in [0,1] but because of hard coded relaxation any difference is ok, even 1e-6 between to possible edges, as can still be checked correctly
                w = np.triu(np.random.uniform(0.0, 1.0, (num_nodes, num_nodes)), k=1)
                w *= adj
                adj = w + w.T
            
            random_numbers = None
            if self.generate_random_numbers:
                random_numbers = np.random.rand(num_nodes, num_nodes)

            goal = (start + 1) % num_nodes # Dummy
            instance = ProblemInstance(adj, start, goal, self.weighted, random_numbers)
            
            # Use new BFS which handles 2D pos cleanly
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

        # Return the ProblemInstance compatible with your code
        # Pass final_pos so the A* function can calculate h(n)
        return ProblemInstance(
            adj=final_adj, 
            start=start, 
            goal=goal, 
            weighted=True, 
            randomness=None, 
            pos=final_pos
        )

class GridGraphSampler:
    def __init__(self, config):
        self.weighted = True 
        
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

        adj = nx.to_numpy_array(G, weight=None) 
        adj[adj > 0] = 1.0 

        start = np.random.randint(0, N)
        goal = np.random.randint(0, N)
        while start == goal:
             goal = np.random.randint(0, N)
             
        return ProblemInstance(adj, start, goal, True, None, pos=pos_arr)

class RoadmapGraphSampler:
    def __init__(self, config):
        self.weighted = True
        
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
        start = np.random.randint(0, N)
        goal = np.random.randint(0, N)
        while start == goal: goal = np.random.randint(0, N)

        return ProblemInstance(adj, start, goal, True, None, pos=final_pos)


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

ALGORITHMS = {
    "bfs": bfs, #==breadth first search
    "dfs": dfs, #==depth first search
    "mst": mst, #==PRIM as that is the algorithm used
    "dijkstra": dijkstra, #==SP as that is the algorithm used
    "mis": mis, #==maximum independent set
    "a_star": a_star, #TOOD: for a_star we would need to implement edge based reasoning so it can properly compare edges 
    "eccentricity": eccentricity, #==eccentricity of source node (max shorttest distance to reach any other node)
}

def _pad_features(tensor: torch.Tensor, target_features: int) -> torch.Tensor:
    """
    Pad the feature dimension (last dim) of a tensor with zeros to reach target_features.
    
    Args:
        tensor: Shape (time, nodes/edges, current_features)
        target_features: Desired number of features
    
    Returns:
        Tensor with shape (time, nodes/edges, target_features)
    """
    current_features = tensor.shape[-1]
    if current_features >= target_features:
        return tensor
    
    # Create padding: zeros for additional features
    pad_size = target_features - current_features
    padding = torch.zeros(*tensor.shape[:-1], pad_size, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-1)


def create_dataloader(config: base_config.Config, split: str, seed: int, device):
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
        range(config.num_samples[split]), f"Generate samples for {split}"
    ):
        instance = sampler(config.problem_size[split])
        
        node_fts, edge_fts, scalars = ALGORITHMS[config.algorithm](instance)

        edge_index = torch.tensor(instance.edge_index).contiguous()

        # Reshape to (Batch/Time, Nodes, Feats)
        node_fts = torch.transpose(torch.tensor(node_fts), 0, 1)
        edge_fts = torch.transpose(
            torch.tensor(edge_fts)[:, edge_index[0], edge_index[1]], 0, 1
        )
        scalars = torch.transpose(torch.tensor(scalars), 0, 1)
        
        # Pad features to expected dimensions (important for multitask training)
        # This ensures data from algorithms with fewer states can work with
        # models configured for max_node_states/max_edge_states
        node_fts = _pad_features(node_fts, config.num_node_states)
        edge_fts = _pad_features(edge_fts, config.num_edge_states)

        output_fts = edge_fts if config.output_type == "pointer" else node_fts
        y = output_fts[:, -1, config.output_idx].clone().detach()

        datapoints.append(
            Data(
                node_fts=node_fts,
                edge_fts=edge_fts,
                scalars=scalars,
                edge_index=edge_index,
                y=y,
                # Pass Planning Inputs Explicitly
                pos=torch.tensor(instance.pos, dtype=torch.float),
                goal=torch.tensor(instance.goal)
            ).to(device)
        )
    return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True)

class LazyDataset(Dataset):
    def __init__(self, config, split, sampler, algorithm):
        super().__init__()
        self.config = config
        self.split = split
        self.sampler = sampler
        self.algorithm = algorithm
        self.num_samples = config.num_samples[split]
        self.problem_size = config.problem_size[split]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Generate Instance on the fly
        # Note: We ignore 'idx' and just sample random instances.
        # Determinism is handled by worker_init_fn if needed.
        instance = self.sampler(self.problem_size)
        
        # 2. Run Algorithm
        node_fts, edge_fts, scalars = self.algorithm(instance)

        # 3. Format Tensors (Same logic as original code)
        edge_index = torch.tensor(instance.edge_index).contiguous()

        # Transpose to (Time, Nodes, Feats)
        node_fts = torch.transpose(torch.tensor(node_fts), 0, 1)
        edge_fts = torch.transpose(
            torch.tensor(edge_fts)[:, edge_index[0], edge_index[1]], 0, 1
        )
        scalars = torch.transpose(torch.tensor(scalars), 0, 1)

        # 4. Define Target
        output_fts = edge_fts if self.config.output_type == "pointer" else node_fts
        y = output_fts[:, -1, self.config.output_idx].clone().detach()

        # 5. Return Data Object (CPU)
        # We generally do not move to GPU inside __getitem__ to allow 
        # multi-process data loading without CUDA errors.
        return Data(
            node_fts=node_fts,
            edge_fts=edge_fts,
            scalars=scalars,
            edge_index=edge_index,
            y=y,
            pos=torch.tensor(instance.pos, dtype=torch.float),
            goal=torch.tensor(instance.goal)
        )

def create_lazy_dataloader(config, split, seed, device, num_workers=0):
    np.random.seed(seed)
    # 1. Setup Sampler based on config
    graph_type = getattr(config, 'graph_type', 'er')
    if graph_type == 'grid':
        sampler = GridGraphSampler(config)
    elif graph_type == 'roadmap':
        # Use the Geometric sampler for A*
        sampler = GeometricGraphSampler(config) 
    else:
        sampler = ErdosRenyiGraphSampler(config)
        
    algorithm = ALGORITHMS[config.algorithm]

    # 2. Create Dataset
    dataset = LazyDataset(config, split, sampler, algorithm)

    # 3. Worker Init Function for proper randomness in multi-processing
    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)

    # 4. Create DataLoader
    # We use the standard torch DataLoader with a custom collate function 
    # to handle PyG Data objects (batching graphs correctly).
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False, # Shuffling doesn't matter for random generation
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    
    return loader


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

    args = argparser.parse_args()

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

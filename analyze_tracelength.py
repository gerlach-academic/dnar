# analyze_tracelength.py
"""
Analyze algorithm trace lengths vs graph size.
Compare sequential algorithms with theoretical parallel versions.

Usage:
    python analyze_tracelength.py --algorithms bfs top_sort mst --sizes 16 32 64
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
import math
import sys

from generate_data import (
    ProblemInstance, 
    ErdosRenyiGraphSampler, 
    GeometricGraphSampler,
    GridGraphSampler,
    RoadmapGraphSampler,
    bfs, dfs, dijkstra, a_star, mst, topological_sort,
    er_probabilities
)

class SimpleConfig:
    """Minimal config for samplers."""
    def __init__(self, algorithm='bfs', graph_type='er', edge_weights=False):
        # Ensure internal name matches what generate_data expects
        self.algorithm = 'topological_sort' if algorithm == 'top_sort' else algorithm
        self.graph_type = graph_type
        self.edge_weights = edge_weights
        self.generate_random_numbers = False

# -----------------------------------------------------------------------------
# PARALLEL TRACE LENGTH ESTIMATORS
# -----------------------------------------------------------------------------

def parallel_bfs_trace_length(instance: ProblemInstance) -> int:
    """Parallel BFS steps = Diameter + Overhead."""
    n = instance.adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    layer = [instance.start]
    visited[instance.start] = True
    num_layers = 0
    
    while layer:
        next_layer = []
        for node in layer:
            for neighbor in instance.out_nodes[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    next_layer.append(neighbor)
        if next_layer:
            num_layers += 1
        layer = next_layer
    
    return num_layers + 2

def parallel_dijkstra_trace_length(instance: ProblemInstance, epsilon: float = 0.01) -> int:
    """Parallel Dijkstra steps (Delta-stepping approximation)."""
    n = instance.adj.shape[0]
    edge_weights = instance.adj[instance.adj > 0]
    if len(edge_weights) == 0: return n
    
    max_weight = edge_weights.max()
    delta = max_weight * epsilon
    
    g_scores = np.full(n, np.inf)
    g_scores[instance.start] = 0
    processed = np.zeros(n, dtype=bool)
    num_buckets = 0
    
    while not np.all(processed):
        unprocessed_g = np.where(processed, np.inf, g_scores)
        if np.min(unprocessed_g) == np.inf: break
        min_g = np.min(unprocessed_g)
        
        bucket_mask = (g_scores >= min_g) & (g_scores < min_g + delta) & (~processed)
        bucket_nodes = np.where(bucket_mask)[0]
        
        if len(bucket_nodes) == 0: break
        
        for node in bucket_nodes:
            processed[node] = True
            for neighbor in instance.out_nodes[node]:
                if not processed[neighbor]:
                    new_g = g_scores[node] + instance.adj[node, neighbor]
                    if new_g < g_scores[neighbor]:
                        g_scores[neighbor] = new_g
        num_buckets += 1
    
    return num_buckets + 1

def parallel_mst_trace_length(instance: ProblemInstance) -> int:
    """Parallel MST steps (Borůvka's rounds)."""
    n = instance.adj.shape[0]
    if n == 1: return 1
    
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py: return False
        if rank[px] < rank[py]: parent[px] = py
        elif rank[px] > rank[py]: parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1
        return True
    
    num_rounds = 1
    edges_added = 0
    # Safety break
    max_rounds = 2 * int(np.ceil(np.log2(n))) + 5
    
    while edges_added < n - 1 and num_rounds < max_rounds:
        component_min_edge = {}
        # Assumes undirected graph (symmetric adj or upper triangle access)
        for i in range(n):
            for j in range(i + 1, n):
                if instance.adj[i, j] > 0:
                    comp_i, comp_j = find(i), find(j)
                    if comp_i != comp_j:
                        w = instance.adj[i, j]
                        if comp_i not in component_min_edge or w < component_min_edge[comp_i][0]:
                            component_min_edge[comp_i] = (w, i, j)
                        if comp_j not in component_min_edge or w < component_min_edge[comp_j][0]:
                            component_min_edge[comp_j] = (w, i, j)
        
        if not component_min_edge: break
        
        edges_to_add = set()
        for _, u, v in component_min_edge.values():
            edges_to_add.add(tuple(sorted([u, v])))
            
        for u, v in edges_to_add:
            if union(u, v): edges_added += 1
        
        num_rounds += 1
        
    return num_rounds

def parallel_a_star_trace_length(instance: ProblemInstance, epsilon: float = 0.01) -> int:
    """Parallel A* steps (Full tree expansion)."""
    n = instance.adj.shape[0]
    
    if instance.pos.ndim == 1:
        h_vals = np.abs(instance.pos - instance.pos[instance.goal])
    else:
        goal_pos = instance.pos[instance.goal]
        h_vals = np.linalg.norm(instance.pos - goal_pos, axis=1)
    
    g_scores = np.full(n, np.inf)
    g_scores[instance.start] = 0
    f_scores = g_scores + h_vals
    
    in_open = np.zeros(n, dtype=bool)
    in_open[instance.start] = True
    in_closed = np.zeros(n, dtype=bool)
    num_steps = 0
    
    while np.any(in_open):
        open_f = np.where(in_open, f_scores, np.inf)
        f_min = np.min(open_f)
        if f_min == np.inf: break
        
        edge_weights = instance.adj[instance.adj > 0]
        delta = np.mean(edge_weights) * epsilon if len(edge_weights) > 0 else 0.01
        
        expand_mask = in_open & (f_scores < f_min + delta)
        expand_nodes = np.where(expand_mask)[0]
        if len(expand_nodes) == 0: break
        
        for node in expand_nodes:
            in_open[node] = False
            in_closed[node] = True
            for neighbor in instance.out_nodes[node]:
                if in_closed[neighbor]: continue
                tentative_g = g_scores[node] + instance.adj[node, neighbor]
                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + h_vals[neighbor]
                    in_open[neighbor] = True
        num_steps += 1
    
    return num_steps + 1

def parallel_top_sort_trace_length(instance: ProblemInstance) -> int:
    """Parallel Top Sort steps (Kahn's Layers)."""
    n = instance.adj.shape[0]
    adj_bool = (instance.adj > 0).astype(np.int32)
    in_degree = adj_bool.sum(axis=0)
    
    processed = np.zeros(n, dtype=bool)
    steps = 1 # Initial state
    
    remaining_nodes = n
    while remaining_nodes > 0:
        current_layer = np.where((in_degree == 0) & (~processed))[0]
        if len(current_layer) == 0: break # Cycle
        
        steps += 1
        processed[current_layer] = True
        remaining_nodes -= len(current_layer)
        
        for u in current_layer:
            for v in instance.out_nodes[u]:
                if not processed[v]: in_degree[v] -= 1
    
    return steps + 1

# -----------------------------------------------------------------------------
# DRIVER LOGIC
# -----------------------------------------------------------------------------

def get_sequential_trace_length(algo_name: str, instance: ProblemInstance) -> int:
    """Run sequential algorithm."""
    if algo_name == 'bfs':
        node_fts, _, _ = bfs(instance)
    elif algo_name == 'dfs':
        node_fts, _, _ = dfs(instance)
    elif algo_name == 'dijkstra':
        node_fts, _, _ = dijkstra(instance)
    elif algo_name == 'a_star':
        node_fts, _, _ = a_star(instance, build_full_tree=True)
    elif algo_name == 'mst':
        node_fts, _, _ = mst(instance)
    elif algo_name == 'top_sort':
        node_fts, _, _ = topological_sort(instance)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    return node_fts.shape[0]

def analyze_trace_lengths(
    algorithms: list,
    sizes: list,
    graph_type: str = 'er',
    num_samples: int = 20,
    seed: int = 42
):
    np.random.seed(seed)
    results = defaultdict(lambda: defaultdict(lambda: {'seq': [], 'par': []}))
    
    for size in sizes:
        print(f"\nProcessing size {size}...")
        for sample_idx in range(num_samples):
            # Reproducible seed for this sample slot
            current_seed = seed + sample_idx
            
            for algo in algorithms:
                # -------------------------------------------------
                # 1. GENERATE GRAPH SPECIFIC TO THE ALGORITHM
                # -------------------------------------------------
                # This ensures:
                # - top_sort gets DAGs
                # - bfs/mst/dijkstra get Undirected
                # - Weighted algos get weights
                np.random.seed(current_seed)
                
                # Check requirements
                needs_weights = algo in ['dijkstra', 'a_star', 'mst']
                
                # Config
                config = SimpleConfig(algorithm=algo, graph_type=graph_type, edge_weights=needs_weights)
                
                # Sampler
                if graph_type == 'geometric': sampler = GeometricGraphSampler(config)
                elif graph_type == 'grid': sampler = GridGraphSampler(config)
                elif graph_type == 'roadmap': sampler = RoadmapGraphSampler(config)
                else: sampler = ErdosRenyiGraphSampler(config)
                
                try:
                    instance = sampler(size)
                    
                    # -------------------------------------------------
                    # 2. MEASURE TRACES
                    # -------------------------------------------------
                    
                    # Sequential
                    try:
                        seq_len = get_sequential_trace_length(algo, instance)
                        results[algo][size]['seq'].append(seq_len)
                    except Exception: 
                        pass
                    
                    # Parallel
                    try:
                        par_len = None
                        if algo == 'bfs': par_len = parallel_bfs_trace_length(instance)
                        elif algo == 'dijkstra': par_len = parallel_dijkstra_trace_length(instance)
                        elif algo == 'a_star': par_len = parallel_a_star_trace_length(instance)
                        elif algo == 'mst': par_len = parallel_mst_trace_length(instance)
                        elif algo == 'top_sort': par_len = parallel_top_sort_trace_length(instance)
                        
                        if par_len is not None:
                            results[algo][size]['par'].append(par_len)
                    except Exception: 
                        pass
                        
                except Exception as e:
                    print(f"Sample generation failed for {algo}: {e}")
            
            if (sample_idx + 1) % 5 == 0:
                print(f"  Completed {sample_idx + 1}/{num_samples} samples")
    
    return results

def plot_results(results: dict, sizes: list, output_path: str):
    num_algos = len(results)
    if num_algos == 0: return

    fig, axes = plt.subplots(1, num_algos, figsize=(5 * num_algos, 5))
    if num_algos == 1: axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax, (algo, size_data) in zip(axes, results.items()):
        seq_means, seq_stds = [], []
        par_means, par_stds = [], []
        valid_sizes = []
        
        for size in sizes:
            if size in size_data:
                s_vals = size_data[size]['seq']
                p_vals = size_data[size]['par']
                if s_vals:
                    valid_sizes.append(size)
                    seq_means.append(np.mean(s_vals))
                    seq_stds.append(np.std(s_vals))
                    if p_vals:
                        par_means.append(np.mean(p_vals))
                        par_stds.append(np.std(p_vals))
                    else:
                        par_means.append(None)
                        par_stds.append(None)
        
        valid_sizes = np.array(valid_sizes)
        if len(valid_sizes) == 0: continue

        ax.errorbar(valid_sizes, seq_means, yerr=seq_stds, label='Sequential', color=colors[0], marker='o')
        
        if any(p is not None for p in par_means):
            pm = [p for p in par_means if p is not None]
            ps = [s for s, p in zip(par_stds, par_means) if p is not None]
            sz = [s for s, p in zip(valid_sizes, par_means) if p is not None]
            ax.errorbar(sz, pm, yerr=ps, label='Parallel', color=colors[1], marker='s')

        # References
        ax.plot(valid_sizes, valid_sizes, '--', color='gray', alpha=0.3, label='O(n)')
        if len(valid_sizes) > 0 and seq_means[0] > 0:
            base_log = math.log2(valid_sizes[0]) if valid_sizes[0] > 1 else 1
            scale_log = seq_means[0] / base_log
            log_ref = np.log2(valid_sizes) * scale_log
            ax.plot(valid_sizes, log_ref, ':', color='gray', alpha=0.5, label='O(log n)')
        
        ax.set_title(algo.upper())
        ax.set_xlabel('N')
        ax.set_ylabel('Steps')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2, nonpositive='clip')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(valid_sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    try:
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nPlot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

def print_summary(results: dict, sizes: list):
    print("\n" + "=" * 60)
    print("TRACE LENGTH SUMMARY")
    print("=" * 60)
    for algo, data in results.items():
        print(f"\n{algo.upper()}")
        print("-" * 45)
        print(f"{'Size':>6} | {'Seq':>10} | {'Par':>10} | {'Speedup':>8}")
        print("-" * 45)
        for sz in sizes:
            if sz in data:
                s = data[sz]['seq']
                p = data[sz]['par']
                if s:
                    sm = np.mean(s)
                    pm_str, sp_str = "N/A", "N/A"
                    if p:
                        pm = np.mean(p)
                        pm_str = f"{pm:.1f}"
                        sp_str = f"{sm/max(pm, 1e-6):.1f}x"
                    print(f"{sz:>6} | {sm:>10.1f} | {pm_str:>10} | {sp_str:>8}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', '-a', nargs='+', 
                       default=['bfs', 'top_sort'],
                       help='Algorithms to analyze')
    parser.add_argument('--sizes', '-s', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--graph_type', '-g', default='er')
    parser.add_argument('--num_samples', '-n', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', '-o', default='trace_length.pdf')
    
    args = parser.parse_args()
    
    results = analyze_trace_lengths(
        algorithms=args.algorithms,
        sizes=args.sizes,
        graph_type=args.graph_type,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    print_summary(results, args.sizes)
    plot_results(results, args.sizes, args.output)

if __name__ == '__main__':
    main()
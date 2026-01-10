"""
Analyze algorithm trace lengths vs graph size.
Compare sequential algorithms with theoretical parallel versions.

Usage:
    python analyze_trace_lengths.py --algorithms bfs dijkstra a_star --sizes 16 32 64 128 256
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from scipy.spatial import distance_matrix
import math

from generate_data import (
    ProblemInstance, 
    ErdosRenyiGraphSampler, 
    GeometricGraphSampler,
    GridGraphSampler,
    RoadmapGraphSampler,
    bfs, dfs, dijkstra, a_star, mst,
    get_scalar_pos_for_legacy
)


class SimpleConfig:
    """Minimal config for samplers."""
    def __init__(self, algorithm='bfs', graph_type='er', edge_weights=False):
        self.algorithm = algorithm
        self.graph_type = graph_type
        self.edge_weights = edge_weights
        self.generate_random_numbers = False


def parallel_bfs_trace_length(instance: ProblemInstance) -> int:
    """
    Compute the trace length for parallel BFS.
    
    In parallel BFS, all nodes at the same distance are expanded simultaneously.
    The trace length equals the eccentricity of the start node (max distance to any node).
    
    Returns:
        Number of parallel steps (= graph diameter from start node)
    """
    n = instance.adj.shape[0]
    
    # BFS to find distances
    visited = np.zeros(n, dtype=bool)
    distances = np.full(n, -1, dtype=int)
    
    layer = [instance.start]
    visited[instance.start] = True
    distances[instance.start] = 0
    
    num_layers = 0
    
    while layer:
        next_layer = []
        for node in layer:
            for neighbor in instance.out_nodes[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    distances[neighbor] = distances[node] + 1
                    next_layer.append(neighbor)
        
        if next_layer:
            num_layers += 1
        layer = next_layer
    
    # +1 for initial state, +1 for final state
    return num_layers + 2


def parallel_dijkstra_trace_length(instance: ProblemInstance, epsilon: float = 0.01) -> int:
    """
    Estimate trace length for parallel Dijkstra (delta-stepping style).
    
    In delta-stepping, nodes within the same "bucket" (g-score range) can be 
    processed in parallel. This is an approximation of the parallelism.
    
    Args:
        epsilon: Bucket width as fraction of max edge weight
        
    Returns:
        Estimated number of parallel steps
    """
    n = instance.adj.shape[0]
    
    # Find max edge weight for bucket sizing
    edge_weights = instance.adj[instance.adj > 0]
    if len(edge_weights) == 0:
        return n
    
    max_weight = edge_weights.max()
    delta = max_weight * epsilon
    
    # Run Dijkstra but count bucket transitions
    g_scores = np.full(n, np.inf)
    g_scores[instance.start] = 0
    processed = np.zeros(n, dtype=bool)
    
    num_buckets = 0
    
    while not np.all(processed):
        # Find minimum unprocessed g-score
        unprocessed_g = np.where(processed, np.inf, g_scores)
        if np.min(unprocessed_g) == np.inf:
            break
            
        min_g = np.min(unprocessed_g)
        
        # Process all nodes in bucket [min_g, min_g + delta)
        bucket_mask = (g_scores >= min_g) & (g_scores < min_g + delta) & (~processed)
        bucket_nodes = np.where(bucket_mask)[0]
        
        if len(bucket_nodes) == 0:
            break
        
        for node in bucket_nodes:
            processed[node] = True
            for neighbor in instance.out_nodes[node]:
                if not processed[neighbor]:
                    new_g = g_scores[node] + instance.adj[node, neighbor]
                    if new_g < g_scores[neighbor]:
                        g_scores[neighbor] = new_g
        
        num_buckets += 1
    
    return num_buckets + 1  # +1 for initial state


def parallel_mst_trace_length(instance: ProblemInstance) -> int:
    """
    Estimate trace length for parallel MST using Borůvka's algorithm.
    
    Borůvka's algorithm is naturally parallel:
    1. Each connected component finds its minimum outgoing edge
    2. All components add their minimum edge simultaneously
    3. Components merge, repeat until one component remains
    
    This exploits edge weight ties: components can add edges in parallel
    even if they have different weights.
    
    Time complexity: O(log n) rounds, each round processes components in parallel.
    
    Returns:
        Estimated number of parallel steps (Borůvka rounds)
    """
    n = instance.adj.shape[0]
    
    if n == 1:
        return 1
    
    # Union-Find structure for tracking components
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        # Union by rank
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1
        return True
    
    num_rounds = 1  # Initial state
    edges_added = 0
    
    # Borůvka's algorithm: iterate until all nodes in one component
    while edges_added < n - 1:
        # Find minimum outgoing edge for each component
        component_min_edge = {}  # component_id -> (weight, u, v)
        
        for i in range(n):
            for j in range(i + 1, n):
                if instance.adj[i, j] > 0:
                    comp_i = find(i)
                    comp_j = find(j)
                    
                    if comp_i != comp_j:
                        weight = instance.adj[i, j]
                        # Track minimum edge for component i
                        if comp_i not in component_min_edge or weight < component_min_edge[comp_i][0]:
                            component_min_edge[comp_i] = (weight, i, j)
                        # Track minimum edge for component j
                        if comp_j not in component_min_edge or weight < component_min_edge[comp_j][0]:
                            component_min_edge[comp_j] = (weight, i, j)
        
        if not component_min_edge:
            break
        
        # Add all minimum edges in parallel (one per component)
        edges_this_round = set()
        for weight, u, v in component_min_edge.values():
            # Use tuple with sorted endpoints to avoid duplicates
            edge_key = tuple(sorted([u, v]))
            edges_this_round.add(edge_key)
        
        # Apply the unions
        for u, v in edges_this_round:
            if union(u, v):
                edges_added += 1
        
        num_rounds += 1
        
        # Safety check
        if num_rounds > 2 * int(np.ceil(np.log2(n))) + 5:
            # Borůvka should take at most O(log n) rounds
            print(f"Warning: parallel_mst_trace_length exceeded expected Borůvka rounds for n={n}")
            break
    
    return num_rounds


def parallel_a_star_trace_length(instance: ProblemInstance, epsilon: float = 0.01) -> int:
    """
    Estimate trace length for parallel A* (PA*) - FULL TREE VERSION.
    
    PA* processes nodes with similar f-scores in parallel.
    Uses epsilon-admissible expansion: expand all nodes with f < f_min * (1 + epsilon).
    
    This version builds the FULL TREE (expands all nodes), not just until goal.
    
    Returns:
        Estimated number of parallel steps
    """
    n = instance.adj.shape[0]
    
    # Compute heuristic (Euclidean distance to goal)
    if instance.pos.ndim == 1:
        # 1D positions - use absolute difference
        h_vals = np.abs(instance.pos - instance.pos[instance.goal])
    else:
        # 2D positions - use Euclidean distance
        goal_pos = instance.pos[instance.goal]
        h_vals = np.linalg.norm(instance.pos - goal_pos, axis=1)
    
    # Initialize
    g_scores = np.full(n, np.inf)
    g_scores[instance.start] = 0
    f_scores = g_scores + h_vals
    
    in_open = np.zeros(n, dtype=bool)
    in_open[instance.start] = True
    in_closed = np.zeros(n, dtype=bool)
    
    num_steps = 0
    
    # Build FULL tree - don't stop at goal
    while np.any(in_open):
        # Find minimum f-score in open set
        open_f = np.where(in_open, f_scores, np.inf)
        f_min = np.min(open_f)
        
        if f_min == np.inf:
            break
        
        # Parallel A*: expand all nodes with f < f_min + delta
        edge_weights = instance.adj[instance.adj > 0]
        if len(edge_weights) > 0:
            delta = np.mean(edge_weights) * epsilon
        else:
            delta = 0.01
        
        expand_mask = in_open & (f_scores < f_min + delta)
        expand_nodes = np.where(expand_mask)[0]
        
        if len(expand_nodes) == 0:
            break
        
        # Expand all selected nodes in parallel
        for node in expand_nodes:
            in_open[node] = False
            in_closed[node] = True
            
            # DON'T stop at goal - continue building full tree
            
            # Relax neighbors
            for neighbor in instance.out_nodes[node]:
                if in_closed[neighbor]:
                    continue
                
                tentative_g = g_scores[node] + instance.adj[node, neighbor]
                
                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + h_vals[neighbor]
                    in_open[neighbor] = True
        
        num_steps += 1
    
    return num_steps + 1  # +1 for initial state


def get_sequential_trace_length(algorithm_name: str, instance: ProblemInstance) -> int:
    """Run algorithm and return trace length."""
    if algorithm_name == 'bfs':
        node_fts, _, _ = bfs(instance)
    elif algorithm_name == 'dfs':
        node_fts, _, _ = dfs(instance)
    elif algorithm_name == 'dijkstra':
        node_fts, _, _ = dijkstra(instance)
    elif algorithm_name == 'a_star':
        node_fts, _, _ = a_star(instance, build_full_tree=True)
    elif algorithm_name == 'mst':
        node_fts, _, _ = mst(instance)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return node_fts.shape[0]

# Update the analyze_trace_lengths function:
def analyze_trace_lengths(
    algorithms: list,
    sizes: list,
    graph_type: str = 'er',
    num_samples: int = 20,
    seed: int = 42
):
    """
    Analyze trace lengths for various algorithms and graph sizes.
    
    Returns:
        dict: {algorithm: {size: {'seq': [...], 'par': [...]}}}
    """
    np.random.seed(seed)
    
    results = defaultdict(lambda: defaultdict(lambda: {'seq': [], 'par': []}))
    
    # Create sampler
    config = SimpleConfig(
        algorithm=algorithms[0] if algorithms else 'bfs',
        graph_type=graph_type,
        edge_weights=('dijkstra' in algorithms or 'a_star' in algorithms or 'mst' in algorithms)
    )
    
    if graph_type == 'geometric':
        sampler = GeometricGraphSampler(config)
    elif graph_type == 'grid':
        sampler = GridGraphSampler(config)
    elif graph_type == 'roadmap':
        sampler = RoadmapGraphSampler(config)
    else:
        sampler = ErdosRenyiGraphSampler(config)
    
    for size in sizes:
        print(f"\nProcessing size {size}...")
        
        for sample_idx in range(num_samples):
            # Generate instance
            instance = sampler(size)
            actual_size = instance.adj.shape[0]
            
            for algo in algorithms:
                # Sequential trace length
                try:
                    seq_len = get_sequential_trace_length(algo, instance)
                    results[algo][size]['seq'].append(seq_len)
                except Exception as e:
                    print(f"  Error in {algo}: {e}")
                    raise e
                
                # Parallel trace length (algorithm-specific)
                try:
                    if algo == 'bfs':
                        par_len = parallel_bfs_trace_length(instance)
                        results[algo][size]['par'].append(par_len)
                    elif algo == 'dijkstra':
                        par_len = parallel_dijkstra_trace_length(instance)
                        results[algo][size]['par'].append(par_len)
                    elif algo == 'a_star':
                        par_len = parallel_a_star_trace_length(instance)
                        results[algo][size]['par'].append(par_len)
                    elif algo == 'mst':
                        par_len = parallel_mst_trace_length(instance)
                        results[algo][size]['par'].append(par_len)
                    # DFS has no parallel version - skip
                except Exception as e:
                    print(f"  Error in parallel {algo}: {e}")
            
            if (sample_idx + 1) % 5 == 0:
                print(f"  Completed {sample_idx + 1}/{num_samples} samples")
    
    return results


def compute_graph_diameter(instance: ProblemInstance) -> int:
    """Compute the diameter of the graph (longest shortest path)."""
    n = instance.adj.shape[0]
    G = nx.from_numpy_array(instance.adj)
    
    if not nx.is_connected(G):
        return -1
    
    return nx.diameter(G)


def plot_results(results: dict, sizes: list, output_path: str = 'trace_length_analysis.png'):
    """Create visualization of trace length analysis."""
    num_algos = len(results)
    fig, axes = plt.subplots(1, num_algos, figsize=(5 * num_algos, 5))
    
    if num_algos == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax, (algo, size_data) in zip(axes, results.items()):
        seq_means = []
        seq_stds = []
        par_means = []
        par_stds = []
        valid_sizes = []
        
        for size in sizes:
            if size in size_data:
                seq_vals = size_data[size]['seq']
                par_vals = size_data[size]['par']
                
                if seq_vals:
                    valid_sizes.append(size)
                    seq_means.append(np.mean(seq_vals))
                    seq_stds.append(np.std(seq_vals))
                    
                    if par_vals:
                        par_means.append(np.mean(par_vals))
                        par_stds.append(np.std(par_vals))
                    else:
                        par_means.append(None)
                        par_stds.append(None)
        
        valid_sizes = np.array(valid_sizes)
        seq_means = np.array(seq_means)
        seq_stds = np.array(seq_stds)
        
        # Plot sequential
        ax.errorbar(valid_sizes, seq_means, yerr=seq_stds, 
                   label='Sequential', color=colors[0], marker='o', capsize=3)
        
        # Plot parallel if available
        if any(p is not None for p in par_means):
            par_means_clean = [p for p in par_means if p is not None]
            par_stds_clean = [s for s, p in zip(par_stds, par_means) if p is not None]
            sizes_clean = [s for s, p in zip(valid_sizes, par_means) if p is not None]
            
            ax.errorbar(sizes_clean, par_means_clean, yerr=par_stds_clean,
                       label='Parallel', color=colors[1], marker='s', capsize=3)
        
        # Plot O(n) reference line
        ax.plot(valid_sizes, valid_sizes, '--', color='gray', alpha=0.5, label='O(n)')
        
        # Plot O(log n) reference for parallel
        log_ref = np.log2(valid_sizes) * 5  # Scaled for visibility
        ax.plot(valid_sizes, log_ref, ':', color='gray', alpha=0.5, label='O(log n) scaled')

        #Plot O(sqrt(n)) reference 
        sqrt_ref = np.sqrt(valid_sizes) * 5  # Scaled for visibility
        ax.plot(valid_sizes, sqrt_ref, '-.', color='gray', alpha=0.5, label='O(√n) scaled')
        
        ax.set_xlabel('Graph Size (n)')
        ax.set_ylabel('Trace Length (steps)')
        ax.set_title(f'{algo.upper()}')
        ax.legend()
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.grid(True, alpha=0.3)

        ax.set_xticks(valid_sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def print_summary(results: dict, sizes: list):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("TRACE LENGTH ANALYSIS SUMMARY")
    print("=" * 80)
    
    for algo, size_data in results.items():
        print(f"\n{algo.upper()}")
        print("-" * 40)
        print(f"{'Size':>8} | {'Seq Mean':>10} | {'Par Mean':>10} | {'Speedup':>10}")
        print("-" * 40)
        
        for size in sizes:
            if size in size_data:
                seq_vals = size_data[size]['seq']
                par_vals = size_data[size]['par']
                
                if seq_vals:
                    seq_mean = np.mean(seq_vals)
                    
                    if par_vals:
                        par_mean = np.mean(par_vals)
                        speedup = seq_mean / par_mean
                        print(f"{size:>8} | {seq_mean:>10.1f} | {par_mean:>10.1f} | {speedup:>10.1f}x")
                    else:
                        print(f"{size:>8} | {seq_mean:>10.1f} | {'N/A':>10} | {'N/A':>10}")


def main():
    parser = argparse.ArgumentParser(description='Analyze algorithm trace lengths')
    parser.add_argument('--algorithms', '-a', nargs='+', 
                       default=['bfs', 'dijkstra'],
                       choices=['bfs', 'dfs', 'dijkstra', 'a_star', 'mst'],
                       help='Algorithms to analyze')
    parser.add_argument('--sizes', '-s', nargs='+', type=int,
                       default=[16, 32, 64, 128, 256, 512],
                       help='Graph sizes to test')
    parser.add_argument('--graph_type', '-g', type=str, default='er',
                       choices=['er', 'geometric', 'grid', 'roadmap'],
                       help='Type of graphs to generate')
    parser.add_argument('--num_samples', '-n', type=int, default=20,
                       help='Number of samples per size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', '-o', type=str, default='trace_length_analysis.png',
                       help='Output plot filename')
    
    args = parser.parse_args()
    
    print(f"Analyzing trace lengths for: {args.algorithms}")
    print(f"Graph sizes: {args.sizes}")
    print(f"Graph type: {args.graph_type}")
    print(f"Samples per size: {args.num_samples}")
    
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
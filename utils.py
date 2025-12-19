import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import torch
from torch_geometric.utils import group_argsort, scatter, softmax
import time
import os
import shutil


# def reverse_edge_index(edge_index): #dunno how this was supposed to work lol 
#     rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
#     rev_index = torch.argsort(rev_edge_index, stable=True)[0]
#     assert torch.all(edge_index[:, rev_index] == rev_edge_index)
#     return rev_index

def reverse_edge_index(edge_index): #now we actually inverse and keep track of
    src, dst = edge_index
    n = edge_index.size(1)

    # map (u,v) -> list of indices
    edge_map = defaultdict(deque)
    for i in range(n):
        edge_map[(src[i].item(), dst[i].item())].append(i)

    rev_index = torch.empty(n, dtype=torch.long)

    for i in range(n):
        u = dst[i].item()
        v = src[i].item()
        rev_index[i] = edge_map[(u, v)].popleft()

    # correctness check
    assert torch.all(edge_index[:, rev_index] == edge_index.flip(0))
    return rev_index


def temp_by_step(step, high, low, num_steps, temp_on_eval):
    if step == -1:
        return temp_on_eval
    assert step >= 0 and low <= high
    return np.geomspace(high, low, max(num_steps, step + 1))[step]


def gumbel_softmax(logits, index, tau=1.0, use_noise=False):
    if use_noise:
        noise = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        logits = logits + noise
    if tau == 0.0:
        y_hard = 1.0 * (group_argsort(logits, index, descending=True, stable=True) == 0)
        return y_hard
    logits = logits / tau
    return softmax(logits, index)


def from_binary_states(binary_states):
    assert binary_states.ndim == 2
    states = torch.zeros_like(binary_states[:, 0])
    num_degs = binary_states.shape[1]
    degs = torch.tensor([2**deg for deg in range(num_degs)]).to(binary_states.device)

    states = (binary_states * torch.unsqueeze(degs, 0)).sum(-1)
    return states.long()


def node_pointer_loss(logits, gt, index):
    eps = 1e-12
    log_probs = torch.log(softmax(logits, index) + eps)
    return -scatter(gt * log_probs, index).mean()


def pointer_accuracy(graph, prediction):
    edge_index = graph.edge_index
    is_predicted_pointer = 1.0 * (
        group_argsort(prediction, edge_index[0], descending=True, stable=True) == 0
    )

    assert is_predicted_pointer.sum() == graph.num_nodes
    return (graph.y * is_predicted_pointer).sum() / graph.num_nodes


def pointer_accuracy_graph_level(graph, prediction):
    return 1.0 * (pointer_accuracy(graph, prediction) == 1.0)


def node_mask_accuracy(graph, prediction):
    pred_mask = prediction > 0.0
    return (1.0 * (pred_mask == graph.y)).mean()


def node_mask_accuracy_graph_level(graph, prediction):
    return 1.0 * (node_mask_accuracy(graph, prediction) == 1.0)


def evaluate(model, dataloader, calculators):
    scores = defaultdict(float)
    total_points = 0
    device = model.parameters().__next__().device
    for data in dataloader:
        data = data.to(device)
        batched_prediction, _ = model(data)
        for batch_idx, graph in enumerate(data.to_data_list()):
            batch_pred_idx = (
                data.batch[data.edge_index[0]]
                if model.output_type == "pointer"
                else data.batch
            )
            prediction = batched_prediction[batch_pred_idx == batch_idx]

            for calculator in calculators:
                value = calculator(graph, prediction)
                scores[calculator.__name__] += (
                    value if isinstance(value, float) else value.item()
                )
            total_points += 1
    for calculator in calculators:
        scores[calculator.__name__] /= total_points

    return scores


def evaluate_multitask(model, dataloader, calculators, algorithm: str):
    """
    Evaluate a multitask model with a specific algorithm context.
    
    This ensures the correct algorithm-specific encoders/decoders are used
    during evaluation.
    
    Args:
        model: The multitask model
        dataloader: DataLoader for evaluation data
        calculators: Tuple of metric functions
        algorithm: Algorithm name to use for evaluation
    
    Returns:
        Dictionary of metric scores
    """
    scores = defaultdict(float)
    total_points = 0
    device = model.parameters().__next__().device
    for data in dataloader:
        data = data.to(device)
        # Pass the algorithm to use correct encoder/decoder components
        batched_prediction, _ = model(data, multitask_algorithm=algorithm)
        for batch_idx, graph in enumerate(data.to_data_list()):
            batch_pred_idx = (
                data.batch[data.edge_index[0]]
                if model.output_type == "pointer"
                else data.batch
            )
            prediction = batched_prediction[batch_pred_idx == batch_idx]

            for calculator in calculators:
                value = calculator(graph, prediction)
                scores[calculator.__name__] += (
                    value if isinstance(value, float) else value.item()
                )
            total_points += 1
    for calculator in calculators:
        scores[calculator.__name__] /= total_points

    return scores

#evaluates but also prints the predicted pointer structure and the true output structure for reference 
def evaluate_print(model, dataloader, calculators, output_path="evaluation_results.json", 
                   algorithm: Optional[str] = None, step: Optional[int] = None,
                   split: str = "test", print_results: bool = True):
    """
    Evaluates the model and saves detailed results including predicted and true pointers.
    
    Args:
        model: The trained model to evaluate
        dataloader: DataLoader with evaluation data
        calculators: Tuple of metric functions to compute
        output_path: Path to save the results JSON file
        algorithm: Algorithm name for multitask models (None for single-task)
        step: Current training step (for checkpoint naming)
        split: Data split name ("train", "val", "test")
        print_results: Whether to print summary results to console
    
    Returns:
        Dictionary with aggregate scores
    """
    scores = defaultdict(float)
    total_points = 0
    
    # Store detailed results for each graph
    graph_results:dict[int, dict] = {}
    total_pointer_mistakes = 0
    total_pointers = 0
    correct_graphs = 0
    
    device = model.parameters().__next__().device
    for data in dataloader:
        data = data.to(device)
        # Support multitask by passing algorithm if provided
        if algorithm is not None:
            batched_prediction, _ = model(data, multitask_algorithm=algorithm)
        else:
            batched_prediction, _ = model(data)
        
        for batch_idx, graph in enumerate(data.to_data_list()):
            batch_pred_idx = (
                data.batch[data.edge_index[0]]
                if model.output_type == "pointer"
                else data.batch
            )
            prediction = batched_prediction[batch_pred_idx == batch_idx]
            
            # Extract predicted and true pointers
            edge_index = graph.edge_index
            
            if model.output_type == "pointer":
                # Get predicted pointer indices (argmax per source node)
                is_predicted_pointer = (
                    group_argsort(prediction, edge_index[0], descending=True, stable=True) == 0
                )
                
                # Build pointer mappings: node -> pointed_to_node
                pred_pointers = {}
                true_pointers = {}
                
                for i in range(edge_index.shape[1]):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()
                    
                    if is_predicted_pointer[i]:
                        pred_pointers[src] = dst
                    if graph.y[i] > 0.5:  # True pointer
                        true_pointers[src] = dst
                
                # Count mistakes for this graph
                graph_mistakes = 0
                num_nodes = graph.num_nodes
                for node in range(num_nodes):
                    pred_ptr = pred_pointers.get(node, node)  # Default to self
                    true_ptr = true_pointers.get(node, node)
                    if pred_ptr != true_ptr:
                        graph_mistakes += 1
                
                total_pointer_mistakes += graph_mistakes
                total_pointers += num_nodes
                
                is_correct = graph_mistakes == 0
                if is_correct:
                    correct_graphs += 1
            else:
                # For node_mask output type
                pred_mask = (prediction > 0.0).cpu().numpy().tolist()
                true_mask = graph.y.cpu().numpy().tolist()
                pred_pointers = {"mask": pred_mask}
                true_pointers = {"mask": true_mask}
                graph_mistakes = sum(1 for p, t in zip(pred_mask, true_mask) if p != t)
                total_pointer_mistakes += graph_mistakes
                total_pointers += len(pred_mask)
                is_correct = graph_mistakes == 0
                if is_correct:
                    correct_graphs += 1
            
            # Compute individual metrics for this graph
            graph_metrics = {}
            for calculator in calculators:
                value = calculator(graph, prediction)
                value = value if isinstance(value, float) else value.item()
                graph_metrics[calculator.__name__] = value
                scores[calculator.__name__] += value
            
            # Store graph result
            graph_result = {
                "graph_idx": total_points,
                "num_nodes": graph.num_nodes,
                "num_edges": graph.edge_index.shape[1],
                "predicted_pointers": {str(k): v for k, v in pred_pointers.items()},
                "true_pointers": {str(k): v for k, v in true_pointers.items()},
                "pointer_mistakes": graph_mistakes,
                "is_correct": is_correct,
                "metrics": graph_metrics,
            }
            
            # Add edge_index for reference
            graph_result["edge_index"] = edge_index.cpu().numpy().tolist()
            
            graph_results[total_points] = graph_result
            total_points += 1
    
    # Compute aggregate scores
    for calculator in calculators:
        scores[calculator.__name__] /= total_points
    
    # Build final results
    results = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": algorithm,
        "step": step,
        "split": split,
        "total_graphs": total_points,
        "summary": {
            "accuracy": scores.get("pointer_accuracy", scores.get("node_mask_accuracy", 0.0)),
            "graph_level_accuracy": correct_graphs / total_points if total_points > 0 else 0.0,
            "total_pointer_mistakes": total_pointer_mistakes,
            "total_pointers": total_pointers,
            "mistake_rate": total_pointer_mistakes / total_pointers if total_pointers > 0 else 0.0,
            "correct_graphs": correct_graphs,
            "incorrect_graphs": total_points - correct_graphs,
        },
        "metrics": dict(scores),
        "graphs": graph_results,
    }
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    if print_results:
        algo_str = f"[{algorithm}] " if algorithm else ""
        step_str = f"step {step} " if step else ""
        print(f"{algo_str}Evaluation results {step_str}saved to: {output_path}")
        print(f"Summary:")
        print(f"  Total graphs: {total_points}")
        print(f"  Accuracy: {results['summary']['accuracy']:.4f}")
        print(f"  Graph-level accuracy: {results['summary']['graph_level_accuracy']:.4f}")
        print(f"  Total pointer mistakes: {total_pointer_mistakes} / {total_pointers}")
        print(f"  Mistake rate: {results['summary']['mistake_rate']:.4f}")
        print(f"  Correct graphs: {correct_graphs} / {total_points}")
        
    return scores


def filter_graphs(results_path, filter_fn, n=None):
    """
    Returns up to N graphs from evaluation results that match a filter condition.
    
    Args:
        results_path: Path to the JSON file created by evaluate_print
        filter_fn: A callable that takes a graph_result dict and returns True/False
                   Examples:
                     - lambda g: g["is_correct"] == True
                     - lambda g: g["pointer_mistakes"] > 0
                     - lambda g: g["num_nodes"] >= 10 and not g["is_correct"]
                     - lambda g: g["metrics"]["pointer_accuracy"] < 0.5
        n: Maximum number of graphs to return (None = return all matching)
    
    Returns:
        List of graph_result dicts that match the filter
    
    Examples:
        # Get 5 incorrect graphs
        incorrect = filter_graphs("results.json", lambda g: not g["is_correct"], n=5)
        
        # Get all graphs with more than 2 mistakes
        many_mistakes = filter_graphs("results.json", lambda g: g["pointer_mistakes"] > 2)
        
        # Get graphs with low accuracy
        low_acc = filter_graphs("results.json", lambda g: g["metrics"]["pointer_accuracy"] < 0.8, n=10)
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    
    matching = []
    for graph_idx, graph_data in results["graphs"].items():
        if filter_fn(graph_data):
            matching.append(graph_data)
            if n is not None and len(matching) >= n:
                break
    
    print(f"Found {len(matching)} graphs matching filter" + (f" (limited to {n})" if n else ""))
    return matching


def get_graph_indices(results_path, filter_fn, n=None):
    """
    Returns the indices of up to N graphs that match a filter condition.
    Useful for passing to visualize_pointers.
    
    Args:
        results_path: Path to the JSON file created by evaluate_print
        filter_fn: A callable that takes a graph_result dict and returns True/False
        n: Maximum number of indices to return (None = return all matching)
    
    Returns:
        List of graph indices (integers)
    
    Example:
        # Get indices of first 5 incorrect graphs, then visualize one
        indices = get_graph_indices("results.json", lambda g: not g["is_correct"], n=5)
        visualize_pointers("results.json", indices[0])
    """
    graphs = filter_graphs(results_path, filter_fn, n)
    return [g["graph_idx"] for g in graphs]


def count_graphs(results_path, filter_fn=None):
    """
    Counts the number of graphs that match a filter condition.
    
    Args:
        results_path: Path to the JSON file created by evaluate_print
        filter_fn: A callable that takes a graph_result dict and returns True/False
                   If None, returns total graph count.
    
    Returns:
        Tuple of (matching_count, total_count, percentage)
    
    Examples:
        # Count incorrect graphs
        count, total, pct = count_graphs("results.json", lambda g: not g["is_correct"])
        print(f"{count}/{total} graphs are incorrect ({pct:.1%})")
        
        # Count graphs with mistakes
        count_graphs("results.json", lambda g: g["pointer_mistakes"] > 0)
        
        # Count graphs with high accuracy
        count_graphs("results.json", lambda g: g["metrics"]["pointer_accuracy"] >= 0.9)
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    
    total = len(results["graphs"])
    
    if filter_fn is None:
        return total, total, 1.0
    
    count = sum(1 for graph_data in results["graphs"].values() if filter_fn(graph_data))
    percentage = count / total if total > 0 else 0.0
    
    print(f"{count}/{total} graphs match filter ({percentage:.1%})")
    return count, total, percentage


def visualize_pointers(results_path, graph_idx, output_path=None):
    """
    Visualizes the predicted and true pointers for a specific graph from evaluation results.
    
    Args:
        results_path: Path to the JSON file created by evaluate_print
        graph_idx: Index of the graph to visualize
        output_path: Optional path to save the figure (if None, displays interactively)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Get the graph
    graph_data = results["graphs"][graph_idx] if graph_idx < len(results["graphs"]) else None
    
    if graph_data is None:
        raise ValueError(f"Graph with idx {graph_idx} not found in results")
    
    num_nodes = graph_data["num_nodes"]
    edge_index = graph_data["edge_index"]
    pred_pointers = {int(k): v for k, v in graph_data["predicted_pointers"].items()}
    true_pointers = {int(k): v for k, v in graph_data["true_pointers"].items()}
    
    # Build adjacency for layout
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(len(edge_index[0])):
        src, dst = edge_index[0][i], edge_index[1][i]
        if src != dst:  # Skip self-loops for layout
            G.add_edge(src, dst)
    
    # Compute layout
    pos = nx.spring_layout(G, seed=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    titles = ["Predicted Pointers", "True Pointers"]
    pointer_dicts = [pred_pointers, true_pointers]
    
    for ax, title, pointers in zip(axes, titles, pointer_dicts):
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw base edges (gray, thin)
        for i in range(len(edge_index[0])):
            src, dst = edge_index[0][i], edge_index[1][i]
            if src != dst:
                x = [pos[src][0], pos[dst][0]]
                y = [pos[src][1], pos[dst][1]]
                ax.plot(x, y, 'lightgray', linewidth=1, zorder=1)
        
        # Draw pointer edges (colored arrows)
        for src, dst in pointers.items():
            if src == dst:
                # Self-pointer: draw a small loop
                circle = plt.Circle(pos[src], 0.08, fill=False, color='blue', linewidth=2, zorder=2)
                ax.add_patch(circle)
            else:
                # Draw arrow from src to dst
                dx = pos[dst][0] - pos[src][0]
                dy = pos[dst][1] - pos[src][1]
                ax.annotate(
                    "", xy=(pos[dst][0], pos[dst][1]), xytext=(pos[src][0], pos[src][1]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                    zorder=3
                )
        
        # Draw nodes
        node_colors = []
        for node in range(num_nodes):
            pred_ptr = pred_pointers.get(node, node)
            true_ptr = true_pointers.get(node, node)
            if pred_ptr == true_ptr:
                node_colors.append('lightgreen')  # Correct
            else:
                node_colors.append('salmon')  # Mistake
        
        for node in range(num_nodes):
            circle = plt.Circle(pos[node], 0.06, color=node_colors[node], ec='black', linewidth=1.5, zorder=4)
            ax.add_patch(circle)
            ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add legend
    correct_patch = mpatches.Patch(color='lightgreen', label='Correct pointer')
    mistake_patch = mpatches.Patch(color='salmon', label='Incorrect pointer')
    fig.legend(handles=[correct_patch, mistake_patch], loc='lower center', ncol=2, fontsize=10)
    
    # Add summary info
    fig.suptitle(
        f"Graph {graph_idx} | Nodes: {num_nodes} | Mistakes: {graph_data['pointer_mistakes']} | "
        f"Accuracy: {graph_data['metrics'].get('pointer_accuracy', graph_data['metrics'].get('node_mask_accuracy', 0)):.2%}",
        fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


incorrect_filter = lambda g: not g["is_correct"]
correct_filter = lambda g: g["is_correct"]
low_acc_filter = lambda g: g["metrics"].get("pointer_accuracy", g["metrics"].get("node_mask_accuracy", 1.0)) < 0.8
n_mistakes_filter = lambda n: lambda g: g["pointer_mistakes"] >= n

class ModelSaver:
    def __init__(self, models_directory: str, model_name: str):
        Path(models_directory).mkdir(parents=True, exist_ok=True)
        model_name = "{}/{}".format(models_directory, model_name)

        self.best_vals = defaultdict(float)
        self.model_name = model_name

    def visit(self, model, metrics_stat):
        for metric in metrics_stat:
            if metrics_stat[metric] > self.best_vals[metric]:
                self.best_vals[metric] = metrics_stat[metric]
                self.save(model, metric + "_best")
        self.save(model, "last")

    def save(self, model, suffix):
        path = "{}_{}.pt".format(self.model_name, suffix)
        print("saving model: ", path)
        torch.save(model.state_dict(), path)


NODE_POINTER_METRICS = (pointer_accuracy, pointer_accuracy_graph_level)
NODE_MASK_METRICS = (node_mask_accuracy, node_mask_accuracy_graph_level)
METRICS = {"pointer": NODE_POINTER_METRICS, "node_mask": NODE_MASK_METRICS}


# =============================================================================
# MULTITASK DECORATOR
# =============================================================================
# This decorator enables algorithm-specific encoder/decoder components while
# sharing the latent space processor. It handles:
# 1. Encoder side: self.emb embeddings in StatesEncoder, SelectBest
# 2. Decoder side: node_projections, edge_projections in StatesBottleneck
# 3. Algorithm spec for loss calculation in StatesBottleneck
# =============================================================================

from generate_data import SPEC
from contextlib import contextmanager
import copy


def _find_embedding_modules(module, prefix=""):
    """
    Recursively find all modules that have 'emb' as an Embedding attribute.
    Returns list of (parent_module, attr_name, embedding_module, full_path).
    """
    results = []
    for name, child in module.named_children():
        full_path = f"{prefix}.{name}" if prefix else name
        # Check if this child has an 'emb' attribute that is an Embedding
        if hasattr(child, 'emb') and isinstance(child.emb, torch.nn.Embedding):
            results.append((child, 'emb', child.emb, full_path))
        # Recurse into children
        results.extend(_find_embedding_modules(child, full_path))
    return results


def _find_states_bottleneck(module, prefix=""):
    """
    Recursively find StatesBottleneck modules that have projections and spec.
    Returns list of (module, full_path).
    """
    results = []
    for name, child in module.named_children():
        full_path = f"{prefix}.{name}" if prefix else name
        # Check if this is a StatesBottleneck (has node_projections, edge_projections, spec)
        if (hasattr(child, 'node_projections') and 
            hasattr(child, 'edge_projections') and 
            hasattr(child, 'spec')):
            results.append((child, full_path))
        # Recurse into children
        results.extend(_find_states_bottleneck(child, full_path))
    return results


def _clone_embedding(emb: torch.nn.Embedding) -> torch.nn.Embedding:
    """Create a new Embedding with the same configuration."""
    new_emb = torch.nn.Embedding(emb.num_embeddings, emb.embedding_dim, 
                                  padding_idx=emb.padding_idx)
    # Initialize with same weights (can be changed to random if desired)
    new_emb.weight.data.copy_(emb.weight.data)
    return new_emb


def _clone_projection_list(proj_list: torch.nn.ModuleList) -> torch.nn.ModuleList:
    """Clone a ModuleList of Linear layers."""
    new_list = torch.nn.ModuleList()
    for proj in proj_list:
        new_proj = torch.nn.Linear(proj.in_features, proj.out_features, 
                                    bias=proj.bias is not None)
        new_proj.weight.data.copy_(proj.weight.data)
        if proj.bias is not None:
            new_proj.bias.data.copy_(proj.bias.data)
        new_list.append(new_proj)
    return new_list


class MultitaskRegistry:
    """
    Stores algorithm-specific components and handles runtime swapping.
    Attached to the decorated module as _multitask_registry.
    """
    def __init__(self, num_algorithms: int):
        self.num_algorithms = num_algorithms
        self.algorithm_dict: Dict[str, int] = {}  # algorithm_name -> index
        
        # Stores: { (parent_module_id, attr_name): ModuleList of algorithm-specific modules }
        self.embedding_variants: Dict[tuple, torch.nn.ModuleList] = {}
        
        # Stores: { bottleneck_module_id: { 'node_proj': ModuleList, 'edge_proj': ModuleList, 'specs': list } }
        self.bottleneck_variants: Dict[int, dict] = {}
        
        # Original modules for restoration
        self.original_embeddings: Dict[tuple, torch.nn.Module] = {}
        self.original_bottlenecks: Dict[int, dict] = {}
    
    def register_embedding(self, parent: torch.nn.Module, attr_name: str, 
                          original_emb: torch.nn.Embedding) -> torch.nn.ModuleList:
        """Register an embedding for multitask and create algorithm variants."""
        key = (id(parent), attr_name)
        if key in self.embedding_variants:
            return self.embedding_variants[key]
        
        # Create list: first is the original, rest are clones
        variants = torch.nn.ModuleList([original_emb])
        for _ in range(self.num_algorithms - 1):
            variants.append(_clone_embedding(original_emb))
        
        self.embedding_variants[key] = variants
        self.original_embeddings[key] = original_emb
        return variants
    
    def register_bottleneck(self, bottleneck: torch.nn.Module) -> dict:
        """Register a StatesBottleneck for multitask."""
        key = id(bottleneck)
        if key in self.bottleneck_variants:
            return self.bottleneck_variants[key]
        
        # Create variants for projections
        node_proj_variants = torch.nn.ModuleList([bottleneck.node_projections])
        edge_proj_variants = torch.nn.ModuleList([bottleneck.edge_projections])
        
        for _ in range(self.num_algorithms - 1):
            node_proj_variants.append(_clone_projection_list(bottleneck.node_projections))
            edge_proj_variants.append(_clone_projection_list(bottleneck.edge_projections))
        
        # Specs will be set dynamically based on algorithm name
        variants = {
            'node_proj': node_proj_variants,
            'edge_proj': edge_proj_variants,
            'specs': [bottleneck.spec] * self.num_algorithms,  # Will be updated on register_algorithm
        }
        
        self.bottleneck_variants[key] = variants
        self.original_bottlenecks[key] = {
            'node_proj': bottleneck.node_projections,
            'edge_proj': bottleneck.edge_projections,
            'spec': bottleneck.spec,
        }
        return variants
    
    def get_or_register_algorithm(self, algorithm_name: str) -> int:
        """Get index for algorithm, registering it if new."""
        if algorithm_name not in self.algorithm_dict:
            idx = len(self.algorithm_dict)
            if idx >= self.num_algorithms:
                raise ValueError(
                    f"Cannot register algorithm '{algorithm_name}': already at max "
                    f"({self.num_algorithms}). Registered: {list(self.algorithm_dict.keys())}"
                )
            self.algorithm_dict[algorithm_name] = idx
            
            # Update specs for bottlenecks - pad with MASK (0) to match projection counts
            if algorithm_name in SPEC:
                from generate_data import MASK  # MASK = 0
                for key, variants in self.bottleneck_variants.items():
                    original_spec = SPEC[algorithm_name]
                    
                    # Get the projection counts from the first variant
                    # (all variants have the same number of projections)
                    num_node_proj = len(variants['node_proj'][0])
                    num_edge_proj = len(variants['edge_proj'][0])
                    
                    # Pad the spec to match projection counts
                    # original_spec = ((node_hints...), (edge_hints...))
                    node_spec = original_spec[0] + (MASK,) * (num_node_proj - len(original_spec[0]))
                    edge_spec = original_spec[1] + (MASK,) * (num_edge_proj - len(original_spec[1]))
                    
                    variants['specs'][idx] = (node_spec, edge_spec)
        
        return self.algorithm_dict[algorithm_name]


class MultitaskContext:
    """
    Context manager that swaps in algorithm-specific components during forward pass.
    """
    def __init__(self, model: torch.nn.Module, algorithm_name: Optional[str]):
        self.model = model
        self.algorithm_name = algorithm_name
        self.registry: Optional[MultitaskRegistry] = getattr(model, '_multitask_registry', None)
        self.swapped_embeddings: list = []
        self.swapped_bottlenecks: list = []
    
    def __enter__(self):
        if self.registry is None or self.algorithm_name is None:
            return self
        
        alg_idx = self.registry.get_or_register_algorithm(self.algorithm_name)
        
        # Swap embeddings
        for (parent_id, attr_name), variants in self.registry.embedding_variants.items():
            # Find the parent module by id
            for parent, name, emb, path in _find_embedding_modules(self.model):
                if id(parent) == parent_id and name == attr_name:
                    original = getattr(parent, attr_name)
                    setattr(parent, attr_name, variants[alg_idx])
                    self.swapped_embeddings.append((parent, attr_name, original))
                    break
        
        # Swap bottleneck components
        for bottleneck_id, variants in self.registry.bottleneck_variants.items():
            for bottleneck, path in _find_states_bottleneck(self.model):
                if id(bottleneck) == bottleneck_id:
                    orig_node = bottleneck.node_projections
                    orig_edge = bottleneck.edge_projections
                    orig_spec = bottleneck.spec
                    
                    bottleneck.node_projections = variants['node_proj'][alg_idx]
                    bottleneck.edge_projections = variants['edge_proj'][alg_idx]
                    bottleneck.spec = variants['specs'][alg_idx]
                    
                    self.swapped_bottlenecks.append((
                        bottleneck, orig_node, orig_edge, orig_spec
                    ))
                    break
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore embeddings
        for parent, attr_name, original in self.swapped_embeddings:
            setattr(parent, attr_name, original)
        
        # Restore bottlenecks
        for bottleneck, orig_node, orig_edge, orig_spec in self.swapped_bottlenecks:
            bottleneck.node_projections = orig_node
            bottleneck.edge_projections = orig_edge
            bottleneck.spec = orig_spec
        
        return False


def multitask(cls):
    """
    Class decorator to add multitask functionality to a model.
    
    Usage:
        @multitask
        class Dnar(Module):
            ...
    
    Then instantiate with:
        model = Dnar(config, multitask_num_algorithms=5)
    
    And call forward with:
        output, loss = model(batch, multitask_algorithm="dijkstra")
    
    The decorator will automatically:
    1. Find all self.emb embeddings in submodules and create algorithm-specific copies
    2. Find StatesBottleneck modules and create algorithm-specific projections
    3. Swap the correct components based on multitask_algorithm at runtime
    
    Backpropagation works correctly because:
    - All algorithm-specific modules are registered as submodules (in model.parameters())
    - The computation graph holds references to the actual weight tensors used
    - Restoring attributes after forward doesn't affect the gradient computation
    """
    original_init = cls.__init__
    original_forward = cls.forward

    def new_init(self, *args, multitask_num_algorithms: Optional[int] = None, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # If multitask is enabled, set up the registry
        if multitask_num_algorithms is not None and multitask_num_algorithms > 1:
            registry = MultitaskRegistry(multitask_num_algorithms)
            
            # Find and register all embeddings
            embedding_modules = _find_embedding_modules(self)
            for parent, attr_name, emb, path in embedding_modules:
                variants = registry.register_embedding(parent, attr_name, emb)
                # Store variants as a proper submodule for parameter tracking
                safe_name = path.replace('.', '_')
                setattr(self, f'_multitask_emb_{safe_name}', variants)
            
            # Find and register all StatesBottleneck modules
            bottleneck_modules = _find_states_bottleneck(self)
            for bottleneck, path in bottleneck_modules:
                variants = registry.register_bottleneck(bottleneck)
                safe_name = path.replace('.', '_')
                setattr(self, f'_multitask_node_proj_{safe_name}', variants['node_proj'])
                setattr(self, f'_multitask_edge_proj_{safe_name}', variants['edge_proj'])
            
            self._multitask_registry = registry
            self._multitask_enabled = True
        else:
            self._multitask_enabled = False

    def new_forward(self, *args, multitask_algorithm: Optional[str] = None, **kwargs):
        # Check if multitask is expected but not configured
        if multitask_algorithm is not None and not getattr(self, '_multitask_enabled', False):
            raise ValueError(
                f"Received multitask_algorithm='{multitask_algorithm}' but multitask is not enabled. "
                f"Set multitask_num_algorithms > 1 when creating the model."
            )
        
        # Use context manager to swap components during forward
        with MultitaskContext(self, multitask_algorithm):
            return original_forward(self, *args, **kwargs)
    
    def get_multitask_info(self):
        """
        Get information about the multitask configuration.
        Useful for debugging and verifying parameter registration.
        """
        if not getattr(self, '_multitask_enabled', False):
            return {"enabled": False}
        
        registry = self._multitask_registry
        
        # Count parameters per algorithm variant
        param_counts = {}
        
        # Count embedding parameters
        emb_params = {i: 0 for i in range(registry.num_algorithms)}
        for key, variants in registry.embedding_variants.items():
            for i, emb in enumerate(variants):
                emb_params[i] += sum(p.numel() for p in emb.parameters())
        
        # Count projection parameters
        proj_params = {i: 0 for i in range(registry.num_algorithms)}
        for key, variants in registry.bottleneck_variants.items():
            for i in range(registry.num_algorithms):
                for proj in variants['node_proj'][i]:
                    proj_params[i] += sum(p.numel() for p in proj.parameters())
                for proj in variants['edge_proj'][i]:
                    proj_params[i] += sum(p.numel() for p in proj.parameters())
        
        return {
            "enabled": True,
            "num_algorithms": registry.num_algorithms,
            "registered_algorithms": dict(registry.algorithm_dict),
            "num_embedding_modules": len(registry.embedding_variants),
            "num_bottleneck_modules": len(registry.bottleneck_variants),
            "embedding_params_per_algorithm": emb_params,
            "projection_params_per_algorithm": proj_params,
            "total_model_params": sum(p.numel() for p in self.parameters()),
        }

    cls.__init__ = new_init
    cls.forward = new_forward
    cls.get_multitask_info = get_multitask_info
    return cls

"""
Restart manager for interrupted training jobs.

Handles:
- Time-limited training sessions (e.g., 25 min on test nodes)
- Automatic checkpointing and resume
- Temporary storage for models during training
- State tracking across restarts
"""
class TrainingSession:
    """Manages training state for restartable sessions with multitask support."""
    
    def __init__(
        self,
        config_path: str,
        seed: int,
        max_runtime_seconds: int = 1500,  # 25 minutes
        temp_dir: Optional[str] = None,
        multitask: bool = False,
        algorithms: Optional[list] = None
    ):
        self.config_path = config_path
        self.seed = seed
        self.max_runtime_seconds = max_runtime_seconds
        self.multitask = multitask
        self.algorithms = algorithms or []
        self.start_time = time.time()
        
        # Setup temp directory
        if temp_dir is None:
            import os
            user = os.environ.get('USER', 'default')
            temp_dir = f"/tmp/{user}/experiments"
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # State file
        self.state_file = self.temp_dir / f"training_state_{self._get_session_id()}.json"
        
        # Load or initialize state
        self.state = self._load_state()
    
    def _get_session_id(self) -> str:
        """Generate unique session ID from config and seed."""
        config_name = Path(self.config_path).stem
        if self.multitask:
            algo_str = "_".join(sorted(self.algorithms))
            return f"{config_name}_multitask_{algo_str}_seed{self.seed}"
        return f"{config_name}_seed{self.seed}"
    
    def _load_state(self) -> Dict[str, Any]:
        """Load existing state or create new one."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            print(f"Resuming training from step {state.get('current_step', 0)}")
            
            # For multitask, restore algorithm order
            if self.multitask and 'algorithms' in state:
                self.algorithms = state['algorithms']
            
            return state
        else:
            return {
                "config_path": self.config_path,
                "seed": self.seed,
                "multitask": self.multitask,
                "algorithms": self.algorithms if self.multitask else [],
                "current_step": 0,
                "steps_per_algorithm": {alg: 0 for alg in self.algorithms} if self.multitask else {},
                "completed": False,
                "restarts": 0,
                "total_runtime": 0.0,
                "last_checkpoint": None,
            }
    
    def save_state(self, current_step: int, checkpoint_path: Optional[str] = None,
                   steps_per_algorithm: Optional[Dict[str, int]] = None):
        """Save current training state."""
        self.state["current_step"] = current_step
        self.state["restarts"] = self.state.get("restarts", 0)
        self.state["total_runtime"] = self.state.get("total_runtime", 0.0) + self.elapsed_time()
        
        if checkpoint_path:
            self.state["last_checkpoint"] = checkpoint_path
        
        if steps_per_algorithm:
            self.state["steps_per_algorithm"] = steps_per_algorithm
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_completed(self, final_model_path: Optional[str] = None):
        """Mark training as completed."""
        self.state["completed"] = True
        self.state["total_runtime"] = self.state.get("total_runtime", 0.0) + self.elapsed_time()
        if final_model_path:
            self.state["final_model"] = final_model_path
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        print(f"Training completed after {self.state['restarts']} restarts")
        print(f"Total runtime: {self.state['total_runtime']:.1f}s")
    
    def elapsed_time(self) -> float:
        """Get elapsed time since session start."""
        return time.time() - self.start_time
    
    def should_stop(self) -> bool:
        """Check if we should stop training to avoid timeout."""
        # Leave 30 seconds buffer for checkpoint saving
        return self.elapsed_time() >= (self.max_runtime_seconds - 30)
    
    def is_completed(self) -> bool:
        """Check if training is completed."""
        return self.state.get("completed", False)
    
    def get_resume_step(self) -> int:
        """Get step to resume from."""
        return self.state.get("current_step", 0)
    
    def get_steps_per_algorithm(self) -> Dict[str, int]:
        """Get per-algorithm step counts (for multitask)."""
        return self.state.get("steps_per_algorithm", {})
    
    def get_last_checkpoint(self) -> Optional[str]:
        """Get path to last checkpoint."""
        return self.state.get("last_checkpoint")
    
    def increment_restart_count(self):
        """Increment restart counter and record timestamp."""
        self.state["restarts"] = self.state.get("restarts", 0) + 1
        self.state["last_restart_time"] = time.time()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
    def cleanup(self):
        """Clean up temporary files after successful completion."""
        if self.state_file.exists():
            self.state_file.unlink()


class RestartManager:
    """Manages discovery and restart of incomplete training jobs."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        if temp_dir is None:
            user = os.environ.get('USER', 'default')
            temp_dir = f"/tmp/{user}/experiments"
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def find_incomplete_jobs(self) -> list[Dict[str, Any]]:
        """Find all incomplete training jobs."""
        incomplete = []
        
        for state_file in self.temp_dir.glob("training_state_*.json"):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            if not state.get("completed", False):
                incomplete.append(state)
        
        return incomplete
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get next incomplete job to restart."""
        jobs = self.find_incomplete_jobs()
        if not jobs:
            return None
        
        # Sort by current step (resume furthest progress first)
        jobs.sort(key=lambda x: x.get("current_step", 0), reverse=True)
        
        # Filter out jobs that are currently being processed
        # (indicated by recent restart timestamp)
        current_time = time.time()
        available_jobs = []
        
        for job in jobs:
            # If job was restarted in the last 60 seconds, skip it
            # (another process is likely working on it)
            last_restart_time = job.get('last_restart_time', 0)
            if current_time - last_restart_time > 60:
                available_jobs.append(job)
        
        return available_jobs[0] if available_jobs else None
    
    def list_jobs(self):
        """Print all incomplete jobs."""
        jobs = self.find_incomplete_jobs()
        
        if not jobs:
            print("No incomplete training jobs found.")
            return
        
        print(f"\nFound {len(jobs)} incomplete training job(s):")
        print("-" * 80)
        for i, job in enumerate(jobs, 1):
            config = Path(job['config_path']).stem
            print(f"{i}. {config} (seed {job['seed']})")
            print(f"   Step: {job['current_step']} | Restarts: {job.get('restarts', 0)} | "
                  f"Runtime: {job.get('total_runtime', 0.0):.1f}s")
            if job.get('last_checkpoint'):
                print(f"   Checkpoint: {job['last_checkpoint']}")
        print("-" * 80)
    
    def clean_all_jobs(self) -> int:
        """Clean up all incomplete training job state files. Returns count of cleaned jobs."""
        state_files = list(self.temp_dir.glob("training_state_*.json"))
        
        if not state_files:
            return 0
        
        print(f"\nCleaning {len(state_files)} job state file(s)...")
        for state_file in state_files:
            print(f"  Removing: {state_file.name}")
            state_file.unlink()
        
        return len(state_files)
    
    def clean_job(self, session_id: str) -> bool:
        """Clean up a specific job by session ID."""
        state_file = self.temp_dir / f"training_state_{session_id}.json"
        
        if not state_file.exists():
            return False
        
        print(f"Removing state file: {state_file}")
        state_file.unlink()
        return True
    
    def _get_session_id_from_job(self, job: Dict[str, Any]) -> str:
        """Extract session ID from job state."""
        config_name = Path(job['config_path']).stem
        seed = job['seed']
        
        if job.get('multitask', False):
            algorithms = job.get('algorithms', [])
            algo_str = "_".join(sorted(algorithms))
            return f"{config_name}_multitask_{algo_str}_seed{seed}"
        
        return f"{config_name}_seed{seed}"


def get_temp_model_dir(models_directory: str, model_name: str, temp_dir: Optional[str] = None) -> Path:
    """
    Get temporary directory for model saving during training.
    
    Args:
        models_directory: Original models directory
        model_name: Name of the model
        temp_dir: Optional temp directory (default: /tmp/$USER/experiments)
    
    Returns:
        Path to temporary model directory
    """
    if temp_dir is None:
        user = os.environ.get('USER', 'default')
        temp_dir = f"/tmp/{user}/experiments"
    
    temp_path = Path(temp_dir) / "models" / model_name
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


def finalize_model(temp_model_path: str, final_model_path: str):
    """
    Move model from temp to final location after training completes.
    
    Args:
        temp_model_path: Temporary model path
        final_model_path: Final destination path
    """
    temp_path = Path(temp_model_path)
    final_path = Path(final_model_path)
    
    if not temp_path.exists():
        print(f"Warning: Temp model not found at {temp_path}")
        return
    
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temp_path, final_path)
    print(f"Model finalized: {final_path}")
    
    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

def save_multitask_checkpoint(model, checkpoint_path: str, step: int, 
                               algorithm_dict: Optional[Dict[str, int]] = None):
    """
    Save a complete checkpoint including multitask state.
    
    Args:
        model: The model to save
        checkpoint_path: Path to save checkpoint
        step: Current training step
        algorithm_dict: Optional dict mapping algorithm names to indices
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
    }
    
    # Add multitask registry state if enabled
    if hasattr(model, '_multitask_registry'):
        registry = model._multitask_registry
        checkpoint['multitask_state'] = {
            'num_algorithms': registry.num_algorithms,
            'algorithm_dict': registry.algorithm_dict.copy(),
        }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def load_multitask_checkpoint(model, checkpoint_path: str) -> int:
    """
    Load a checkpoint including multitask state.
    
    Args:
        model: The model to load into
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Training step to resume from
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore multitask registry if present
    if 'multitask_state' in checkpoint and hasattr(model, '_multitask_registry'):
        registry = model._multitask_registry
        multitask_state = checkpoint['multitask_state']
        
        # Verify compatibility
        if registry.num_algorithms != multitask_state['num_algorithms']:
            raise ValueError(
                f"Checkpoint has {multitask_state['num_algorithms']} algorithms, "
                f"but model has {registry.num_algorithms}"
            )
        
        # Restore algorithm dictionary
        registry.algorithm_dict = multitask_state['algorithm_dict'].copy()
        
        print(f"Restored multitask state with algorithms: {list(registry.algorithm_dict.keys())}")
    
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded from step {step}")
    return step

# gpu stuff
def get_least_used_gpus() -> list[int]:
    """
    Get list of GPU IDs sorted by usage (least used first).
    Falls back to all GPUs if nvidia-smi is not available.
    """
    try:
        import subprocess
        # Get GPU memory usage using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print("Warning: nvidia-smi failed, using all GPUs")
            return list(range(torch.cuda.device_count()))
        
        # Parse output: index, memory_used_mb, gpu_util_percent
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                try:
                    gpu_id = int(parts[0])
                    mem_used = int(parts[1])
                    
                    # Handle [N/A] or [Not Supported] for gpu_util
                    gpu_util_str = parts[2] if len(parts) > 2 else '0'
                    if '[N/A]' in gpu_util_str or '[Not Supported]' in gpu_util_str or gpu_util_str == '':
                        gpu_util = 0  # Assume idle if not available
                    else:
                        gpu_util = int(gpu_util_str)
                    
                    # Combined score: prioritize low memory usage, then low GPU util
                    score = mem_used * 0.7 + gpu_util * 0.3
                    gpu_stats.append((gpu_id, score, mem_used, gpu_util))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse GPU line '{line}': {e}")
                    continue
        
        if not gpu_stats:
            print("Warning: No valid GPU stats parsed, using all GPUs")
            return list(range(torch.cuda.device_count()))
        
        # Sort by score (lowest first)
        gpu_stats.sort(key=lambda x: x[1])
        sorted_gpus = [gpu_id for gpu_id, _, _, _ in gpu_stats]
        
        print(f"GPU usage (sorted by availability):")
        for gpu_id, score, mem_used, gpu_util in gpu_stats:
            util_str = f"{gpu_util}%" if gpu_util > 0 else "N/A"
            print(f"  GPU {gpu_id}: {mem_used}MB used, {util_str} util, score {score:.1f}")
        
        return sorted_gpus
    
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Warning: Could not query GPU stats ({e}), using all GPUs")
        return list(range(torch.cuda.device_count()))

def get_gpus(int: int) -> list[int]:
    """
    Get a list of GPU IDs to use, selecting the least used ones.
    
    Args:
        int: Number of GPUs to select.
    Returns:
        List of GPU IDs.
    """
    available_gpus = get_least_used_gpus()
    selected_gpus = available_gpus[:int]
    print(f"Selected GPUs: {selected_gpus}")
    return selected_gpus
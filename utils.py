import argparse
import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, overload
import numpy as np
import torch
from torch_geometric.utils import group_argsort, scatter, softmax
import time
import os
import shutil
from generate_data import MASK
from configs import base_config


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
    #get the true node count based on edge index, padding nodes have no edges
    is_predicted_pointer = 1.0 * (group_argsort(prediction, edge_index[0], descending=True, stable=True) == 0)
    # Robust N calculation for padded batches
    valid_nodes_count = is_predicted_pointer.sum()
    if valid_nodes_count == 0: valid_nodes_count = graph.num_nodes
    return (graph.y * is_predicted_pointer).sum() / valid_nodes_count


def pointer_accuracy_graph_level(graph, prediction):
    return 1.0 * (pointer_accuracy(graph, prediction) == 1.0)


def node_mask_accuracy(graph, prediction):
    pred_mask = prediction > 0.0
    return (1.0 * (pred_mask == graph.y)).mean()


def node_mask_accuracy_graph_level(graph, prediction):
    return 1.0 * (node_mask_accuracy(graph, prediction) == 1.0)

def score(data, batched_prediction, calculators, output_type) -> Dict[str, float]:
    """
    Scores a batch of graphs using provided calculators.
    Args:
        data: PyG batch with multiple graphs
        batched_prediction: Model predictions for the batch
        calculators: Tuple of metric functions
        output_type: 'pointer' or 'node_mask'
    Returns:
        Dictionary of metric scores
    """
    scores = defaultdict(float)
    for batch_idx, graph in enumerate(data.to_data_list()):
        batch_pred_idx = (
            data.batch[data.edge_index[0]]
            if output_type == "pointer"
            else data.batch
        )
        prediction = batched_prediction[batch_pred_idx == batch_idx]

        for calculator in calculators:
            value = calculator(graph, prediction)
            scores[calculator.__name__] += (
                value if isinstance(value, float) else value.item()
            )
    return scores

def evaluate(model, dataloader, calculators) -> tuple[Dict[str, float], float]:    
    """
    Evaluate the model on a dataset.

    Args:
        model: The trained model to evaluate
        dataloader: DataLoader with evaluation data
        calculators: Tuple of metric functions to compute
    Returns:
        Tuple of (scores dict, average loss)
    """
    scores = defaultdict(float)
    total_points = 0
    total_loss = 0.0
    device = model.parameters().__next__().device

    for data in dataloader:
        data = data.to(device)
        batched_prediction, batched_loss = model(data, training_step=-1)
        data_list = data.to_data_list()
        total_loss += batched_loss.detach().item() * len(data_list)
        total_points += len(data_list)
        # print(len(data_list))
        score_batch = score(data, batched_prediction, calculators, model.output_type)
        for calculator in calculators:
            scores[calculator.__name__] += score_batch[calculator.__name__]
        
    for calculator in calculators:
        scores[calculator.__name__] /= total_points


    return scores, total_loss / len(dataloader)


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
    total_loss = 0.0
    device = model.parameters().__next__().device
    for data in dataloader:
        data = data.to(device)
        # Pass the algorithm to use correct encoder/decoder components
        batched_prediction, batched_loss = model(data, multitask_algorithm=algorithm, training_step=-1)
        
        data_list = data.to_data_list()
        total_loss += batched_loss.detach().item()
        total_points += len(data_list)
        score_batch = score(data, batched_prediction, calculators, model.output_type)

        for calculator in calculators:
            scores[calculator.__name__] += score_batch[calculator.__name__]

    for calculator in calculators:
        scores[calculator.__name__] /= total_points

    return scores, total_loss / len(dataloader)

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
            batched_prediction, loss = model(data, multitask_algorithm=algorithm)
        else:
            batched_prediction, loss = model(data)
        
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
                #use the self-loop edges to infer num nodes because we may pad the nodes

                if graph.edge_index.numel() > 0:
                    num_nodes = int(graph.edge_index[0].max().item()) + 1
                else:
                    num_nodes = graph.num_nodes # Fallback
               
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
                num_nodes = graph.y.shape[0]
                # For node_mask output type
                if prediction.shape[0] > num_nodes: prediction = prediction[:num_nodes] #if we have padded nodes, cut them off
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
                "loss": loss.detach().item(),
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
            "average_loss": sum(graph["loss"] for graph in graph_results.values()) / len(dataloader) if total_points > 0 else 0.0,
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
        self.algorithm_dict: Dict[str, int] = {}
        self.embedding_variants: Dict[tuple, torch.nn.ModuleList] = {}
        self.bottleneck_variants: Dict[int, dict] = {}
        # New set to track if specs are padded for an index
        self.specs_initialized = set()

    def register_embedding(self, parent, attr_name, original_emb):
        key = (id(parent), attr_name)
        if key in self.embedding_variants: return self.embedding_variants[key]
        variants = torch.nn.ModuleList([original_emb])
        for _ in range(self.num_algorithms - 1): variants.append(_clone_embedding(original_emb))
        self.embedding_variants[key] = variants
        return variants

    def register_bottleneck(self, bottleneck):
        key = id(bottleneck)
        if key in self.bottleneck_variants: return self.bottleneck_variants[key]
        node_proj = torch.nn.ModuleList([bottleneck.node_projections])
        edge_proj = torch.nn.ModuleList([bottleneck.edge_projections])
        for _ in range(self.num_algorithms - 1):
            node_proj.append(_clone_projection_list(bottleneck.node_projections))
            edge_proj.append(_clone_projection_list(bottleneck.edge_projections))
        variants = {'node_proj': node_proj, 'edge_proj': edge_proj, 'specs': [bottleneck.spec] * self.num_algorithms}
        self.bottleneck_variants[key] = variants
        return variants

    def get_or_register_algorithm(self, algorithm_name: str) -> int:
        # 1. Resolve Index
        if algorithm_name not in self.algorithm_dict:
            idx = len(self.algorithm_dict)
            if idx >= self.num_algorithms:
                raise ValueError(f"Max algorithms ({self.num_algorithms}) reached.")
            self.algorithm_dict[algorithm_name] = idx
        else:
            idx = self.algorithm_dict[algorithm_name]
        
        # 2. Update specs if not done for this index (e.g. after restart)
        # This fixes the IndexError because restored dicts didn't trigger padding logic
        if idx not in self.specs_initialized:
            if algorithm_name in SPEC:
                original_spec = SPEC[algorithm_name]
                for key, variants in self.bottleneck_variants.items():
                    # Padding logic
                    num_node_proj = len(variants['node_proj'][0])
                    num_edge_proj = len(variants['edge_proj'][0])
                    current_node_spec = original_spec[0]
                    current_edge_spec = original_spec[1]
                    
                    if len(current_node_spec) < num_node_proj:
                        current_node_spec += (MASK,) * (num_node_proj - len(current_node_spec))
                    if len(current_edge_spec) < num_edge_proj:
                        current_edge_spec += (MASK,) * (num_edge_proj - len(current_edge_spec))
                    
                    variants['specs'][idx] = (current_node_spec, current_edge_spec)
            self.specs_initialized.add(idx)
        return idx


class MultitaskContext:
    """
    Context manager that swaps in algorithm-specific components during forward pass.
    """
    def __init__(self, model, algorithm_name):
        self.model = model
        self.algorithm_name = algorithm_name
        self.registry = getattr(model, '_multitask_registry', None)
        self.swapped = []

    def __enter__(self):
        if self.registry is None or self.algorithm_name is None: return self
        alg_idx = self.registry.get_or_register_algorithm(self.algorithm_name)
        
        # Swap Embeddings
        for (pid, attr), variants in self.registry.embedding_variants.items():
            for parent, name, _, _ in _find_embedding_modules(self.model):
                if id(parent) == pid and name == attr:
                    orig = getattr(parent, attr)
                    setattr(parent, attr, variants[alg_idx])
                    self.swapped.append((parent, attr, orig))
                    break
        
        # Swap Bottlenecks
        for bid, variants in self.registry.bottleneck_variants.items():
            for btn, _ in _find_states_bottleneck(self.model):
                if id(btn) == bid:
                    orig = (btn.node_projections, btn.edge_projections, btn.spec)
                    btn.node_projections = variants['node_proj'][alg_idx]
                    btn.edge_projections = variants['edge_proj'][alg_idx]
                    btn.spec = variants['specs'][alg_idx]
                    self.swapped.append((btn, "bottleneck", orig))
                    break
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, attr, orig in self.swapped:
            if attr == "bottleneck":
                obj.node_projections, obj.edge_projections, obj.spec = orig
            else:
                setattr(obj, attr, orig)
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

    def new_init(self, *args, multitask_num_algorithms=None, **kwargs):
        original_init(self, *args, **kwargs)
        if multitask_num_algorithms and multitask_num_algorithms > 1:
            reg = MultitaskRegistry(multitask_num_algorithms)
            for p, a, e, path in _find_embedding_modules(self):
                vars = reg.register_embedding(p, a, e)
                safe_name = path.replace(".","_")
                setattr(self, f'_multitask_emb_{safe_name}', vars)
            for b, path in _find_states_bottleneck(self):
                vars = reg.register_bottleneck(b)
                safe_name = path.replace(".","_")
                setattr(self, f'_multitask_node_proj_{safe_name}', vars['node_proj'])
                setattr(self, f'_multitask_edge_proj_{safe_name}', vars['edge_proj'])
            self._multitask_registry = reg
            self._multitask_enabled = True

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
        algorithms: Optional[list] = None,
        patience: int = 5,
    ):
        self.config_path = config_path
        self.seed = seed
        self.max_runtime_seconds = max_runtime_seconds
        self.multitask = multitask
        self.algorithms = algorithms or []
        self.start_time = time.time()
        self.patience = patience
        
        # Setup temp directory
        if temp_dir is None:
            import os
            user = os.environ.get('USER', 'default')
            temp_dir = f"tmp" #rn here
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
            
            if self.multitask and 'algorithms' in state:
                self.algorithms = state['algorithms']
            
            return state
        else:
            # FRESH START: Generate a unique ID with timestamp
            # This ensures we don't merge with old deleted runs on WandB
            unique_suffix = int(time.time())
            wandb_id = f"{self._get_session_id()}_{unique_suffix}"
            
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
                "wandb_run_id": wandb_id,  # <--- Persist the unique ID
                "best_val_score": -1.0,
                "patience_counter": 0,
            }

    def check_early_stopping(self, current_val_score: float) -> bool:
        """
        Updates patience state. Returns True if training should stop.
        Currently set to only stop early on repeated perfect validation score.
        """
        best_score = self.state.get("best_val_score", -1.0)
        
        # Check for improvement (assuming higher is better, like accuracy)
        # Using a small epsilon 1e-4 to prevent stopping on floating point jitter
        if current_val_score > (best_score + 1e-4):
            self.state["best_val_score"] = current_val_score
            self.state["patience_counter"] = 0
            print(f"  > New best val score: {current_val_score:.4f}")
        else:
            self.state["patience_counter"] = self.state.get("patience_counter", 0) + 1 if current_val_score==1.0 else 0 #reset if not perfect, as we want to continue training if not perfect until we run out of time
            print(f"  > No improvement. Patience: {self.state['patience_counter']}/{self.patience} (Best: {best_score:.4f})")
        
        # Save state immediately to persist counter across restarts
        with open(self.state_file, 'w') as f: 
            json.dump(self.state, f, indent=2)
            
        return self.state["patience_counter"] >= self.patience

    def get_wandb_run_id(self) -> str:
        """Get the persistent WandB run ID for this session."""
        # Fallback if key missing in old state files
        if "wandb_run_id" not in self.state:
            self.state["wandb_run_id"] = f"{self._get_session_id()}_{int(time.time())}"
        return self.state["wandb_run_id"]
    
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
            config = base_config.Config(self.config_path)
            #put it into the output directory instead of deleting
            output_dir = Path(config.out_directory)
            archived_state_file = output_dir / f"archived/{self.state_file.name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Archiving state file to: {archived_state_file}")
            #wait for the dir to exist:
            archived_state_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.state_file), str(archived_state_file))

    @staticmethod
    @overload
    def resurrect(config_path:str, seed:int, temp_dir:str=None) -> 'TrainingSession': ...
    @staticmethod
    @overload
    def resurrect(archived_path:str|Path) -> 'TrainingSession': ...
    @staticmethod
    def resurrect(*args, **kwargs) -> 'TrainingSession':
        """Resurrect a TrainingSession from existing state, useful for when it stopped but shouldn't have."""
        
        #get args
        archived_path = None
        if len(args) == 2:
            config_path = args[0]
            seed = args[1]
            temp_dir = kwargs.get('temp_dir', None)
        elif len(args) == 1:
            archived_path = Path(args[0])

        if archived_path is not None:
            #load state from archived path
            with open(archived_path, 'r') as f:
                state = json.load(f)
            config_path = state['config_path']
            seed = state['seed']
            temp_dir = kwargs.get('temp_dir', None)
            session = TrainingSession(config_path, seed, temp_dir=temp_dir)
            session.state = state
            #set the completed flag to false
            session.state['completed'] = False
            print(f"Resurrected training session from archived state: {archived_path}.\nReady to resume from step {session.get_resume_step()}.")
            return session
        
        else:#we need to resurrect from config and seed
            # 1. Setup paths
            if temp_dir is None:
                user = os.environ.get('USER', 'default')
                # temp_dir = Path(f"/tmp/{user}/experiments")
                temp_dir = Path(f"tmp")
            else:
                temp_dir = Path(temp_dir)
            
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. Load Config to find model name and algorithms
            print(f"Loading config: {config_path}")
            base_cfg = base_config.read_config(config_path)
            
            algorithms = []
            is_multitask = False
            
            if hasattr(base_cfg, 'multitask_algorithms') and base_cfg.multitask_algorithms:
                is_multitask = True
                algorithms_session = sorted(base_cfg.multitask_algorithms) # Sort for consistent naming
                algorithms = base_cfg.multitask_algorithms# use the original order
                algo_str_session = "_".join(algorithms_session)
                algo_str_checkpoint = "_".join(algorithms)
                model_name = f"multitask_{algo_str_checkpoint}_{seed}"
                config_name = Path(config_path).stem
                session_id = f"{config_name}_multitask_{algo_str_session}_seed{seed}"
            else:
                model_name = f"{base_cfg.algorithm}_{seed}"
                config_name = Path(config_path).stem
                session_id = f"{config_name}_{base_cfg.algorithm}_seed{seed}"
            print(f"Target Session ID: {session_id}")
            
            # 3. Find Checkpoints
            checkpoint_dir = Path(base_cfg.out_directory).parent / "out" / f"checkpoints_{model_name}"
            # Note: Adjust 'out' if your config uses a different base output directory
            if not checkpoint_dir.exists():
                # Try local models dir as fallback
                checkpoint_dir = Path(base_cfg.out_directory) / f"checkpoints_{model_name}"
            
            if not checkpoint_dir.exists():
                print(f"Error: Could not find checkpoint directory at: {checkpoint_dir}")
                print("Please check where your checkpoints are saved (config.out_directory).")
                exit(1)

            # Find latest checkpoint (.pt file)
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if not checkpoints:
                print("Error: No .pt files found in checkpoint directory.")
                exit(1)

            # Helper to get step from filename or load file
            def get_step(p):
                if "final" in p.name:
                    # Load file to get exact step
                    try:
                        state = torch.load(p, map_location='cpu')
                        return state.get('step', 999999999)
                    except:
                        return 0
                # Parse "model_step_12345.pt"
                parts = p.stem.split('_')
                for part in parts:
                    if part.isdigit():
                        return int(part)
                return 0

            latest_ckpt = max(checkpoints, key=get_step)
            current_step = get_step(latest_ckpt)

            if 'final' in latest_ckpt.name:
                #get the step counter from the training_summary.json if available
                summary_file = checkpoint_dir / f"training_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    if is_multitask:
                        steps_per_alg = summary.get('steps_per_algorithm', {})
                        current_step = sum(steps_per_alg.get(alg, 0) for alg in algorithms)
                    else:
                        current_step = summary.get('total_steps', current_step)
            
            print(f"Found latest checkpoint: {latest_ckpt.name} (Step {current_step})")

            # 4. Reconstruct State
            # For multitask, we need to estimate steps_per_algorithm
            steps_per_alg = {}
            if is_multitask:
                # Assuming synchronized training, each alg is roughly at total_step / num_algs
                val = current_step // len(algorithms)
                steps_per_alg = {alg: val for alg in algorithms}
            
            state = {
                "config_path": config_path,
                "seed": seed,
                "multitask": is_multitask,
                "algorithms": algorithms,
                "current_step": current_step,
                "steps_per_algorithm": steps_per_alg,
                "completed": False,  # <--- CRITICAL: Resurrect job
                "restarts": 1e4,  # High number to indicate resurrected job
                "total_runtime": 0.0,
                "last_checkpoint": str(latest_ckpt.absolute()),
                "wandb_run_id": model_name, # Try to reconnect to legacy/named run
                "best_val_score": -1.0, # Reset so it doesn't stop immediately if score dropped slightly
                "patience_counter": 0   # <--- CRITICAL: Reset patience
            }

            # 5. Save State File
            state_file = temp_dir / f"training_state_{session_id}.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"Successfully reconstructed state file: {state_file}")
            print(f"You can now run: python train.py --restart --patience 100")




class RestartManager:
    """Manages discovery and restart of incomplete training jobs."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        if temp_dir is None:
            user = os.environ.get('USER', 'default')
            temp_dir = f"tmp/"
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def find_incomplete_jobs(self) -> list[Dict[str, Any]]:
        """Find all incomplete training jobs."""
        incomplete = []
        # print(list(self.temp_dir.glob("training_state_*.json"))) #debug
        # print(self.temp_dir) #debug
        # print(os.listdir(self.temp_dir)) #debug
        for state_file in self.temp_dir.glob("training_state_*.json"):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            if not state.get("completed", False):
                incomplete.append(state)
        
        return incomplete
    
    def get_next_job(self, filter:str=None, force:bool=False, seed:int=None) -> Optional[Dict[str, Any]]:
        """Get next incomplete job to restart."""
        jobs = self.find_incomplete_jobs()
        if not jobs:
            return None
        # print(jobs) #debug
        if filter:#filer is the config path
            for job in jobs:
                print(f"'{job['config_path']}'", type(job['config_path']))
            jobs = [job for job in jobs if filter in job['config_path']] 
        
        # print("after filter:", jobs) #debug
        # Sort by current step (resume furthest progress first)
        jobs.sort(key=lambda x: x.get("current_step", 0), reverse=True)
        
        # Filter out jobs that are currently being processed
        # (indicated by recent restart timestamp)
        current_time = time.time()
        available_jobs = []
        
        for job in jobs:
            if seed is not None and job['seed'] != seed:
                print(f"Skipping job '{job['config_path']}' (seed {job['seed']} does not match filter {seed})")
                continue
            # If job was restarted in the last 60 seconds, skip it
            # (another process is likely working on it)
            last_restart_time = job.get('last_restart_time', 0)
            if force or current_time - last_restart_time > 1200: #don't choose probably running jobs
                available_jobs.append(job)
            else:
                print(f"Skipping job '{job['config_path']}' (recently restarted)")
        
        print("available jobs:", available_jobs) #debug
        return available_jobs[0] if available_jobs else None

    def get_next_jobs(self, num_jobs: int, filter:str=None, force:bool=False) -> list[Dict[str, Any]]:
        """Get up to N incomplete jobs to restart."""
        jobs = self.find_incomplete_jobs()
        if not jobs:
            return []

        if filter:#filer is the config path
            jobs = [job for job in jobs if filter in job['config_path']]
        
        # Sort by current step (resume furthest progress first)
        jobs.sort(key=lambda x: x.get("current_step", 0), reverse=True)
        
        # Filter out jobs that are currently being processed
        current_time = time.time()
        available_jobs = []
        
        for job in jobs:
            last_restart_time = job.get('last_restart_time', 0)
            if force or current_time - last_restart_time > 1800: #don't choose probably running jobs
                available_jobs.append(job)
                if len(available_jobs) >= num_jobs:
                    break
        
        return available_jobs
    
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
    
    def clean_all_jobs(self, except_substring: Optional[str] = None) -> int:
        """Clean up all incomplete training job state files. Returns count of cleaned jobs."""
        state_files = list(self.temp_dir.glob("training_state_*.json"))
        
        if not state_files:
            return 0
        
        print(f"\nCleaning {len(state_files)} job state file(s)...")
        for state_file in state_files:
            if except_substring and except_substring in state_file.name:
                print(f"  Skipping: {state_file.name}, it contains '{except_substring}'")
                continue
            print(f"  Removing: {state_file.name}")
            state_file.unlink()
        
        return len(state_files)
    
    def clean_job(self, session_id: str) -> bool:
        """Clean up a specific job by session ID."""
        state_file = self.temp_dir / f"training_state_{session_id}.json"
        
        if not state_file.exists():
            print(f"State file not found: {state_file}")
            print("Available state files:")
            self.list_jobs()
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

    def archive_job(self, session_id: str) -> bool:
        """Archive a specific job by session ID."""
        state_file = self.temp_dir / f"training_state_{session_id}.json"
        
        if not state_file.exists():
            print(f"State file not found: {state_file}")
            print("Available state files:")
            self.list_jobs()
            return False
        
        archived_dir = state_file.parent / "archived"
        archived_dir.mkdir(parents=True, exist_ok=True)
        archived_path = archived_dir / state_file.name
        
        print(f"Archiving state file to: {archived_path}")
        shutil.move(str(state_file), str(archived_path))
        return True

    def unarchive_job(self, session_id: str) -> bool:
        """Unarchive a specific job by session ID."""
        archived_path = self.temp_dir / "archived" / f"training_state_{session_id}.json"
        
        if not archived_path.exists():
            print(f"Archived state file not found: {archived_path}")
            return False
        
        restored_path = self.temp_dir / f"training_state_{session_id}.json"
        
        print(f"Restoring state file to: {restored_path}")
        shutil.move(str(archived_path), str(restored_path))
        return True

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
        # temp_dir = f"/tmp/{user}/experiments"
        temp_dir = f"tmp"
    
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
def get_least_used_gpus(mig:int=0) -> list[int|str]:
    """
    Get list of GPU IDs sorted by usage (least used first).
    Falls back to all GPUs if nvidia-smi is not available.
    """
    preferences= [1, 1e5, 1, 1]

    if not mig:
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
            gpu_stats = [(gpu_id, score*preferences[gpu_id], mem_used, gpu_util) for gpu_id, score, mem_used, gpu_util in gpu_stats]
            gpu_stats.sort(key=lambda x: x[1])
            sorted_gpus = [gpu_id for gpu_id, score, _, _ in gpu_stats if score < 1e5]  # Filter out very high usage GPUs
            
            print(f"GPU usage (sorted by availability):")
            for gpu_id, score, mem_used, gpu_util in gpu_stats:
                util_str = f"{gpu_util}%" if gpu_util > 0 else "N/A"
                print(f"  GPU {gpu_id}: {mem_used}MB used, {util_str} util, score {score:.1f}")
            
            return sorted_gpus
        
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"Warning: Could not query GPU stats ({e}), using all GPUs")
            return list(range(torch.cuda.device_count()))

    else:
        #if mig start at the mig value and go around to list all 7 devices by int
        total_gpus = 7
        mig_gpus = []
        for i in range(total_gpus):
            mig_id = (mig + i) % total_gpus
            mig_gpus.append(mig_id)
        return mig_gpus

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart Manager Utility")
    parser.add_argument('--list', action='store_true', help="List incomplete training jobs")
    parser.add_argument('--clean-all', action='store_true', help="Clean all incomplete training job state files")
    parser.add_argument('--clean', type=str, help="Clean specific job by session ID")
    parser.add_argument('--filter', type=str, help="Filter jobs by config path substring")
    parser.add_argument('--num-jobs', type=int, default=1, help="Number of jobs to list")
    parser.add_argument('--resurrect', action='store_true', help="Resurrect a training session from config and seed")
    parser.add_argument('--config_path', type=str, help="Path to config file for resurrection")
    parser.add_argument('--seed', type=int, help="Random seed for resurrection")
    args = parser.parse_args()

    manager = RestartManager()

    if args.list:
        manager.list_jobs()
    elif args.clean_all:
        count = manager.clean_all_jobs()
        print(f"Cleaned {count} job(s).")
    elif args.clean:
        success = manager.clean_job(args.clean)
        if success:
            print(f"Cleaned job with session ID: {args.clean}")
        else:
            print(f"No job found with session ID: {args.clean}")
    elif args.resurrect:
        if not args.config_path or args.seed is None:
            print("Error: --config-path and --seed are required for resurrection.")
            exit(1)
        TrainingSession.resurrect(args.config_path, args.seed)

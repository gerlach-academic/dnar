import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import group_argsort, scatter, softmax


def reverse_edge_index(edge_index):
    rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
    rev_index = torch.argsort(rev_edge_index, stable=True)[0]
    assert torch.all(edge_index[:, rev_index] == rev_edge_index)
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
    for data in dataloader:
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

#evaluates but also prints the predicted pointer structure and the true output structure for reference 
def evaluate_print(model, dataloader, calculators, output_path="evaluation_results.json"):
    """
    Evaluates the model and saves detailed results including predicted and true pointers.
    
    Args:
        model: The trained model to evaluate
        dataloader: DataLoader with evaluation data
        calculators: Tuple of metric functions to compute
        output_path: Path to save the results JSON file
    
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
    
    for data in dataloader:
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
    
    print(f"Evaluation results saved to: {output_path}")
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
        path = "{}_{}".format(self.model_name, suffix)
        print("saving model: ", path)
        torch.save(model.state_dict(), path)


NODE_POINTER_METRICS = (pointer_accuracy, pointer_accuracy_graph_level)
NODE_MASK_METRICS = (node_mask_accuracy, node_mask_accuracy_graph_level)
METRICS = {"pointer": NODE_POINTER_METRICS, "node_mask": NODE_MASK_METRICS}

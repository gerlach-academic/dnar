from collections import defaultdict
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

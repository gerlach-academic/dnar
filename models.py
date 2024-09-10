from torch.nn import Module

from configs import base_config
from processors import DiscreteProcessor
from utils import reverse_edge_index


class Dnar(Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        self.processor = DiscreteProcessor(config)

        self.stepwise_training = config.stepwise_training
        self.output_type = config.output_type
        self.output_idx = config.output_idx

    def forward(self, batch, writer=None, training_step: int = -1):
        teacher_force = self.stepwise_training and training_step != -1
        T = batch.node_fts.shape[1]
        batch.batched_reverse_idx = reverse_edge_index(batch.edge_index)

        loss = 0.0

        node_states = batch.node_fts[:, 0]
        edge_states = batch.edge_fts[:, 0]
        cur_step_scalars = batch.scalars[:, 0]

        for processor_step in range(1, T):
            node_states, edge_states, cur_step_scalars, cur_step_loss = self.processor(
                node_states,
                edge_states,
                cur_step_scalars,
                batch,
                training_step,
                processor_step,
                teacher_force,
            )
            loss += cur_step_loss / T

        states = edge_states if self.output_type == "pointer" else node_states
        output = states[:, self.output_idx]

        if writer is not None:
            writer.add_scalar("Loss/train", loss.detach().item(), training_step)
        return output, loss

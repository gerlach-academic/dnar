from typing import Optional, overload

from torch.nn import Module

from configs import base_config
from processors import DiscreteProcessor
from utils import reverse_edge_index, multitask


@multitask
class Dnar(Module):
    """
    Discrete Neural Algorithmic Reasoner.
    
    When instantiated with multitask_num_algorithms > 1, this model supports
    algorithm-specific encoders/decoders while sharing the latent processor.
    
    Usage:
        # Single-task mode
        model = Dnar(config)
        output, loss = model(batch)
        
        # Multi-task mode
        model = Dnar(config, multitask_num_algorithms=5)
        output, loss = model(batch, multitask_algorithm="dijkstra")
    """
    def __init__(self, config: base_config.Config):
        super().__init__()
        self.processor = DiscreteProcessor(config)

        self.stepwise_training = config.stepwise_training
        self.output_type = config.output_type
        self.output_idx = config.output_idx

    def forward(self, batch, writer=None, training_step: int = -1, multitask_algorithm: Optional[str] = None):
        """
        Forward pass through the model.
        
        Args:
            batch: PyG batch with node_fts, edge_fts, scalars, edge_index
            writer: Optional TensorBoard writer
            training_step: Current training step (-1 for eval mode)
            multitask_algorithm: Algorithm name for multitask mode (e.g., "dijkstra", "bfs")
                                 Only used when model was created with multitask_num_algorithms > 1.
                                 The @multitask decorator handles all component swapping.
        
        Returns:
            (output, loss) tuple
        """
        # Note: The @multitask decorator has already swapped in the correct
        # algorithm-specific components (embeddings, projections, spec) before
        # this method is called. No need to pass multitask_algorithm further.
        
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

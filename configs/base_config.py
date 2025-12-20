from dataclasses import dataclass
from typing import Optional, List

import yaml


@dataclass
class Config:
    algorithm: str = "bfs" #just the default
    name: Optional[str] = None #can be used to give a name to the experiment other than the algorithm
    graph_type: Optional[str] = None
    use_lazy_dataset: bool = False

    # --- train ---
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    num_iterations: int = 10000
    min_iterations: int = 1000 #do at least this before checking early stopping
    eval_each: int = 100 #eval and for early stopping
    stepwise_training: bool = True #whether to use teacher forcing
    processor_upper_t: float = 3. #gumbel annealing temperature start
    processor_lower_t: float = 0.01 #gumbel annealing temperature end
    use_noise: bool = True #whether to use gumbel noise in the softmaxes

    # --- data ---
    num_samples: dict = None
    problem_size: dict = None
    edge_weights: bool = False
    generate_random_numbers: bool = False
    graph_type: str = "er"

    # --- model ---
    h: int = 128
    temp_on_eval: float = 0.
    checkpoint_interval: float = 0.05 # Fraction of total steps between checkpoints and print evals

    num_node_states: int = 1
    num_edge_states: int = 1

    output_type: str = 'pointer'
    output_idx: int = 0
    
    # --- multitask ---
    # Number of algorithms for multitask learning. Set > 1 to enable.
    # When enabled, the model creates separate encoder/decoder components
    # for each algorithm while sharing the latent processor.
    multitask_num_algorithms: Optional[int] = None
    # List of algorithm names for multitask (optional, for documentation)
    # you may also just call the model.forward with algorithm names directly.
    multitask_algorithms: Optional[List[str]] = None

    # --- data cache ---
    use_dataset_cache: bool = True
    cache_directory: str = '/hpcwork/sg114224/dnar_cache'

    # --- logs, io ---
    models_directory: str = 'models'
    out_directory: str = 'out'
    tensorboard_logs: bool = True
    wandb_logs: bool = True
    project: str = 'dnar_single'
    wandb_entity: Optional[str] = None  # Your wandb username or team name


def read_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)

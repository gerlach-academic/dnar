from dataclasses import dataclass

import yaml


@dataclass
class Config:
    algorithm: str = None

    # --- train ---
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    num_iterations: int = 1000
    eval_each: int = 250
    stepwise_training: bool = True
    processor_upper_t: float = 3.
    processor_lower_t: float = 0.01
    use_noise: bool = True

    # --- data ---
    num_samples: dict = None
    problem_size: dict = None
    edge_weights: bool = False
    generate_random_numbers: bool = False

    # --- model ---
    h: int = 128
    temp_on_eval: float = 0.

    num_node_states: int = 1
    num_edge_states: int = 1

    output_type: str = 'pointer'
    output_idx: int = 0

    # --- logs, io ---
    models_directory: str = 'models'
    tensorboard_logs: bool = True


def read_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)

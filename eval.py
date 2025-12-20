import torch

import utils
from configs import base_config
from generate_data import create_dataloader
from models import Dnar

import argparse

def eval_worker(seed: int, config: base_config.Config, model: str, device: torch.device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for model_path in model_paths:
        print(f"Evaluating model: {model_path} on seed {seed}")
        model = Dnar(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device)

        split = "val"
        sampler = create_dataloader(config, split, seed=seed, device=device)

        with torch.no_grad():
            scores = utils.evaluate(model, sampler, utils.METRICS[config.output_type])
            print(f"Scores for model {model_path} on seed {seed}: {scores}")

if __name__ == "__main__":
    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42**2, help='Random seed for reproducibility')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds to run')
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file to use for eval')
    parser.add_argument('--models', type=str, nargs='+', default=None, help='List of model paths to evaluate. Sth like [algorithm]_[seed]=bfs_42, ...\nIf not provided, will determine the algorithm from the config_path and use the first num_seeds models')

    args = parser.parse_args()

    torch.set_printoptions(precision=2)

    gpus = utils.get_available_gpus()
    print(f"Available GPUs: {gpus}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algorithm = "bfs"
    config_path = "./configs/{}.yaml".format(algorithm)
    config = base_config.read_config(config_path)
    print(config)
    split = "val"

    model = Dnar(config)
    model_path = config.models_directory + "/" + "{}_42_last".format(algorithm)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)

    sampler = create_dataloader(config, split, seed=seed, device=device)

    with torch.no_grad():
        scores = utils.evaluate(model, sampler, utils.METRICS[config.output_type])
        print(scores)

import torch

import utils
from configs import base_config
from generate_data import create_dataloader
from models import Dnar

if __name__ == "__main__":
    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)

    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=2)

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

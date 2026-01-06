import torch
import utils
from configs import base_config
from generate_data import create_dataloader
from models import Dnar
from pathlib import Path
import torch.multiprocessing as mp
import tqdm
import argparse
import wandb
import os
os.environ["RAY_DISABLE_METRICS"] = "1"

def eval_worker(seed: int, config_dict: dict, model_path: str, gpu_id: int):
    """
    Eval worker. 
    Assumes data is ALREADY cached by the main process.
    """
    # 1. Setup Environment
    torch.set_num_threads(1) # Use 1 CPU thread per worker (bottleneck is GPU)
    torch.set_default_dtype(torch.double)
    
    torch.manual_seed(seed)
    
    # 2. Setup Device (Map logical ID)
    if torch.cuda.is_available():
        # gpu_id passed here must be 0, 1, or 2 (Logical IDs), not physical IDs
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        
    model_path = Path(model_path)
    # print(f"Worker started: Seed {seed} on {device} (Model: {model_path.stem})")
    
    config = base_config.Config(**config_dict)
    
    model = Dnar(config)
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        return {'error': str(e), 'model_name': model_path.stem, 'seed': seed}

    model = model.to(device)

    # 3. Load Data
    # num_workers=0 ensures we load from cache or run sequentially (fallback).
    # Since we pre-generated in main, this will hit cache instantly.
    sampler = create_dataloader(config, "test", seed=seed, device=device, num_workers=0)

    with torch.no_grad():
        scores = utils.evaluate(model, sampler, utils.METRICS[config.output_type])

    return {
        'model_path': str(model_path),
        'model_name': model_path.stem,
        'seed': seed,
        'scores': scores
    }


if __name__ == "__main__":
    # 1. Fix Multiprocessing Context
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42**2)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--models', type=str, nargs='+', default=None)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--project', type=str, default='dnar_evaluation')
    parser.add_argument('--no-aggregate', action='store_true')
    args = parser.parse_args()

    # 2. Fix GPU Selection (Logical vs Physical)
    # torch.cuda.device_count() returns the number of visible GPUs (e.g., 3)
    # Valid logical IDs are 0, 1, 2.
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        gpus = list(range(num_devices))
        print(f"Using {num_devices} Visible GPUs (Logical IDs: {gpus})")
    else:
        gpus = [0] # CPU fallback
        print("No GPUs found, using CPU.")

    # ... [Load Config and Model Paths logic is same as before] ...
    if args.config_path:
        config_path = args.config_path
        algorithm = Path(config_path).stem
    else:
        algorithm = "bfs"
        config_path = f"./configs/{algorithm}.yaml"
    
    config = base_config.read_config(config_path)
    split = "test"
    config.problem_size = {split: args.size}
    seeds = [args.seed + i for i in range(args.num_seeds)]

    # [Model Path Discovery Logic same as before...]
    if args.models is not None:
        models = args.models
    else:
        models = [f"{algorithm}_{args.seed + i}" for i in range(args.num_seeds)]
    
    model_paths = []
    for model in models:
        # Check standard path
        p = Path(f"./out/checkpoints_{model}/model_final.pt")
        if p.exists():
            model_paths.append(p)
            continue
        # Check step path
        cp_dir = Path(f"./out/checkpoints_{model}")
        if cp_dir.exists():
            steps = list(cp_dir.glob("model_step_*.pt"))
            if steps:
                max_step = max([int(f.stem.split("_")[-1]) for f in steps])
                model_paths.append(cp_dir / f"model_step_{max_step}.pt")

    if not model_paths:
        print("No models found.")
        exit(1)

    print(f"Models to eval: {len(model_paths)}")
    print(f"Seeds per model: {len(seeds)}")

    #pregen data
    cpu_count = os.cpu_count()
    for seed in seeds:
        print(f">> Pre-generating/Caching data for seed {seed}...")
        # num_workers=cpu_count triggers the Parallel Generation in generate_data.py
        # This will create the cache file.
        create_dataloader(config, split, seed=seed, device='cpu', num_workers=cpu_count)


    config_dict = vars(config) if hasattr(config, '__dict__') else config.__dict__

    # We can use a Pool here safely because data gen inside workers will just read cache
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=min(len(gpus) * 2, len(seeds) * len(model_paths))) as pool:
        tasks = []
        for i, (model_path, seed) in enumerate([(m, s) for m in model_paths for s in seeds]):
            gpu_id = gpus[i % len(gpus)]
            tasks.append(pool.apply_async(
                eval_worker, 
                args=(seed, config_dict, str(model_path), gpu_id)
            ))
        
        results = []
        for task in tqdm.tqdm(tasks, desc="Evaluating"):
            try:
                res = task.get()
                if 'error' in res:
                    print(f"Error: {res['error']}")
                else:
                    results.append(res)
            except Exception as e:
                print(f"Task Crash: {e}")

    # [W&B Logging Logic same as before...]
    print("Logging to W&B...")
    for result in results:
        run = wandb.init(
            project=args.project,
            name=f"{result['model_name']}_seed{result['seed']}",
            config={**config_dict, 'model_path': result['model_path']},
            reinit=True
        )
        wandb.log(result['scores'])
        run.finish()

    print("Done.")
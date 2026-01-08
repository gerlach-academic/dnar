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
import pickle
import time
from collections import defaultdict
from torch_geometric.data import Batch
os.environ["RAY_DISABLE_METRICS"] = "1"

def eval_worker(seed: int, config_dict: dict, model_path: str, gpu_id: int):
    """
    Eval worker. 
    Assumes data is ALREADY cached by the main process.
    """
    # 1. Setup Environment
    torch.set_num_threads(1) # Use 1 CPU thread per worker (bottleneck is GPU)
    
    # Ensure we use float32 for compatibility with trained models and PyG kernels
    torch.set_default_dtype(torch.float32)
    
    torch.manual_seed(seed)
    
    # 2. Setup Device 
    if torch.cuda.is_available():
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
    sampler = create_dataloader(config, "test", seed=seed, device=device)

    with torch.no_grad():
        scores = utils.evaluate(model, sampler, utils.METRICS[config.output_type], show_progress=True)

    return {
        'model_path': str(model_path),
        'model_name': model_path.stem,
        'seed': seed,
        'scores': {k: v for k, v in scores[0].items()} | {'loss': scores[1]}
    }


import pickle
import time
import threading

def _data_to_dict(data):
    """Convert a PyG Data object to a plain dict for safe pickling."""
    return {
        'node_fts': data.node_fts.numpy(),
        'edge_fts': data.edge_fts.numpy(),
        'scalars': data.scalars.numpy(),
        'edge_index': data.edge_index.numpy(),
        'y': data.y.numpy(),
        'pos': data.pos.numpy(),
        'goal': int(data.goal.item()) if data.goal.dim() == 0 else int(data.goal[0].item()),
    }

class AsyncCheckpointSaver:
    """
    Saves checkpoints in a background thread to avoid blocking the main processing loop.
    Debounces requests: if a save is already in progress, the next request is queued 
    (overwriting any previous pending request), ensuring we always save the LATEST state.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.pending_payload = None
        self.thread = None
        
    def save(self, path, payload):
        with self.lock:
            self.pending_payload = (path, payload)
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()
    
    def _worker(self):
        while True:
            with self.lock:
                if self.pending_payload is None:
                    return
                path, payload = self.pending_payload
                self.pending_payload = None
            
            # Save outside the lock
            try:
                # Write to tmp file then rename for atomic safety
                tmp_path = str(path) + ".tmp"
                with open(tmp_path, 'wb') as f:
                    pickle.dump(payload, f)
                os.replace(tmp_path, path)
            except Exception as e:
                print(f"  [AsyncSaver] Failed to save checkpoint: {e}")
    
    def flush(self):
        """Block until all pending saves complete."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

# Global singleton
_checkpoint_saver = AsyncCheckpointSaver()

def save_checkpoint_sync(dataset, checkpoint_path, results, seed_idx, total_points, model_scores, model_losses):
    """Synchronous checkpoint save - use before exit() to ensure data is written."""
    # First flush any pending async saves
    _checkpoint_saver.flush()
    
    buffered_samples = []
    if dataset is not None and hasattr(dataset, 'buffer'):
        # Convert Data objects to dicts for safe serialization
        buffered_samples = [_data_to_dict(d) for d in dataset.buffer]
        print(f"Saving {len(buffered_samples)} buffered (generated but unyielded) samples.")
    
    payload = {
        'results': results,
        'next_seed_idx': seed_idx,
        'partial_state': {
            'processed_count': total_points,
            'model_scores': model_scores,
            'model_losses': model_losses,
            'buffered_samples': buffered_samples
        } if dataset is not None else None
    }
    
    # Synchronous atomic write
    tmp_path = str(checkpoint_path) + ".tmp"
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

def save_checkpoint(dataset, checkpoint_path, results, seed_idx, total_points, model_scores, model_losses):
    """Async checkpoint save - fast but may not complete if process exits immediately."""
    buffered_samples = []
    if dataset is not None and hasattr(dataset, 'buffer'):
        # Convert Data objects to dicts for safe serialization
        buffered_samples = [_data_to_dict(d) for d in dataset.buffer]
    
    payload = {
        'results': results,
        'next_seed_idx': seed_idx,
        'partial_state': {
            'processed_count': total_points,
            'model_scores': model_scores,
            'model_losses': model_losses,
            'buffered_samples': buffered_samples
        }
    }
    
    # Offload serialization and I/O to background thread
    _checkpoint_saver.save(checkpoint_path, payload)

if __name__ == "__main__":
    start_time = time.time()
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
    parser.add_argument('--timeout', type=int, default=0, help='Max Runtime in Minutes')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use (-1 for auto)')
    num_eval_workers = 1 #num of gpus for eval, if set to 1 we use gpu==3
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

    # For large graphs (n >= 1000), LazyDataset is auto-enabled and generates on-the-fly.
    # No need to pre-generate data - it would just accumulate in RAM.
    if args.size < 500:
        cpu_count = os.cpu_count()//3
        for seed in seeds:
            print(f">> Pre-generating/Caching data for seed {seed}...")
            # num_workers=cpu_count triggers the Parallel Generation in generate_data.py
            # This will create the cache file.
            create_dataloader(config, split, seed=seed, device='cpu', num_workers=cpu_count)

        config_dict = vars(config) if hasattr(config, '__dict__') else config.__dict__

        # We use ONE process per GPU to avoid "CUDA device busy" errors and memory oversubscription.
        ctx = mp.get_context('spawn')
        num_procs = min(num_eval_workers, min(len(gpus), len(seeds) * len(model_paths)))
        # gpus = gpus[::-1][:num_procs]  # Reverse to distribute load better
        if args.gpu >=0 and args.gpu < num_devices:
            gpus = [args.gpu] * num_procs
            print(f"Using specified GPU ID {args.gpu} for all eval workers.")
        else:
            gpus = [3,2,1,0][:num_procs]  # Manually set GPU order
        print(f"Starting evaluation pool with {num_procs} processes...")
        
        with ctx.Pool(processes=num_procs) as pool:
            tasks = []
            for i, (model_path, seed) in enumerate([(m, s) for m in model_paths for s in seeds]):
                gpu_id = gpus[i % len(gpus)]
                tasks.append(pool.apply_async(
                    eval_worker, 
                    args=(seed, config_dict, str(model_path), gpu_id)
                ))
            
            results = []
            for i, task in tqdm.tqdm(enumerate(tasks), desc="Evaluating", total=len(tasks)):
                try:
                    res = task.get()
                    if 'error' in res:
                        print(f"Error: {res['error']}")
                    else:
                        results.append(res)
                except Exception as e:
                    print(f"Task Crash: {e}")
    else:
        print(f">> Using LazyDataset for large graphs (n={args.size}) - data generated on-the-fly")
        print(f">> Evaluating {len(model_paths)} models on each batch to avoid regenerating data")

        # Checkpoint path
        user = os.getenv("USER") or "default_user"
        checkpoint_path = Path(f"/hpcwork/{user}/eval/eval_checkpoint_{algorithm}_{args.size}_{args.seed}.pkl")
        results = []
        
        # Load checkpoint if exists
        start_seed_idx = 0
        partial_state = None
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    ckpt = pickle.load(f)
                results = ckpt['results']
                start_seed_idx = ckpt['next_seed_idx']
                partial_state = ckpt.get('partial_state')
                if partial_state:
                     print(f">> Resuming SECONDS (partial) seed {start_seed_idx} at sample {partial_state['processed_count']}")
                else:
                     print(f">> Resuming from checkpoint: {len(results)} seeds done, starting at seed index {start_seed_idx}")
            except Exception as e:
                print(f">> Checkpoint load failed: {e}. Starting fresh.")

        # If lazy we must not use a subprocess, otherwise we cannot have subprocesses for the prefetching
        config_dict = vars(config) if hasattr(config, '__dict__') else config.__dict__
        
        gpu_id = 2 if args.gpu < 0 else args.gpu  # Use last GPU for eval
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Ensure we use float32 for compatibility with trained models and PyG kernels
        torch.set_default_dtype(torch.float32)
        
        # Load ALL models once
        config_obj = base_config.Config(**config_dict)
        
        # For large graphs, use batch_size=1 to avoid GPU OOM
        # Each sample at n=1600 is ~500MB, so batching causes OOM
        original_batch_size = config_obj.batch_size
        config_obj.batch_size = 1
        print(f"  Setting batch_size=1 for large graphs (original={original_batch_size})")
        
        models = {}
        # Skip loading models if we are just going to exit immediately (edge case)
        if start_seed_idx < len(seeds):
            for model_path in model_paths:
                model = Dnar(config_obj)
                try:
                    model.load_state_dict(torch.load(model_path, map_location="cpu"))
                    model = model.to(device)
                    model.eval()
                    models[model_path] = model
                    print(f"  Loaded model: {model_path.stem}")
                except Exception as e:
                    print(f"  Failed to load {model_path.stem}: {e}")
            
            if not models:
                print("No models loaded successfully!")
                exit(1)
                
            calculators = utils.METRICS[config_obj.output_type]
        total_samples = config_obj.num_samples[split]
        # Evaluate: loop over seeds, then batches, then models
        for seed_idx in range(start_seed_idx, len(seeds)):
            seed = seeds[seed_idx]

            # CHECK TIMEOUT BEFORE STARTING SEED
            # Only if NOT resuming a partial seed (we want to try to finish it if we just loaded it!)
            if (args.timeout > 0 and (time.time() - start_time) / 60 > args.timeout) and not (seed_idx == start_seed_idx and partial_state):
                print(f">> Timeout reached ({args.timeout} mins). Saving checkpoint and exiting.")
                # Pass dataset=None because if we haven't started this seed, we have no partial state
                save_checkpoint_sync(None, checkpoint_path, results, seed_idx, 0, {}, {})
                exit(0)

            # Reset or Load Partial State
            start_sample_idx = 0
            num_samples_override = None
            preloaded_buffer = []

            if seed_idx == start_seed_idx and partial_state is not None:
                start_sample_idx = partial_state['processed_count']
                model_scores = partial_state['model_scores']
                model_losses = partial_state['model_losses']
                total_points = partial_state['processed_count']
                 
                # The 'buffered_samples' list contains raw sample dicts we saved at checkpoint
                if 'buffered_samples' in partial_state:
                    preloaded_buffer = partial_state['buffered_samples']

                # Calculate remaining samples to yield (including preloaded buffer)
                num_samples_override = total_samples - start_sample_idx
                
                # Validation: ensure consistency
                print(f"\n>> Resuming seed {seed}:")
                print(f"     Processed: {start_sample_idx}/{total_samples}")
                print(f"     Buffered (to restore): {len(preloaded_buffer)}")
                print(f"     Remaining to yield: {num_samples_override}")
                print(f"     New samples to generate: {num_samples_override - len(preloaded_buffer)}")
                
                # Sanity check
                if num_samples_override < len(preloaded_buffer):
                    print(f"WARNING: Buffer has more samples than remaining! This indicates a checkpoint bug.")
                    print(f"         Trimming buffer to match remaining samples.")
                    preloaded_buffer = preloaded_buffer[:num_samples_override]
            else:
                model_scores = {mp: defaultdict(float) for mp in models}
                model_losses = {mp: 0.0 for mp in models}
                total_points = 0
                print(f"\n>> Seed {seed}: Generating data and evaluating all models...")

            torch.manual_seed(seed)
            
            # Create lazy dataset with prefetching - use iter_as_ready() for maximum throughput
            cpu_count = (os.cpu_count() or 4) // 4
            from generate_data import LazyDataset, ALGORITHMS, ErdosRenyiGraphSampler, GridGraphSampler, RoadmapGraphSampler, GeometricGraphSampler
            import numpy as np
            np.random.seed(seed)
            
            graph_type = getattr(config_obj, 'graph_type', 'er')
            if graph_type == 'grid':
                sampler = GridGraphSampler(config_obj)
            elif graph_type == 'roadmap':
                sampler = RoadmapGraphSampler(config_obj)
            elif graph_type == 'geometric':
                sampler = GeometricGraphSampler(config_obj)
            else:
                sampler = ErdosRenyiGraphSampler(config_obj)
            
            algorithm = ALGORITHMS[config_obj.algorithm]
            dataset = LazyDataset(config_obj, split, sampler, algorithm, num_prefetch_workers=cpu_count, seed=seed, start_idx=start_sample_idx, num_samples_override=num_samples_override, preloaded_buffer=preloaded_buffer)
            
            # Accumulators per model (now handled in if/else above)
            # model_scores = {mp: defaultdict(float) for mp in models}
            # model_losses = {mp: 0.0 for mp in models}
            # total_points = 0
            
            timeout_interrupt = False
            
            # Adaptive batch sizing based on algorithm
            dynamic_max_batch = 64
            dynamic_min_batch = 16
            if config_obj.algorithm == 'dfs' and args.size >= 600:
                # DFS has very long traces (~4000 steps for n=800)
                # But with optimized F.pad collation, we can handle moderate batches
                dynamic_max_batch = 8  # Balance speed vs memory
                dynamic_min_batch = 4
                print(f"  Using reduced batch sizes for DFS (max={dynamic_max_batch}, min={dynamic_min_batch}) due to long traces")
            
            with torch.no_grad():
                # Manual iteration to check timeout BEFORE fetching next batch
                batch_iterator = dataset.iter_batches_as_ready(
                    max_batch_size=min(dynamic_max_batch, total_samples-total_points),
                    min_batch_size=min(dynamic_min_batch, total_samples-total_points)
                )
                
                for batch in batch_iterator:
                    # Save current batch BEFORE processing in case we're forced to yield during processing
                    current_batch_samples = batch.to_data_list()
                    
                    # Check timeout BEFORE processing the batch
                    if args.timeout > 0 and (time.time() - start_time) / 60 > args.timeout:
                        print(f">> Timeout reached. Saving checkpoint with current unprocessed batch.")
                        # Add current batch to buffer for the checkpoint (modify in-place)
                        if hasattr(dataset, 'buffer'):
                            dataset.buffer[:0] = current_batch_samples  # Prepend in-place
                        timeout_interrupt = True
                        break
                    
                    # Add current batch to buffer BEFORE processing, then checkpoint
                    # Modify in-place so generator sees the change
                    if hasattr(dataset, 'buffer'):
                        dataset.buffer[:0] = current_batch_samples  # Prepend in-place
                    save_checkpoint(dataset, checkpoint_path, results, seed_idx, total_points, model_scores, model_losses)
                    print(f"  Pre-processing checkpoint: {total_points} processed, {len(dataset.buffer) if hasattr(dataset, 'buffer') else 0} buffered (includes current batch).")
                    
                    batch = batch.to(device)
                    batch_size = len(current_batch_samples)
                    total_points += batch_size
                    
                    print(f"  Starting on batch of {batch_size} samples (total: {total_points}/{total_samples})")

                    # Evaluate ALL models on this batch
                    for model_path, model in models.items():
                        pred, loss = model(batch, training_step=-1)
                        # Loss is already averaged per sample in the model
                        model_losses[model_path] += loss.detach().item() * batch_size
                        
                        # score() returns SUM of scores across batch, so just add directly
                        score_batch = utils.score(batch, pred, calculators, model.output_type)
                        for calc in calculators:
                            model_scores[model_path][calc.__name__] += score_batch[calc.__name__]
                    
                    print(f"  Processed batch of {batch_size} samples (total: {total_points})")
                    
                    # Remove processed batch from buffer (modify in-place)
                    if hasattr(dataset, 'buffer'):
                        del dataset.buffer[:batch_size]  # Remove first batch_size items in-place
                    save_checkpoint(dataset, checkpoint_path, results, seed_idx, total_points, model_scores, model_losses)
                    print(f"  Post-processing checkpoint: {total_points} processed, {len(dataset.buffer) if hasattr(dataset, 'buffer') else 0} buffered.")

            if timeout_interrupt:
                print(f">> Saving checkpoint (mid-seed {seed}) and exiting.")
                save_checkpoint_sync(dataset, checkpoint_path, results, seed_idx, total_points, model_scores, model_losses)
                exit(0)

            # Finalize scores for this seed
            for model_path in models:
                final_scores = {}
                for calc in calculators:
                    final_scores[calc.__name__] = model_scores[model_path][calc.__name__] / total_points
                final_scores['loss'] = model_losses[model_path] / total_points
                
                results.append({
                    'model_path': str(model_path),
                    'model_name': model_path.stem,
                    'seed': seed,
                    'scores': final_scores
                })
                print(f"  {model_path.stem}: {final_scores}")

            # Save checkpoint after every successful seed
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'next_seed_idx': seed_idx + 1,
                    'partial_state': None
                }, f)

    # [W&B Logging Logic same as before...]
    if not results:
        print("No results to log!")
        exit(1)
    
    print(f"Logging {len(results)} results to W&B...")
    for i, result in enumerate(results):
        print(f"  [{i+1}/{len(results)}] Logging {result['model_name']} seed={result['seed']}...")
        try:
            run = wandb.init(
                project=args.project,
                name=f"{result['model_name']}_seed{result['seed']}_size{args.size}",
                config={**config_dict, 'model_path': result['model_path'], 'eval_size': args.size},
                reinit=True,
                mode="online"  # Force online mode
            )
            wandb.log(result['scores'])
            run.finish()
            print(f"    -> Logged: {result['scores']}")
        except Exception as e:
            print(f"    -> W&B Error: {e}")

    print("Done.")
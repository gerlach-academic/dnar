import argparse
from pathlib import Path
from typing import List, Optional
import itertools
from copy import deepcopy
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from configs import base_config
from generate_data import create_dataloader, SPEC

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DualLogger:
    """
    A logger that writes to both TensorBoard and Weights & Biases.
    Provides the same interface as SummaryWriter.
    """
    def __init__(self, tb_writer: Optional[SummaryWriter] = None, use_wandb: bool = False):
        self.tb_writer = tb_writer
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
    def add_scalar(self, tag: str, scalar_value, global_step: int):
        """Log a scalar value to both TensorBoard and wandb."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)
        
        if self.use_wandb:
            # Convert TensorBoard-style tag (with /) to wandb-style
            wandb.log({tag: scalar_value, "step": global_step}, step=global_step)
    
    def close(self):
        """Close the loggers."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


def create_logger(config: base_config.Config, run_name: str) -> Optional[DualLogger]:
    """
    Create a DualLogger based on config settings.
    
    Args:
        config: Configuration with tensorboard_logs and wandb_logs flags
        run_name: Name for the run (used in both TB and wandb)
    
    Returns:
        DualLogger instance or None if no logging enabled
    """
    tb_writer = None
    use_wandb = False
    
    if config.tensorboard_logs:
        tb_writer = SummaryWriter(comment=f"-{run_name}")
    
    if config.wandb_logs:
        if not WANDB_AVAILABLE:
            print("Warning: wandb_logs=True but wandb is not installed. Run: pip install wandb")
        else:
            # Initialize wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                config={
                    "name": config.name if config.name else config.algorithm,
                    "algorithm": config.algorithm,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "num_iterations": config.num_iterations,
                    "h": config.h,
                    "num_node_states": config.num_node_states,
                    "num_edge_states": config.num_edge_states,
                    "stepwise_training": config.stepwise_training,
                    "multitask_algorithms": config.multitask_algorithms,
                }
            )
            use_wandb = True
    
    if tb_writer is None and not use_wandb:
        return None
    
    return DualLogger(tb_writer, use_wandb)


def evaluate(model, val_data, test_data, metrics_list, model_saver, writer, steps, algorithm=None):
    """
    Evaluate model on validation and test data.
    
    Args:
        algorithm: If provided, used for multitask evaluation (passed to model.forward)
    """
    with torch.no_grad():
        model.eval()
        if algorithm:
            # For multitask, we need to evaluate with the correct algorithm context
            val_scores = utils.evaluate_multitask(model, val_data, metrics_list, algorithm)
            test_scores = utils.evaluate_multitask(model, test_data, metrics_list, algorithm)
            print(f"Eval after {steps} steps [{algorithm}]:")
        else:
            val_scores = utils.evaluate(model, val_data, metrics_list)
            test_scores = utils.evaluate(model, test_data, metrics_list)
            print("Eval after {} steps:".format(steps))
        print("Val scores: ", val_scores)
        print("Test scores: ", test_scores)
        model.train()
    if writer is not None:
        prefix = f"{algorithm}/" if algorithm else ""
        for stat in val_scores:
            writer.add_scalar(f"{prefix}{stat}/val", val_scores[stat], steps)
            writer.add_scalar(f"{prefix}{stat}/test", test_scores[stat], steps)
    model_saver.visit(model, val_scores)


def train(config: base_config.Config, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Dnar(config).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    model_name = "{}_{}".format(config.name if config.name else config.algorithm, seed)
    model_saver = utils.ModelSaver(config.models_directory, model_name)

    train_data = create_dataloader(config, "train", seed=seed, device=device)
    val_data = create_dataloader(config, "val", seed=seed + 1, device=device)
    test_data = create_dataloader(config, "test", seed=seed + 2, device=device)

    writer = create_logger(config, model_name)

    model.train()

    # Checkpoint setup
    checkpoint_interval = config.num_iterations * config.checkpoint_interval #is a portion
    next_checkpoint = checkpoint_interval
    checkpoint_dir = Path(config.out_directory) / f"checkpoints_{model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training:")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Checkpoints saved every {checkpoint_interval} steps to: {checkpoint_dir}")

    steps = 0
    while steps <= config.num_iterations:
        for batch in train_data:
            steps += 1

            _, loss = model(batch, writer, training_step=steps)
            assert not torch.isnan(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if steps % config.eval_each == 1:
                evaluate(
                    model,
                    val_data,
                    test_data,
                    utils.METRICS[config.output_type],
                    model_saver,
                    writer,
                    steps,
                )
            
            # Progress logging
            if steps % 100 == 0:
                print(f"Step {steps}/{config.num_iterations} | Loss: {loss.item():.4f}")
            
            # Checkpoint evaluation
            if checkpoint_interval and steps > next_checkpoint:
                progress_pct = int(100 * steps / config.num_iterations)
                print(f"\n{'='*60}")
                print(f"Checkpoint at {progress_pct}% ({steps}/{config.num_iterations} steps)")
                print(f"{'='*60}")
                
                # Save model checkpoint
                checkpoint_path = checkpoint_dir / f"model_step_{steps}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to: {checkpoint_path}")
                
                # Detailed evaluation
                with torch.no_grad():
                    model.eval()
                    
                    # Validation set
                    val_output_path = checkpoint_dir / f"eval_val_step_{steps}.json"
                    utils.evaluate_print(
                        model, 
                        val_data, 
                        utils.METRICS[config.output_type],
                        output_path=str(val_output_path),
                        step=steps,
                        split="val",
                        print_results=False,
                    )
                    
                    # Test set
                    test_output_path = checkpoint_dir / f"eval_test_step_{steps}.json"
                    utils.evaluate_print(
                        model,
                        test_data,
                        utils.METRICS[config.output_type],
                        output_path=str(test_output_path),
                        step=steps,
                        split="test",
                        print_results=False,
                    )
                    model.train()
                
                next_checkpoint += checkpoint_interval
                print(f"{'='*60}\n")

            if steps >= config.num_iterations:
                break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (100%)")
    print("=" * 60)
    
    # Save final model
    final_model_path = checkpoint_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    model.eval()
    
    with torch.no_grad():
        # Quick evaluation for model saver
        evaluate(
            model,
            val_data,
            test_data,
            utils.METRICS[config.output_type],
            model_saver,
            writer,
            steps,
        )
        
        # Detailed final evaluation
        val_output_path = checkpoint_dir / f"eval_val_final.json"
        utils.evaluate_print(
            model, 
            val_data, 
            utils.METRICS[config.output_type],
            output_path=str(val_output_path),
            step=steps,
            split="val",
            print_results=False,
        )
        
        test_output_path = checkpoint_dir / f"eval_test_final.json"
        utils.evaluate_print(
            model,
            test_data,
            utils.METRICS[config.output_type],
            output_path=str(test_output_path),
            step=steps,
            split="test",
            print_results=False,
        )
    
    if writer is not None:
        writer.close()
    
    # Generate summary
    _save_training_summary_single(checkpoint_dir, config.name if config.name else config.algorithm, steps)
    
    return model


def configs_from_multitask_config(config: base_config.Config) -> List[base_config.Config]:
    """
    Create per-algorithm configs from a single multitask config.
    
    When a config has multitask_algorithms set (e.g., ["bfs", "dijkstra", "dfs"]),
    this function creates a separate config for each algorithm with the correct
    num_node_states and num_edge_states based on the algorithm's SPEC.
    
    Args:
        config: A config with multitask_algorithms list set
        
    Returns:
        List of configs, one per algorithm
    """
    if not config.multitask_algorithms:
        raise ValueError("Config must have multitask_algorithms set")
    
    configs = []
    for algorithm in config.multitask_algorithms:
        if algorithm not in SPEC:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(SPEC.keys())}")
        
        # Get state counts from SPEC
        spec = SPEC[algorithm]
        num_node_states = len(spec[0])  # First tuple is node states
        num_edge_states = len(spec[1])  # Second tuple is edge states
        
        # Create a copy with algorithm-specific settings
        algo_config = base_config.Config(
            algorithm=algorithm,
            graph_type=config.graph_type,
            use_lazy_dataset=config.use_lazy_dataset,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            num_iterations=config.num_iterations,
            eval_each=config.eval_each,
            stepwise_training=config.stepwise_training,
            processor_upper_t=config.processor_upper_t,
            processor_lower_t=config.processor_lower_t,
            use_noise=config.use_noise,
            num_samples=config.num_samples,
            problem_size=config.problem_size,
            edge_weights=config.edge_weights,
            generate_random_numbers=config.generate_random_numbers,
            h=config.h,
            temp_on_eval=config.temp_on_eval,
            checkpoint_interval=config.checkpoint_interval,
            num_node_states=num_node_states,
            num_edge_states=num_edge_states,
            output_type=config.output_type,
            output_idx=config.output_idx,
            models_directory=config.models_directory,
            tensorboard_logs=config.tensorboard_logs,
            wandb_logs=config.wandb_logs,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
        )
        configs.append(algo_config)
    
    return configs


def train_multitask(configs: List[base_config.Config], seed: int):
    """
    Multitask training across multiple algorithms.
    
    Creates a single model with algorithm-specific encoders/decoders (embeddings, projections)
    while sharing the latent processor. Training alternates between algorithms.
    
    Args:
        configs: List of configurations for different algorithms. 
                 All configs should have matching hyperparameters except:
                 - algorithm (must be different)
                 - num_node_states, num_edge_states (will use max across all)
        seed: Random seed for reproducibility.
    
    Returns:
        Trained multitask model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validate configs
    algorithms = [cfg.algorithm for cfg in configs]
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("Each config must have a unique algorithm name")
    
    print(f"Training multitask model on algorithms: {algorithms}")
    
    # Determine the maximum number of node and edge states required
    max_node_states = max(cfg.num_node_states for cfg in configs)
    max_edge_states = max(cfg.num_edge_states for cfg in configs)
    
    print(f"Using max states - nodes: {max_node_states}, edges: {max_edge_states}")
    
    # Create a unified config for the model (copy first config and modify)
    unified_config = base_config.Config(
        algorithm=configs[0].algorithm,  # Default algorithm for spec initialization
        graph_type=configs[0].graph_type,
        batch_size=configs[0].batch_size,
        learning_rate=configs[0].learning_rate,
        weight_decay=configs[0].weight_decay,
        num_iterations=configs[0].num_iterations,
        eval_each=configs[0].eval_each,
        stepwise_training=configs[0].stepwise_training,
        processor_upper_t=configs[0].processor_upper_t,
        processor_lower_t=configs[0].processor_lower_t,
        use_noise=configs[0].use_noise,
        num_samples=configs[0].num_samples,
        problem_size=configs[0].problem_size,
        edge_weights=configs[0].edge_weights,
        generate_random_numbers=configs[0].generate_random_numbers,
        h=configs[0].h,
        temp_on_eval=configs[0].temp_on_eval,
        checkpoint_interval=configs[0].checkpoint_interval,
        num_node_states=max_node_states,
        num_edge_states=max_edge_states,
        output_type=configs[0].output_type,
        output_idx=configs[0].output_idx,
        models_directory=configs[0].models_directory,
        tensorboard_logs=configs[0].tensorboard_logs,
        wandb_logs=getattr(configs[0], 'wandb_logs', False),
        wandb_project=getattr(configs[0], 'wandb_project', 'dnar'),
        wandb_entity=getattr(configs[0], 'wandb_entity', None),
        multitask_num_algorithms=len(configs),
        multitask_algorithms=algorithms,
    )
    
    # Create model with multitask enabled
    model = models.Dnar(unified_config, multitask_num_algorithms=len(configs)).to(device)
    
    # Print multitask info
    if hasattr(model, 'get_multitask_info'):
        info = model.get_multitask_info()
        print(f"Multitask model info: {info}")

    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=unified_config.learning_rate, 
        weight_decay=unified_config.weight_decay
    )

    model_name = "multitask_{}_{}".format("_".join(algorithms), seed)
    model_saver = utils.ModelSaver(unified_config.models_directory, model_name)

    # Create dataloaders for each algorithm (using their original configs for correct data generation)
    train_dataloaders = {}
    val_dataloaders = {}
    test_dataloaders = {}
    
    for cfg in configs:
        # Adjust config to use max states for padding
        cfg.num_node_states = max_node_states
        cfg.num_edge_states = max_edge_states
        
        train_dataloaders[cfg.algorithm] = create_dataloader(cfg, "train", seed=seed, device=device)
        val_dataloaders[cfg.algorithm] = create_dataloader(cfg, "val", seed=seed + 1, device=device)
        test_dataloaders[cfg.algorithm] = create_dataloader(cfg, "test", seed=seed + 2, device=device)

    writer = create_logger(unified_config, model_name)

    model.train()
    
    # Create iterators for interleaved training
    train_iterators = {alg: iter(dl) for alg, dl in train_dataloaders.items()}
    
    # Total steps = num_iterations * num_algorithms
    # This ensures each algorithm gets num_iterations steps worth of training
    # and the latent processor sees num_algorithms times more data
    total_steps = unified_config.num_iterations * len(algorithms)
    steps_per_algorithm = {alg: 0 for alg in algorithms}
    
    steps = 0
    
    # Checkpoint intervals for detailed evaluation (10%, 20%, ..., 100%)
    checkpoint_interval = total_steps // unified_config.checkpoint_interval
    next_checkpoint = checkpoint_interval
    checkpoint_dir = Path(unified_config.out_directory) / f"checkpoints_{model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting multitask training:")
    print(f"  - {len(algorithms)} algorithms: {algorithms}")
    print(f"  - {unified_config.num_iterations} iterations per algorithm")
    print(f"  - {total_steps} total steps (interleaved)")
    print(f"  - Expected time: ~{len(algorithms)}x single-algorithm training")
    print(f"  - Checkpoints saved every {checkpoint_interval} steps to: {checkpoint_dir}")
    
    while steps < total_steps:
        # Cycle through algorithms - one batch each per round
        for algorithm in algorithms:
            if steps >= total_steps:
                break
                
            # Get next batch for this algorithm (reset iterator if exhausted)
            try:
                batch = next(train_iterators[algorithm])
            except StopIteration:
                train_iterators[algorithm] = iter(train_dataloaders[algorithm])
                batch = next(train_iterators[algorithm])
            
            steps += 1
            steps_per_algorithm[algorithm] += 1
            
            # Forward pass with algorithm-specific components
            _, loss = model(batch, writer, training_step=steps_per_algorithm[algorithm], multitask_algorithm=algorithm)
            
            if torch.isnan(loss):
                raise ValueError(f"NaN loss at step {steps} for algorithm {algorithm}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            # Log loss per algorithm (use per-algorithm step for x-axis)
            if writer is not None:
                writer.add_scalar(f"{algorithm}/Loss/train", loss.detach().item(), steps_per_algorithm[algorithm])
                writer.add_scalar("Loss/train_combined", loss.detach().item(), steps)

            # Evaluate periodically (per algorithm, based on that algorithm's step count)
            if steps_per_algorithm[algorithm] % unified_config.eval_each == 1:
                evaluate(
                    model,
                    val_dataloaders[algorithm],
                    test_dataloaders[algorithm],
                    utils.METRICS[unified_config.output_type],
                    model_saver,
                    writer,
                    steps_per_algorithm[algorithm],
                    algorithm=algorithm,
                )
            
            # Progress logging
            if steps % (100 * len(algorithms)) == 0:
                alg_progress = ", ".join([f"{a}: {steps_per_algorithm[a]}" for a in algorithms])
                print(f"Step {steps}/{total_steps} | Per-algorithm steps: {alg_progress}")
                print(f"  Last: {algorithm}, Loss: {loss.item():.4f}")
            
            # Checkpoint evaluation (every 10% of training)
            if checkpoint_interval and steps > next_checkpoint: # not the final one here.
                progress_pct = int(100 * steps / total_steps)
                print(f"\n{'='*60}")
                print(f"Checkpoint at {progress_pct}% ({steps}/{total_steps} steps)")
                print(f"{'='*60}")
                
                # Save model checkpoint
                checkpoint_path = checkpoint_dir / f"model_step_{steps}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to: {checkpoint_path}")
                
                # Detailed evaluation for each algorithm
                with torch.no_grad():
                    model.eval()
                    for alg in algorithms:
                        # Evaluate on validation set with detailed output
                        val_output_path = checkpoint_dir / f"eval_{alg}_val_step_{steps}.json"
                        utils.evaluate_print(
                            model, 
                            val_dataloaders[alg], 
                            utils.METRICS[unified_config.output_type],
                            output_path=str(val_output_path),
                            algorithm=alg,
                            step=steps,
                            split="val"
                        )
                        
                        # Evaluate on test set with detailed output
                        test_output_path = checkpoint_dir / f"eval_{alg}_test_step_{steps}.json"
                        utils.evaluate_print(
                            model,
                            test_dataloaders[alg],
                            utils.METRICS[unified_config.output_type],
                            output_path=str(test_output_path),
                            algorithm=alg,
                            step=steps,
                            split="test"
                        )
                    model.train()
                
                next_checkpoint += checkpoint_interval
                print(f"{'='*60}\n")

    # Final evaluation on all algorithms with detailed output
    print("\n" + "=" * 60)
    print("Final Evaluation (100%)")
    print("=" * 60)
    print(f"Steps per algorithm: {steps_per_algorithm}")
    
    # Save final model
    final_model_path = checkpoint_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    model.eval()
    
    with torch.no_grad():
        for algorithm in algorithms:
            # Quick evaluation for model saver
            evaluate(
                model,
                val_dataloaders[algorithm],
                test_dataloaders[algorithm],
                utils.METRICS[unified_config.output_type],
                model_saver,
                writer,
                steps_per_algorithm[algorithm],
                algorithm=algorithm,
            )
            
            # Detailed final evaluation
            val_output_path = checkpoint_dir / f"eval_{algorithm}_val_final.json"
            utils.evaluate_print(
                model, 
                val_dataloaders[algorithm], 
                utils.METRICS[unified_config.output_type],
                output_path=str(val_output_path),
                algorithm=algorithm,
                step=total_steps,
                split="val",
                print_results=False,
            )
            
            test_output_path = checkpoint_dir / f"eval_{algorithm}_test_final.json"
            utils.evaluate_print(
                model,
                test_dataloaders[algorithm],
                utils.METRICS[unified_config.output_type],
                output_path=str(test_output_path),
                algorithm=algorithm,
                step=total_steps,
                split="test",
                print_results=False,
            )
    
    if writer is not None:
        writer.close()
    
    # Generate summary of all checkpoints
    _save_training_summary(checkpoint_dir, algorithms, total_steps, steps_per_algorithm)
    
    return model


def _save_training_summary(checkpoint_dir: Path, algorithms: List[str], total_steps: int, 
                           steps_per_algorithm: dict):
    """
    Generate a summary JSON file with metrics from all checkpoints.
    """
    import json
    from datetime import datetime
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "algorithms": algorithms,
        "total_steps": total_steps,
        "steps_per_algorithm": steps_per_algorithm,
        "checkpoints": {}
    }
    
    # Collect metrics from all evaluation files
    for eval_file in sorted(checkpoint_dir.glob("eval_*.json")):
        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
            
            # Parse filename to get info
            # Format: eval_{algorithm}_{split}_step_{step}.json or eval_{algorithm}_{split}_final.json
            parts = eval_file.stem.split("_")
            algorithm = parts[1]
            split = parts[2]
            step_info = "_".join(parts[3:])  # "step_1000" or "final"
            
            key = f"{algorithm}_{split}_{step_info}"
            summary["checkpoints"][key] = {
                "algorithm": algorithm,
                "split": split,
                "step": eval_data.get("step"),
                "accuracy": eval_data["summary"]["accuracy"],
                "graph_level_accuracy": eval_data["summary"]["graph_level_accuracy"],
                "mistake_rate": eval_data["summary"]["mistake_rate"],
            }
        except Exception as e:
            print(f"Warning: Could not process {eval_file}: {e}")
    
    # Save summary
    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Training Progress Summary")
    print("=" * 80)
    print(f"{'Checkpoint':<30} {'Algorithm':<12} {'Split':<6} {'Accuracy':<10} {'Graph Acc':<10}")
    print("-" * 80)
    for key, data in sorted(summary["checkpoints"].items()):
        print(f"{key:<30} {data['algorithm']:<12} {data['split']:<6} {data['accuracy']:.4f}     {data['graph_level_accuracy']:.4f}")
    print("=" * 80)


def _save_training_summary_single(checkpoint_dir: Path, algorithm: str, total_steps: int):
    """
    Generate a summary JSON file for single-task training.
    """
    import json
    from datetime import datetime
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": algorithm,
        "total_steps": total_steps,
        "checkpoints": {}
    }
    
    # Collect metrics from all evaluation files
    for eval_file in sorted(checkpoint_dir.glob("eval_*.json")):
        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
            
            # Parse filename: eval_{split}_step_{step}.json or eval_{split}_final.json
            parts = eval_file.stem.split("_")
            split = parts[1]
            step_info = "_".join(parts[2:])  # "step_1000" or "final"
            
            key = f"{split}_{step_info}"
            summary["checkpoints"][key] = {
                "split": split,
                "step": eval_data.get("step"),
                "accuracy": eval_data["summary"]["accuracy"],
                "graph_level_accuracy": eval_data["summary"]["graph_level_accuracy"],
                "mistake_rate": eval_data["summary"]["mistake_rate"],
            }
        except Exception as e:
            print(f"Warning: Could not process {eval_file}: {e}")
    
    # Save summary
    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Training Progress Summary")
    print("=" * 80)
    print(f"{'Checkpoint':<30} {'Split':<6} {'Accuracy':<10} {'Graph Acc':<10}")
    print("-" * 80)
    for key, data in sorted(summary["checkpoints"].items()):
        print(f"{key:<30} {data['split']:<6} {data['accuracy']:.4f}     {data['graph_level_accuracy']:.4f}")
    print("=" * 80)


# Worker function für paralleles Training
def train_worker(args):
    """Worker für paralleles Training eines einzelnen Seeds"""
    config_path, seed, multitask, gpu_id = args
    
    # Set CUDA device für diesen Worker
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Set default dtype statt deprecated set_default_tensor_type
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(5)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n[Seed {seed}] Starting training on GPU {gpu_id}...")
    
    if multitask:
        # Check if it's a single config with multitask_algorithms or multiple configs
        if "," in config_path:
            config_paths = [p.strip() for p in config_path.split(",")]
            configs = [base_config.read_config(path) for path in config_paths]
        else:
            base_cfg = base_config.read_config(config_path)
            if not base_cfg.multitask_algorithms:
                raise ValueError(
                    "For multitask training with a single config, "
                    "'multitask_algorithms' must be set"
                )
            configs = configs_from_multitask_config(base_cfg)
        
        model = train_multitask(configs, seed)
    else:
        config = base_config.read_config(config_path)
        model = train(config, seed)
    
    print(f"[Seed {seed}] Training complete!")
    return seed


if __name__ == "__main__":
    # WICHTIG: Muss für torch.multiprocessing gesetzt werden
    mp.set_start_method('spawn', force=True)
    
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(5)
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/bfs.yaml",
                        help="Path to config file. For multitask, can be: "
                             "1) A single config with 'multitask_algorithms' list, or "
                             "2) Comma-separated paths to multiple config files")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--multitask", action="store_true",
                        help="Enable multitask training")
    parser.add_argument("--parallel", action="store_true",
                        help="Train seeds in parallel")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: num_seeds)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2'). If not set, uses CUDA_VISIBLE_DEVICES or all GPUs")

    options = parser.parse_args()
    
    seeds = list(range(40, 40 + options.num_seeds))
    
    if options.parallel:
        # Bestimme verfügbare GPUs
        if options.gpus:
            gpu_ids = [int(x.strip()) for x in options.gpus.split(',')]
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if not gpu_ids:
            raise ValueError("No GPUs available for parallel training")
        
        num_workers = options.num_workers or len(seeds)
        print(f"\nParallel training:")
        print(f"  Workers: {num_workers}")
        print(f"  Seeds: {seeds}")
        print(f"  GPUs: {gpu_ids}")
        
        # Verteile Seeds auf GPUs (round-robin)
        worker_args = [
            (options.config_path, seed, options.multitask, gpu_ids[i % len(gpu_ids)]) 
            for i, seed in enumerate(seeds)
        ]
        
        # Starte Pool mit spawn context
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(train_worker, worker_args)
        
        print(f"\nAll seeds completed: {results}")
    else:
        # Sequentielles Training (Original)
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            if options.multitask:
                if "," in options.config_path:
                    config_paths = [p.strip() for p in options.config_path.split(",")]
                    configs = [base_config.read_config(path) for path in config_paths]
                    print(f"Multitask training with {len(configs)} separate config files")
                else:
                    base_cfg = base_config.read_config(options.config_path)
                    if not base_cfg.multitask_algorithms:
                        raise ValueError(
                            "For multitask training with a single config, "
                            "'multitask_algorithms' must be set"
                        )
                    configs = configs_from_multitask_config(base_cfg)
                    print(f"Multitask training from single config: {base_cfg.multitask_algorithms}")
                
                model = train_multitask(configs, seed)
            else:
                print("Train with config {}".format(options.config_path))
                config = base_config.read_config(options.config_path)
                model = train(config, seed)
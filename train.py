import argparse
from pathlib import Path
from time import time
from typing import List, Optional, Any
import itertools
from copy import deepcopy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from configs import base_config
from generate_data import create_dataloader, SPEC
from utils import TrainingSession, RestartManager, get_temp_model_dir, finalize_model, get_least_used_gpus, get_gpus

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DualLogger:
    """
    A logger that writes to both TensorBoard and Weights & Biases.
    """
    def __init__(self, tb_writer: Optional[SummaryWriter] = None, wandb_writer: Optional[wandb.sdk.wandb_run.Run] = None):
        self.tb_writer = tb_writer
        self.wandb_writer = wandb_writer
        self.use_wandb = (wandb_writer is not None) and WANDB_AVAILABLE

    def add_scalar(self, tag: str, scalar_value, global_step: int):
        """Log a scalar value to both TensorBoard and wandb."""

        if self.tb_writer is not None:
            if isinstance(scalar_value, dict):
                for k, v in scalar_value.items():
                    self.tb_writer.add_scalar(f"{tag}/{k}", v, global_step)
            else:   
                self.tb_writer.add_scalar(tag, scalar_value, global_step)

        if self.use_wandb:
            # Note: This might increment the internal step if called repeatedly.
            # Use log_dict for atomic multi-metric logging.
            if self.wandb_writer is not None and global_step <= self.wandb_writer.step:
                if not getattr(self, "_wandb_ahead_warning_shown", False):
                    print(f"Warning: WandB is ahead (step {self.wandb_writer.step}) of local (step {global_step}). "
                            "Skipping redundant logs until caught up.")
                    self._wandb_ahead_warning_shown = True
                return
            try:
                if isinstance(scalar_value, dict):
                    log_dict = {f"{tag}/{k}": v for k, v in scalar_value.items()}
                    self.wandb_writer.log(log_dict, step=global_step)
                else:
                    self.wandb_writer.log({tag: scalar_value}, step=global_step)
            except Exception as e:
                print(f"Warning: wandb log failed for {tag}: {e}")

    def log_dict(self, metrics: dict[str, Any], step: int):
        """
        Log a dictionary of metrics to a specific step atomically.
        This ensures all metrics appear at the same x-axis point in WandB.
        """
        if self.tb_writer is not None:
            for tag, value in metrics.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        self.tb_writer.add_scalar(f"{tag}/{k}", v, step)
                else:
                    self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            if self.wandb_writer is not None and step <= self.wandb_writer.step:
                if not getattr(self, "_wandb_ahead_warning_shown", False):
                    print(f"Warning: WandB history (step {self.wandb_writer.step}) is ahead of local training (step {step}). "
                            "Skipping upload of replayed steps.")
                    self._wandb_ahead_warning_shown = True
                return
            try:
                #unify the double dict case
                log_dict = {}
                for tag, value in metrics.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            log_dict[f"{tag}/{k}"] = v
                    else:
                        log_dict[tag] = value
                self.wandb_writer.log(log_dict, step=step)
            except Exception as e:
                print(f"Warning: wandb log_dict failed: {e}")

    def close(self):
        """Close the loggers."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except Exception:
                pass
        if self.use_wandb and self.wandb_writer is not None:
            try:
                if getattr(self.wandb_writer, "finish", None) is not None:
                    self.wandb_writer.finish()
            except Exception as e:
                print(f"Warning: wandb finish failed: {e}")

def create_logger(config: base_config.Config, run_name: str, wandb_run_id: Optional[str]) -> Optional[DualLogger]:
    """
    Create a DualLogger based on config settings.
    
    Args:
        config: Configuration with tensorboard_logs and wandb_logs flags
        run_name: Name for the run (used in both TB and wandb)
    
    Returns:
        DualLogger instance or None if no logging enabled
    """
    tb_writer = None
    wandb_writer = None
    use_wandb = False

    if config.tensorboard_logs:
        tb_writer = SummaryWriter(comment=f"-{run_name}")

    if config.wandb_logs:
        if not WANDB_AVAILABLE:
            print("Warning: wandb_logs=True but wandb is not installed. Run: pip install wandb")
        else:
            # Reuse existing run if present and matches our id, otherwise init new run.
            run_id = wandb_run_id if wandb_run_id is not None else f"{run_name}_{int(time())}"
            try:
                if wandb.run is not None and getattr(wandb.run, "id", None) == run_id:
                    wandb_writer = wandb.run
                else:
                    # Use explicit id + resume to make behavior deterministic across restarts
                    wandb_writer = wandb.init(
                        project=config.wandb_project,
                        entity=config.wandb_entity,
                        name=run_name,
                        id=run_id,
                        resume="allow",
                        config={
                            "name": config.name if getattr(config, "name", None) else config.algorithm,
                            "algorithm": config.algorithm,
                            "batch_size": config.batch_size,
                            "learning_rate": config.learning_rate,
                            "weight_decay": config.weight_decay,
                            "num_iterations": config.num_iterations,
                            "h": config.h,
                            "num_node_states": config.num_node_states,
                            "num_edge_states": config.num_edge_states,
                            "stepwise_training": config.stepwise_training,
                            "multitask_algorithms": getattr(config, "multitask_algorithms", None),
                        },
                        # Do NOT use reinit=True here; we control init explicitly.
                    )
                use_wandb = True
            except Exception as e:
                print(f"Warning: wandb.init() failed: {e}. Continuing without wandb.")
                wandb_writer = None
                use_wandb = False

    if tb_writer is None and not use_wandb:
        return None

    return DualLogger(tb_writer, wandb_writer)


def evaluate(model, val_data, test_data, metrics_list, model_saver, writer, steps, algorithm=None):
    """
    Evaluate model on validation and test data.
    
    Args:
        algorithm: If provided, used for multitask evaluation (passed to model.forward)
    """
    with torch.no_grad():
        model.eval()
        if algorithm:
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
    
    # If a writer is provided (Single Task), log immediately.
    # For Multitask, we pass writer=None and handle logging in the loop buffer.
    if writer is not None:
        prefix = f"{algorithm}/" if algorithm else ""
        for stat in val_scores:
            writer.add_scalar(f"{prefix}{stat}/val", val_scores[stat], steps)
            writer.add_scalar(f"{prefix}{stat}/test", test_scores[stat], steps)
    model_saver.visit(model, val_scores)

    return val_scores, test_scores

def train(config: base_config.Config, seed, session: Optional[TrainingSession] = None, gpu_id: Optional[int] = None):
    device = torch.device(
        (f"cuda:{gpu_id}" if gpu_id else get_gpus(1)[0]) 
            if torch.cuda.is_available() else "cpu"
    )

    model = models.Dnar(config).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    model_name = "{}_{}".format(config.name if config.name else config.algorithm, seed)
    
    # Use temp directory for model saving during training
    temp_model_dir = get_temp_model_dir(config.models_directory, model_name) if session else None
    model_saver_dir = str(temp_model_dir) if temp_model_dir else config.models_directory
    model_saver = utils.ModelSaver(model_saver_dir, model_name)

    train_data:DataLoader = create_dataloader(config, "train", seed=seed, device=device)
    val_data:DataLoader = create_dataloader(config, "val", seed=seed + 1, device=device)
    test_data:DataLoader = create_dataloader(config, "test", seed=seed + 2, device=device)

    writer = create_logger(config, model_name, session.get_wandb_run_id() if session else None)

    model.train()

    # Checkpoint setup
    checkpoint_interval = config.num_iterations * config.checkpoint_interval if config.checkpoint_interval > 0 else 0
    next_checkpoint = checkpoint_interval
    checkpoint_dir = Path(config.out_directory) / f"checkpoints_{model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if session provided
    resume_from_step = 0
    if session and session.get_resume_step() > 0:
        resume_from_step = session.get_resume_step()
        last_checkpoint = session.get_last_checkpoint()
        if last_checkpoint and Path(last_checkpoint).exists():
            print(f"Loading checkpoint from {last_checkpoint}")
            # CHANGED: Use simple torch.load for single-task
            checkpoint = torch.load(last_checkpoint)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                resume_from_step = checkpoint.get('step', resume_from_step)
            else:
                model.load_state_dict(checkpoint)
        print(f"Resuming from step {resume_from_step}")
        
        # Fast-forward checkpoint counter
        if checkpoint_interval > 0:
            next_checkpoint = ((resume_from_step // checkpoint_interval) + 1) * checkpoint_interval
        
        # Increment restart counter
        session.increment_restart_count()
    
    print(f"\nStarting training:")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Resume from: {resume_from_step}")
    if checkpoint_interval > 0:
        print(f"  Checkpoints every {checkpoint_interval} steps ({config.checkpoint_interval} checkpoints total)")
        print(f"  Checkpoint dir: {checkpoint_dir}")
    if session:
        print(f"  Max runtime: {session.max_runtime_seconds}s")
        print(f"  Temp model dir: {temp_model_dir}")
        print(f"  Restarts so far: {session.state.get('restarts', 0)}")

    steps = resume_from_step
    training_interrupted = False
    early_stopped = False
    while steps <= config.num_iterations:
        for batch in train_data:
            # Skip batches until we reach resume point (already at resume_from_step)
            if steps < resume_from_step:
                steps += 1
                continue
            
            steps += 1

            # Check timeout
            if session and session.should_stop():
                print(f"\nReaching time limit, saving checkpoint at step {steps}...")
                checkpoint_path = checkpoint_dir / f"model_step_{steps}_interrupt.pt"
                # CHANGED: Save as dict with step info
                torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict()
                }, checkpoint_path)
                session.save_state(steps, str(checkpoint_path))
                training_interrupted = True
                break

            batch = batch.to(device)

            _, loss = model(batch, writer, training_step=steps)
            assert not torch.isnan(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if steps % config.eval_each == 1:
                val_scores = evaluate(
                    model,
                    val_data,
                    test_data,
                    utils.METRICS[config.output_type],
                    model_saver,
                    writer,
                    steps,
                )

                #early stopping check
                if session:
                    # Determine metric name
                    metric_name = "pointer_accuracy_graph_level" if config.output_type == "pointer" else "node_mask_accuracy_graph_level"
                    current_score = val_scores.get(metric_name, 0.0)
                    if session.check_early_stopping(current_score):
                        print(f"\nEarly stopping triggered! (Val Acc: {current_score:.4f})")
                        training_interrupted = True
                        early_stopped = True
                        break
            
            # Progress logging
            if steps % 100 == 0:
                elapsed = session.elapsed_time() if session else 0
                print(f"Step {steps}/{config.num_iterations} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
            
            # Checkpoint evaluation
            if checkpoint_interval > 0 and steps >= next_checkpoint and steps < config.num_iterations:
                progress_pct = int(100 * steps / config.num_iterations)
                print(f"\n{'='*60}")
                print(f"Checkpoint at {progress_pct}% ({steps}/{config.num_iterations} steps)")
                print(f"{'='*60}")
                
                # Save model checkpoint
                checkpoint_path = checkpoint_dir / f"model_step_{steps}.pt"
                # CHANGED: Save as dict with step info
                torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict()
                }, checkpoint_path)
                print(f"Model saved to: {checkpoint_path}")
                
                # Save session state
                if session:
                    session.save_state(steps, str(checkpoint_path))
                
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
        
        if training_interrupted:
            break
    
    if training_interrupted:
        if not early_stopped:
            print(f"\nTraining interrupted at step {steps}. Restart with --restart to continue.")
            #don't close as this will finish the runs
            # if writer is not None:
            #     writer.close()
            return None
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (100%)")
    print("=" * 60)
    
    # Save final model to temp first
    if temp_model_dir:
        temp_model_path = temp_model_dir / f"{model_name}_final.pt"
        torch.save(model.state_dict(), temp_model_path)
        print(f"Temp model saved to: {temp_model_path}")
    
    # Save checkpoint to permanent location
    final_checkpoint_path = checkpoint_dir / "model_final.pt"
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    
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
    _save_training_summary_single(checkpoint_dir, config.algorithm, steps)
    
    # Finalize model (move from temp to permanent location)
    if temp_model_dir:
        final_model_path = Path(config.models_directory) / f"{model_name}_final.pt"
        finalize_model(str(temp_model_path), str(final_model_path))
        
        # Mark session as completed
        if session:
            session.mark_completed(str(final_model_path))
            session.cleanup()
    
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
    
    if 'mis' in config.multitask_algorithms and not config.generate_random_numbers:
        print("INFO: Automatically enabling generate_random_numbers for MIS algorithm")
        config.generate_random_numbers = True

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


def train_multitask(configs: List[base_config.Config], seed: int, 
                    session: Optional[TrainingSession] = None, gpu_id: Optional[int] = None):
    """
    Multitask training with restart support.
    """
    device = torch.device(
        (f"cuda:{gpu_id}" if gpu_id else get_gpus(1)[0]) 
        if torch.cuda.is_available() else "cpu"
    )
    
    
    # Validate configs
    algorithms = [cfg.algorithm for cfg in configs]
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("Each config must have a unique algorithm name")
    
    print(f"Training multitask model on algorithms: {algorithms}")
    
    # Determine the maximum number of node and edge states required
    max_node_states = max(cfg.num_node_states for cfg in configs)
    max_edge_states = max(cfg.num_edge_states for cfg in configs)
    
    print(f"Using max states - nodes: {max_node_states}, edges: {max_edge_states}")
    
    # Create unified config
    unified_config = base_config.Config(
        algorithm=configs[0].algorithm,
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

    # Create dataloaders for each algorithm
    train_dataloaders = {}
    val_dataloaders = {}
    test_dataloaders = {}
    
    for cfg in configs:
        cfg.num_node_states = max_node_states
        cfg.num_edge_states = max_edge_states
        
        train_dataloaders[cfg.algorithm] = create_dataloader(cfg, "train", seed=seed, device=device)
        val_dataloaders[cfg.algorithm] = create_dataloader(cfg, "val", seed=seed + 1, device=device)
        test_dataloaders[cfg.algorithm] = create_dataloader(cfg, "test", seed=seed + 2, device=device)

    writer = create_logger(unified_config, model_name, session.get_wandb_run_id() if session else None)
    model.train()
    
    # Create iterators for interleaved training
    train_iterators = {alg: iter(dl) for alg, dl in train_dataloaders.items()}
    
    # Total steps = num_iterations * num_algorithms
    total_steps = unified_config.num_iterations * len(algorithms)
    
    # Checkpoint intervals
    checkpoint_interval = total_steps * unified_config.checkpoint_interval if unified_config.checkpoint_interval > 0 else 0
    next_checkpoint = checkpoint_interval
    checkpoint_dir = Path(unified_config.out_directory) / f"checkpoints_{model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    #Resume from checkpoint if session provided
    if session and session.get_resume_step() > 0:
        resume_step = session.get_resume_step()
        steps_per_algorithm = session.get_steps_per_algorithm()
        last_checkpoint = session.get_last_checkpoint()
        
        if last_checkpoint and Path(last_checkpoint).exists():
            print(f"Loading checkpoint from {last_checkpoint}")
            checkpoint = torch.load(last_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore multitask registry state
            if 'multitask_state' in checkpoint:
                registry = model._multitask_registry
                multitask_state = checkpoint['multitask_state']
                registry.algorithm_dict = multitask_state['algorithm_dict'].copy()
                print(f"Restored multitask state with algorithms: {list(registry.algorithm_dict.keys())}")
        
        print(f"Resuming from step {resume_step}")
        print(f"Steps per algorithm: {steps_per_algorithm}")
        
        steps = resume_step
        
        # Fast-forward checkpoint counter
        if checkpoint_interval > 0:
            next_checkpoint = ((steps // checkpoint_interval) + 1) * checkpoint_interval
        
        # Increment restart counter
        session.increment_restart_count()
    else:
        steps = 0
        steps_per_algorithm = {alg: 0 for alg in algorithms}
    
    print(f"\nStarting multitask training:")
    print(f"  - {len(algorithms)} algorithms: {algorithms}")
    print(f"  - {unified_config.num_iterations} iterations per algorithm")
    print(f"  - {total_steps} total steps (interleaved)")
    if checkpoint_interval > 0:
        print(f"  - Checkpoints every {checkpoint_interval} steps")
    if session:
        print(f"  - Max runtime: {session.max_runtime_seconds}s")
        print(f"  - Restarts so far: {session.state.get('restarts', 0)}")
    
    training_interrupted = False
    early_stopped = False
    while steps < total_steps:
        #get the minimum steps across algorithms to have all algorithms aligned
        # 1. Determine the "floor" progress across all algorithms
        min_steps = min(steps_per_algorithm.values())
        
        round_metrics_buffer = {}

        for algorithm in algorithms:
            if steps >= total_steps: break
            
            # 3. SKIP AHEAD LOGIC
            # If this algorithm is ahead of the pack (from a previous partial run),
            # skip it until others catch up.
            if steps_per_algorithm[algorithm] > min_steps:
                continue
            
            #Check timeout
            if session and session.should_stop():
                print(f"\nReaching time limit, saving checkpoint at step {steps}...")
                checkpoint_path = checkpoint_dir / f"model_step_{steps}_interrupt.pt"
                
                # Save complete checkpoint including multitask state
                checkpoint = {
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'multitask_state': {
                        'num_algorithms': model._multitask_registry.num_algorithms,
                        'algorithm_dict': model._multitask_registry.algorithm_dict.copy(),
                    }
                }
                torch.save(checkpoint, checkpoint_path)
                
                session.save_state(steps, str(checkpoint_path), steps_per_algorithm)
                training_interrupted = True
                break
                
            # Get next batch for this algorithm (reset iterator if exhausted)
            try:
                batch = next(train_iterators[algorithm])
            except StopIteration:
                train_iterators[algorithm] = iter(train_dataloaders[algorithm])
                batch = next(train_iterators[algorithm])
            
            batch = batch.to(device) #if data was cached

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
            
            # buffer loss per algorithm
            round_metrics_buffer[f"{algorithm}/Loss/train"] = loss.detach().item()
            # Evaluate periodically
            if steps_per_algorithm[algorithm] % unified_config.eval_each == 1:
                val_scores, test_scores = evaluate(
                    model,
                    val_dataloaders[algorithm],
                    test_dataloaders[algorithm],
                    utils.METRICS[unified_config.output_type],
                    model_saver,
                    None, #writer is None for multitask as we buffer here
                    steps,
                    algorithm=algorithm,
                )
                # Buffer the scores
                for k, v in val_scores.items():
                    round_metrics_buffer[f"{algorithm}/{k}/val"] = v
                for k, v in test_scores.items():
                    round_metrics_buffer[f"{algorithm}/{k}/test"] = v
            
            # Progress logging
            if steps % (100 * len(algorithms)) == 0:
                alg_progress = ", ".join([f"{a}: {steps_per_algorithm[a]}" for a in algorithms])
                elapsed = session.elapsed_time() if session else 0
                print(f"Step {steps}/{total_steps} | Per-algorithm: {alg_progress} | Time: {elapsed:.1f}s")
            
            # Checkpoint evaluation
            if checkpoint_interval > 0 and steps >= next_checkpoint and steps < total_steps:
                progress_pct = int(100 * steps / total_steps)
                print(f"\n{'='*60}")
                print(f"Checkpoint at {progress_pct}% ({steps}/{total_steps} steps)")
                print(f"{'='*60}")
                
                # CHANGED: Save model checkpoint with multitask state
                checkpoint_path = checkpoint_dir / f"model_step_{steps}.pt"
                checkpoint = {
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'multitask_state': {
                        'num_algorithms': model._multitask_registry.num_algorithms,
                        'algorithm_dict': model._multitask_registry.algorithm_dict.copy(),
                    }
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Model saved to: {checkpoint_path}")
                
                # Save session state
                if session:
                    session.save_state(steps, str(checkpoint_path), steps_per_algorithm)
                
                # Detailed evaluation for each algorithm
                with torch.no_grad():
                    model.eval()
                    for alg in algorithms:
                        val_output_path = checkpoint_dir / f"eval_{alg}_val_step_{steps}.json"
                        utils.evaluate_print(
                            model, 
                            val_dataloaders[alg], 
                            utils.METRICS[unified_config.output_type],
                            output_path=str(val_output_path),
                            algorithm=alg,
                            step=steps,
                            split="val",
                            print_results=False,
                        )
                        
                        test_output_path = checkpoint_dir / f"eval_{alg}_test_step_{steps}.json"
                        utils.evaluate_print(
                            model,
                            test_dataloaders[alg],
                            utils.METRICS[unified_config.output_type],
                            output_path=str(test_output_path),
                            algorithm=alg,
                            step=steps,
                            split="test",
                            print_results=False,
                        )
                    model.train()
                
                next_checkpoint += checkpoint_interval
                print(f"{'='*60}\n")

        #write buffered metrics for this round
        if writer is not None:
            writer.log_dict(round_metrics_buffer, step=min_steps + 1)

        # --- EARLY STOPPING CHECK ---
        # Only check if we actually evaluated this round (look for keys in buffer)
        if session and any("/val" in k for k in round_metrics_buffer):
            # Determine metric name based on output type
            metric_name = "pointer_accuracy_graph_level" if unified_config.output_type == "pointer" else "node_mask_accuracy_graph_level"
            
            # Extract scores for this specific metric across all algorithms
            # Key format: "{algorithm}/{metric_name}/val"
            val_accs = []
            for k, v in round_metrics_buffer.items():
                if k.endswith(f"/{metric_name}/val"):
                    val_accs.append(v)
            
            if val_accs:
                # Average across algorithms
                avg_val_acc = sum(val_accs) / len(val_accs)
                
                if session.check_early_stopping(avg_val_acc):
                    print(f"\nEarly stopping triggered! (Avg Val Acc: {avg_val_acc:.4f})")
                    training_interrupted = True
                    early_stopped = True
                    break
                
        #Break if interrupted
        if training_interrupted:
            break
    
    #Handle interruption
    if training_interrupted:
        if not early_stopped:
            print(f"\nTraining interrupted at step {steps}. Restart with --restart to continue.")
            #don't close as this will finish the runs
            # if writer is not None:
            #     writer.close()
            return None
        #else: if early stopped continue to final evaluation

    # Final evaluation on all algorithms (rest remains mostly the same but save with multitask state)
    print("\n" + "=" * 60)
    print("Final Evaluation (100%)")
    print("=" * 60)
    print(f"Steps per algorithm: {steps_per_algorithm}")
    
    # CHANGED: Save final model with multitask state
    final_model_path = checkpoint_dir / "model_final.pt"
    checkpoint = {
        'step': total_steps,
        'model_state_dict': model.state_dict(),
        'multitask_state': {
            'num_algorithms': model._multitask_registry.num_algorithms,
            'algorithm_dict': model._multitask_registry.algorithm_dict.copy(),
        }
    }
    torch.save(checkpoint, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # ... rest of evaluation code remains the same ...
    
    model.eval()
    with torch.no_grad():
        for algorithm in algorithms:
            evaluate(
                model,
                val_dataloaders[algorithm],
                test_dataloaders[algorithm],
                utils.METRICS[unified_config.output_type],
                model_saver,
                writer,
                steps,
                algorithm=algorithm,
            )
            
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
    
    _save_training_summary(checkpoint_dir, algorithms, total_steps, steps_per_algorithm)
    
    #Mark session as completed
    if session:
        session.mark_completed(str(final_model_path))
        session.cleanup()
    
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
    if len(args)==3:
        options, seed, gpu_id = args
        config_path = options.config_path
        multitask = options.multitask
        max_runtime_minutes = options.max_runtime_minutes
        patience = options.patience
    else:
        raise ValueError("Invalid number of arguments for train_worker")
    
    # --- Device setup ---
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # --- Reproducibility ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- Algorithms (identical to sequential) ---
    algorithms = []
    if multitask:
        if "," in config_path:
            config_paths = [p.strip() for p in config_path.split(",")]
            configs = [base_config.read_config(path) for path in config_paths]
        else:
            base_cfg = base_config.read_config(config_path)
            if not base_cfg.multitask_algorithms:
                raise ValueError("Config must have multitask_algorithms set")
            configs = configs_from_multitask_config(base_cfg)

        algorithms = [cfg.algorithm for cfg in configs]

    # --- Session (identical to sequential) ---
    session = TrainingSession(
        config_path,
        seed,
        max_runtime_seconds=max_runtime_minutes * 60,
        multitask=multitask,
        algorithms=algorithms,
        patience=patience,
    )

    # --- Training ---
    if multitask:
        model = train_multitask(configs, seed, session=session, gpu_id=gpu_id)
    else:
        print(f"Train with config {config_path}")
        config = base_config.read_config(config_path)
        model = train(config, seed, session=session, gpu_id=gpu_id)
    return {
        "seed": seed,
        "gpu": gpu_id,
        "status": "done"
    }


if __name__ == "__main__":
    # WICHTIG: Muss für torch.multiprocessing gesetzt werden
    mp.set_start_method('spawn', force=True)
    
    torch.set_num_threads(5)
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/bfs.yaml",
                        help="Path to config file. For multitask, can be: "
                             "1) A single config with 'multitask_algorithms' list, or "
                             "2) Comma-separated paths to multiple config files")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of random seeds to train, If set on restart this many seeds will be checked for incomplete jobs.")
    parser.add_argument("--multitask", action="store_true",
                        help="Enable multitask training")
    parser.add_argument("--parallel", action="store_true",
                        help="Train seeds in parallel")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: num_seeds)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2'). If not set, uses CUDA_VISIBLE_DEVICES or all GPUs")
    parser.add_argument("--restart", action="store_true",
                        help="Restart incomplete training jobs")
    parser.add_argument("--list_jobs", action="store_true",
                        help="List all incomplete training jobs")
    parser.add_argument("--clean_jobs", action="store_true",
                        help="Clean up all incomplete training job state files")
    parser.add_argument("--clean_job", type=str, default=None,
                        help="Clean up specific job by session ID (e.g., 'bfs_seed40')")
    parser.add_argument("--max_runtime_minutes", type=int, default=23,
                        help="Maximum runtime per training session in minutes (default: 23)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience in number of evaluations (default: 5)")

    options = parser.parse_args()
    
    # Handle job listing
    if options.list_jobs:
        manager = RestartManager()
        manager.list_jobs()
        exit()

    # Handle job cleaning
    if options.clean_jobs:
        manager = RestartManager()
        manager.clean_all_jobs()
        exit()
    
    if options.clean_job:
        manager = RestartManager()
        manager.clean_job(options.clean_job)
        exit()
    
    # Handle restart mode
    if options.restart:
        if options.parallel:
            num_workers = options.num_workers if options.num_workers else options.num_seeds
            print(f"\nParallel restart of incomplete jobs with {num_workers} workers...")
            num_jobs = options.num_seeds

            manager = RestartManager()
            restart_filter = options.config_path if options.config_path else None
            jobs = manager.get_next_jobs(num_jobs=num_jobs, filter=restart_filter)

            if not jobs:
                print("No incomplete jobs found.")
                exit()

            if len(jobs) < num_jobs:
                print(f"Only found {len(jobs)} incomplete jobs, fewer than requested {num_jobs}. Running Anyways.")
            if len(jobs) < num_workers:
                num_workers = len(jobs)
                print(f"Reducing number of workers to {num_workers}.")

            #get gpu odering
            if options.gpus:
                gpu_ids = [int(x.strip()) for x in options.gpus.split(',')]
            else:
                try:
                    gpu_ids = get_least_used_gpus()
                except Exception as e:
                    if 'CUDA_VISIBLE_DEVICES' in os.environ:
                        gpu_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x]
                    else:
                        gpu_ids = list(range(torch.cuda.device_count()))
            if not gpu_ids:
                raise ValueError("No GPUs available for parallel training")
            print(f"\nUsing GPUs: {gpu_ids[:min(num_workers, len(gpu_ids))]}")

            worker_args = []
            for i, job in enumerate(jobs):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                #get the session parameters and prepare the options object
                config_path = job['config_path']
                multitask = job.get('multitask', False)
                worker_args.append((
                    argparse.Namespace(
                        config_path=config_path,
                        num_seeds=1,
                        multitask=multitask,
                        parallel=False,
                        num_workers=1,
                        gpus=str(gpu_id),
                        restart=False,
                        list_jobs=False,
                        clean_jobs=False,
                        clean_job=None,
                        max_runtime_minutes=options.max_runtime_minutes,
                        patience=job.get('patience', options.patience)
                    ),                    
                    job['seed'],
                    gpu_id
                ))
                algorithms = job.get('algorithms', [])
                
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(train_worker, worker_args)

            print(f"\nAll restarted jobs completed: {results}")

        else:
            manager = RestartManager()
            job = manager.get_next_job()
            
            if job is None:
                print("No incomplete jobs found.")
                exit()
            
            print(f"\nRestarting incomplete job:")
            print(f"  Config: {job['config_path']}")
            print(f"  Seed: {job['seed']}")
            print(f"  Current step: {job['current_step']}")
            print(f"  Restarts: {job.get('restarts', 0)}")
            
            #Get algorithms for multitask
            algorithms = job.get('algorithms', [])
            
            # Create session and train
            session = TrainingSession(
                job['config_path'],
                job['seed'],
                max_runtime_seconds=options.max_runtime_minutes * 60,
                multitask=job.get('multitask', False),
                algorithms=algorithms,
                patience=options.patience
            )
            
            np.random.seed(job['seed'])
            torch.manual_seed(job['seed'])
            
            if job.get('multitask', False):
                # Multitask restart
                if "," in job['config_path']:
                    config_paths = [p.strip() for p in job['config_path'].split(",")]
                    configs = [base_config.read_config(path) for path in config_paths]
                else:
                    base_cfg = base_config.read_config(job['config_path'])
                    if not base_cfg.multitask_algorithms:
                        raise ValueError("Config must have multitask_algorithms set")
                    configs = configs_from_multitask_config(base_cfg)
                
                model = train_multitask(configs, job['seed'], session=session)
            else:
                config = base_config.read_config(job['config_path'])
                model = train(config, job['seed'], session=session)
            
            exit()
    
    seeds = list(range(40, 40 + options.num_seeds))
    
    if options.parallel:
        # Bestimme verfügbare GPUs
        if options.gpus:
            gpu_ids = [int(x.strip()) for x in options.gpus.split(',')]
        else:
            try:
                gpu_ids = get_least_used_gpus()
            except Exception as e:
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    gpu_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x]
                else:
                    gpu_ids = list(range(torch.cuda.device_count()))
        
        if not gpu_ids:
            raise ValueError("No GPUs available for parallel training")
        
        num_workers = min(len(seeds), len(gpu_ids))
        print("\nParallel training:")
        print(f"  Workers: {num_workers}")
        print(f"  Seeds: {seeds}")
        print(f"  GPUs: {gpu_ids}")

        worker_args = [
            (options, seed, gpu_ids[i % len(gpu_ids)])
            for i, seed in enumerate(seeds)
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(train_worker, worker_args)

        print(f"\nAll seeds completed: {results}")
    
    else:
        # Sequential training
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            #Get algorithms list for session
            algorithms = []
            if options.multitask:
                if "," in options.config_path:
                    config_paths = [p.strip() for p in options.config_path.split(",")]
                    configs = [base_config.read_config(path) for path in config_paths]
                else:
                    base_cfg = base_config.read_config(options.config_path)
                    if not base_cfg.multitask_algorithms:
                        raise ValueError("Config must have multitask_algorithms set")
                    configs = configs_from_multitask_config(base_cfg)
                algorithms = [cfg.algorithm for cfg in configs]
            
            # Create session for restart management
            session = TrainingSession(
                options.config_path,
                seed,
                max_runtime_seconds=options.max_runtime_minutes * 60,
                multitask=options.multitask,
                algorithms=algorithms,
                patience=options.patience,
            )

            if options.multitask:
                model = train_multitask(configs, seed, session=session)
            else:
                print("Train with config {}".format(options.config_path))
                config = base_config.read_config(options.config_path)
                model = train(config, seed, session=session)
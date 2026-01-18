"""
Training functions for PPO-LLM Strategy Shaping experiments.

Provides callbacks, training loops, and parallel execution utilities.
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from .config import Config
from .env_wrappers import make_train_env, warmup_mlam
from .utils import (
    set_global_seed,
    ensure_dir,
    get_run_dir,
    get_checkpoint_dir,
    get_model_path,
    format_time,
)


class StopTrainingOnMaxSteps(BaseCallback):
    """
    Callback to stop training after a specified number of timesteps.
    
    More reliable than n_timesteps parameter when using early stopping.
    """
    
    def __init__(self, max_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_steps = max_steps
    
    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_steps


class PBTCallback(BaseCallback):
    """
    Population-Based Training (PBT) callback for hyperparameter evolution.
    
    Periodically evaluates the current policy and potentially mutates
    hyperparameters based on population statistics.
    """
    
    def __init__(
        self,
        eval_freq: int = 50_000,
        perturb_prob: float = 0.2,
        perturb_factor: float = 1.2,
        verbose: int = 0
    ):
        """
        Args:
            eval_freq: Steps between evaluations
            perturb_prob: Probability of perturbing each hyperparameter
            perturb_factor: Multiplicative factor for perturbation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.perturb_prob = perturb_prob
        self.perturb_factor = perturb_factor
        self.last_eval_step = 0
        self.best_reward = float("-inf")
        self.rewards_history: List[float] = []
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_and_maybe_perturb()
            self.last_eval_step = self.num_timesteps
        return True
    
    def _evaluate_and_maybe_perturb(self) -> None:
        """Evaluate current performance and potentially mutate hyperparameters."""
        # Get recent episode rewards
        if len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            mean_reward = np.mean(recent_rewards)
            self.rewards_history.append(mean_reward)
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: mean_reward={mean_reward:.2f}")
            
            # Update best and potentially perturb if underperforming
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
            elif np.random.random() < self.perturb_prob:
                self._perturb_hyperparameters()
    
    def _perturb_hyperparameters(self) -> None:
        """Randomly perturb learning hyperparameters."""
        if np.random.random() < 0.5:
            # Perturb learning rate
            old_lr = self.model.learning_rate
            factor = self.perturb_factor if np.random.random() < 0.5 else 1/self.perturb_factor
            if callable(old_lr):
                # Learning rate schedule - skip
                return
            new_lr = np.clip(old_lr * factor, 1e-6, 1e-2)
            self.model.learning_rate = new_lr
            self.model.lr_schedule = lambda _: new_lr
            
            if self.verbose > 0:
                print(f"  Perturbed lr: {old_lr:.2e} -> {new_lr:.2e}")


def train_one_run(
    config: Config,
    baseline: str,
    env_name: str,
    seed: int,
    verbose: int = 0,
    save_checkpoints: bool = True,
    checkpoint_freq: int = 100_000,
) -> Tuple[PPO, str, float]:
    """
    Train a single PPO agent for one configuration.
    
    Args:
        config: Experiment configuration
        baseline: Baseline name (e.g., "PPO+LLM", "CC_PPO")
        env_name: Environment name (e.g., "No Noise", "Delay")
        seed: Random seed
        verbose: Verbosity level
        save_checkpoints: Whether to save intermediate checkpoints
        checkpoint_freq: Steps between checkpoints
        
    Returns:
        Tuple of (trained model, run directory path, training time in seconds)
    """
    # Setup
    set_global_seed(seed)
    run_dir = get_run_dir(config.runs_root, baseline, env_name, seed)
    ensure_dir(run_dir)
    
    # Create environment
    env = make_train_env(config, baseline, env_name, seed)
    
    # Determine training steps
    total_steps = config.baseline_steps.get(baseline, config.llm_total_steps)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.lr,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=verbose,
        seed=seed,
    )
    
    # Setup callbacks
    callbacks = [StopTrainingOnMaxSteps(total_steps, verbose=verbose)]
    
    if save_checkpoints:
        ckpt_dir = get_checkpoint_dir(run_dir)
        ensure_dir(ckpt_dir)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=ckpt_dir,
                name_prefix="ppo",
            )
        )
    
    if baseline == "PBT_PPO":
        callbacks.append(PBTCallback(verbose=verbose))
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_steps, callback=callbacks)
    training_time = time.time() - start_time
    
    # Save final model
    model_path = get_model_path(run_dir)
    model.save(model_path)
    
    if verbose > 0:
        print(f"Completed {baseline}/{env_name}/seed_{seed} in {format_time(training_time)}")
    
    return model, run_dir, training_time


def train_all(
    config: Config,
    baselines: Optional[List[str]] = None,
    env_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_jobs: int = 1,
    verbose: int = 0,
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
) -> Dict[Tuple[str, str, int], Tuple[PPO, str, float]]:
    """
    Train all configurations (baselines × environments × seeds).
    
    Args:
        config: Experiment configuration
        baselines: List of baselines to train (default: all)
        env_names: List of environments to train (default: all)
        seeds: List of seeds to use (default: from config)
        n_jobs: Number of parallel jobs (1 = sequential)
        verbose: Verbosity level
        progress_callback: Optional callback(baseline, env, seed) called after each run
        
    Returns:
        Dictionary mapping (baseline, env, seed) to (model, run_dir, time)
    """
    from joblib import Parallel, delayed
    
    # Defaults
    if baselines is None:
        baselines = list(config.baseline_steps.keys())
    if env_names is None:
        env_names = config.all_envs
    if seeds is None:
        seeds = config.seeds
    
    # Warmup MLAM once before parallel training
    warmup_mlam(config.layout, config.horizon)
    
    # Build task list
    tasks = [
        (baseline, env_name, seed)
        for baseline in baselines
        for env_name in env_names
        for seed in seeds
    ]
    
    if verbose > 0:
        print(f"Training {len(tasks)} configurations with {n_jobs} workers")
    
    def run_task(baseline: str, env_name: str, seed: int):
        result = train_one_run(config, baseline, env_name, seed, verbose=verbose)
        if progress_callback:
            progress_callback(baseline, env_name, seed)
        return (baseline, env_name, seed), result
    
    if n_jobs == 1:
        # Sequential execution
        results = {}
        for baseline, env_name, seed in tasks:
            key, value = run_task(baseline, env_name, seed)
            results[key] = value
    else:
        # Parallel execution
        parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_task)(baseline, env_name, seed)
            for baseline, env_name, seed in tasks
        )
        results = dict(parallel_results)
    
    return results


def load_or_train(
    config: Config,
    baseline: str,
    env_name: str,
    seed: int,
    verbose: int = 0,
) -> Tuple[PPO, str]:
    """
    Load a trained model if it exists, otherwise train it.
    
    Args:
        config: Experiment configuration
        baseline: Baseline name
        env_name: Environment name
        seed: Random seed
        verbose: Verbosity level
        
    Returns:
        Tuple of (model, run directory path)
    """
    from .env_wrappers import make_env
    
    run_dir = get_run_dir(config.runs_root, baseline, env_name, seed)
    model_path = get_model_path(run_dir)
    
    if os.path.exists(model_path):
        if verbose > 0:
            print(f"Loading existing model from {model_path}")
        env = make_env(config, baseline, env_name, seed)
        model = PPO.load(model_path, env=env)
        return model, run_dir
    else:
        if verbose > 0:
            print(f"Training new model for {baseline}/{env_name}/seed_{seed}")
        model, run_dir, _ = train_one_run(config, baseline, env_name, seed, verbose=verbose)
        return model, run_dir

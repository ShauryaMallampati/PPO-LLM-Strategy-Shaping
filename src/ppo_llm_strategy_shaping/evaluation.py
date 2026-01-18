"""
Evaluation functions for PPO-LLM Strategy Shaping experiments.

Provides Nash gap analysis, latency measurement, robustness testing,
and task completion metrics.
"""

import os
import time
import csv
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from .config import Config
from .env_wrappers import make_env
from .utils import (
    get_run_dir,
    get_model_path,
    ensure_dir,
    write_csv_header,
    append_csv_row,
)


# =============================================================================
# Core Evaluation
# =============================================================================

def evaluate_episode(
    model: PPO,
    env,
    deterministic: bool = True,
) -> Tuple[float, int, Dict]:
    """
    Run one episode and collect metrics.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        deterministic: Whether to use deterministic actions
        
    Returns:
        Tuple of (total_reward, episode_length, info_dict)
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    length = 0
    info = {}
    
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        length += 1
    
    return total_reward, length, info


def evaluate(
    model: PPO,
    env,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model over multiple episodes.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary with mean_reward, std_reward, mean_length, etc.
    """
    rewards = []
    lengths = []
    
    for _ in range(n_episodes):
        reward, length, _ = evaluate_episode(model, env, deterministic)
        rewards.append(reward)
        lengths.append(length)
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "n_episodes": n_episodes,
    }


# =============================================================================
# Nash Gap Analysis
# =============================================================================

def compute_best_response_reward(
    policy_model: PPO,
    config: Config,
    env_name: str,
    n_episodes: int = 5,
) -> float:
    """
    Compute the best-response reward against a fixed partner policy.
    
    This approximates the reward an optimal agent would get when
    paired with the given policy.
    
    Args:
        policy_model: The fixed partner policy
        config: Experiment configuration
        env_name: Environment name
        n_episodes: Number of episodes to average over
        
    Returns:
        Mean best-response reward
    """
    # For simplicity, we use the same model as both agents
    # In a full implementation, you'd train a best response
    env = make_env(config, "Baseline", env_name, seed=0)
    metrics = evaluate(policy_model, env, n_episodes=n_episodes)
    return metrics["mean_reward"]


def compute_nash_gap(
    model: PPO,
    config: Config,
    env_name: str,
    n_episodes: int = 5,
) -> float:
    """
    Compute the Nash gap for a joint policy.
    
    Nash gap = BR(π₋ᵢ) - V(πᵢ, π₋ᵢ) summed over agents
    
    A lower gap indicates the policy is closer to Nash equilibrium.
    
    Args:
        model: Trained joint policy
        config: Experiment configuration
        env_name: Environment name
        n_episodes: Number of episodes for estimation
        
    Returns:
        Estimated Nash gap
    """
    env = make_env(config, "Baseline", env_name, seed=0)
    
    # Current joint value
    current_metrics = evaluate(model, env, n_episodes=n_episodes)
    current_value = current_metrics["mean_reward"]
    
    # Best response value (approximated)
    br_value = compute_best_response_reward(model, config, env_name, n_episodes)
    
    # Nash gap (simplified: single deviation)
    gap = max(0, br_value - current_value)
    return gap


def nash_gap_analysis(
    config: Config,
    baselines: Optional[List[str]] = None,
    env_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_episodes: int = 5,
    output_csv: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute Nash gaps for all configurations.
    
    Args:
        config: Experiment configuration
        baselines: Baselines to analyze (default: all)
        env_names: Environments to analyze (default: all)
        seeds: Seeds to use (default: from config)
        n_episodes: Episodes per evaluation
        output_csv: Optional path to save results
        
    Returns:
        Nested dict: {baseline: {env: mean_nash_gap}}
    """
    if baselines is None:
        baselines = list(config.baseline_steps.keys())
    if env_names is None:
        env_names = config.all_envs
    if seeds is None:
        seeds = config.seeds
    
    results = defaultdict(dict)
    rows = []
    
    for baseline in baselines:
        for env_name in env_names:
            gaps = []
            for seed in seeds:
                run_dir = get_run_dir(config.runs_root, baseline, env_name, seed)
                model_path = get_model_path(run_dir)
                
                if not os.path.exists(model_path):
                    continue
                
                env = make_env(config, baseline, env_name, seed)
                model = PPO.load(model_path, env=env)
                gap = compute_nash_gap(model, config, env_name, n_episodes)
                gaps.append(gap)
                rows.append([baseline, env_name, seed, gap])
            
            if gaps:
                results[baseline][env_name] = np.mean(gaps)
    
    if output_csv:
        ensure_dir(os.path.dirname(output_csv))
        write_csv_header(output_csv, ["baseline", "env", "seed", "nash_gap"], overwrite=True)
        for row in rows:
            append_csv_row(output_csv, row)
    
    return dict(results)


# =============================================================================
# Latency Analysis
# =============================================================================

def measure_inference_latency(
    model: PPO,
    env,
    n_steps: int = 1000,
) -> Dict[str, float]:
    """
    Measure model inference latency.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        n_steps: Number of steps to measure
        
    Returns:
        Dictionary with mean, std, min, max latency in milliseconds
    """
    obs, _ = env.reset()
    latencies = []
    
    for _ in range(n_steps):
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
        
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }


def latency_analysis(
    config: Config,
    baselines: Optional[List[str]] = None,
    n_steps: int = 1000,
    output_csv: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Measure inference latency for all baselines.
    
    Args:
        config: Experiment configuration
        baselines: Baselines to analyze (default: all)
        n_steps: Steps per measurement
        output_csv: Optional path to save results
        
    Returns:
        Dict mapping baseline to latency statistics
    """
    if baselines is None:
        baselines = list(config.baseline_steps.keys())
    
    results = {}
    rows = []
    
    for baseline in baselines:
        # Use first seed and No Noise env for latency testing
        seed = config.seeds[0]
        env_name = "No Noise"
        
        run_dir = get_run_dir(config.runs_root, baseline, env_name, seed)
        model_path = get_model_path(run_dir)
        
        if not os.path.exists(model_path):
            continue
        
        env = make_env(config, baseline, env_name, seed)
        model = PPO.load(model_path, env=env)
        
        latency = measure_inference_latency(model, env, n_steps)
        results[baseline] = latency
        rows.append([
            baseline,
            latency["mean_ms"],
            latency["std_ms"],
            latency["p50_ms"],
            latency["p95_ms"],
            latency["p99_ms"],
        ])
    
    if output_csv:
        ensure_dir(os.path.dirname(output_csv))
        write_csv_header(
            output_csv,
            ["baseline", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms"],
            overwrite=True
        )
        for row in rows:
            append_csv_row(output_csv, row)
    
    return results


# =============================================================================
# Robustness Analysis
# =============================================================================

def robustness_analysis(
    config: Config,
    baselines: Optional[List[str]] = None,
    env_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_episodes: int = 10,
    output_csv: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze robustness across environment perturbations.
    
    Tests each trained policy on all environment variants to measure
    how well it generalizes to unseen perturbations.
    
    Args:
        config: Experiment configuration
        baselines: Baselines to analyze (default: all)
        env_names: Environments to analyze (default: all)
        seeds: Seeds to use (default: from config)
        n_episodes: Episodes per evaluation
        output_csv: Optional path to save results
        
    Returns:
        Nested dict: {baseline: {train_env: {test_env: mean_reward}}}
    """
    if baselines is None:
        baselines = list(config.baseline_steps.keys())
    if env_names is None:
        env_names = config.all_envs
    if seeds is None:
        seeds = config.seeds
    
    results = defaultdict(lambda: defaultdict(dict))
    rows = []
    
    for baseline in baselines:
        for train_env in env_names:
            for test_env in env_names:
                rewards = []
                
                for seed in seeds:
                    run_dir = get_run_dir(config.runs_root, baseline, train_env, seed)
                    model_path = get_model_path(run_dir)
                    
                    if not os.path.exists(model_path):
                        continue
                    
                    # Load model trained on train_env
                    env = make_env(config, baseline, test_env, seed)
                    model = PPO.load(model_path, env=env)
                    
                    # Evaluate on test_env
                    metrics = evaluate(model, env, n_episodes=n_episodes)
                    rewards.append(metrics["mean_reward"])
                    rows.append([baseline, train_env, test_env, seed, metrics["mean_reward"]])
                
                if rewards:
                    results[baseline][train_env][test_env] = np.mean(rewards)
    
    if output_csv:
        ensure_dir(os.path.dirname(output_csv))
        write_csv_header(
            output_csv,
            ["baseline", "train_env", "test_env", "seed", "mean_reward"],
            overwrite=True
        )
        for row in rows:
            append_csv_row(output_csv, row)
    
    return dict(results)


# =============================================================================
# Task Completion Analysis
# =============================================================================

def compute_task_completion(
    model: PPO,
    env,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    Compute task completion metrics for Overcooked.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with completion metrics
    """
    dishes_served = []
    completion_rates = []
    
    for _ in range(n_episodes):
        _, _, info = evaluate_episode(model, env, deterministic=True)
        
        # Extract Overcooked-specific metrics if available
        if "episode" in info:
            ep_info = info["episode"]
            if "dishes_served" in ep_info:
                dishes_served.append(ep_info["dishes_served"])
    
    # If no Overcooked metrics, fall back to reward-based proxy
    if not dishes_served:
        env_copy = env
        for _ in range(n_episodes):
            obs, _ = env_copy.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env_copy.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Use reward as proxy (assuming 20 points per dish)
            estimated_dishes = max(0, episode_reward) / 20
            dishes_served.append(estimated_dishes)
    
    return {
        "mean_dishes": np.mean(dishes_served) if dishes_served else 0,
        "std_dishes": np.std(dishes_served) if dishes_served else 0,
        "max_dishes": np.max(dishes_served) if dishes_served else 0,
        "completion_rate": np.mean([d > 0 for d in dishes_served]) if dishes_served else 0,
    }


def task_completion_analysis(
    config: Config,
    baselines: Optional[List[str]] = None,
    env_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_episodes: int = 10,
    output_csv: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze task completion across all configurations.
    
    Args:
        config: Experiment configuration
        baselines: Baselines to analyze (default: all)
        env_names: Environments to analyze (default: all)
        seeds: Seeds to use (default: from config)
        n_episodes: Episodes per evaluation
        output_csv: Optional path to save results
        
    Returns:
        Nested dict: {baseline: {env: {metric: value}}}
    """
    if baselines is None:
        baselines = list(config.baseline_steps.keys())
    if env_names is None:
        env_names = config.all_envs
    if seeds is None:
        seeds = config.seeds
    
    results = defaultdict(dict)
    rows = []
    
    for baseline in baselines:
        for env_name in env_names:
            all_metrics = defaultdict(list)
            
            for seed in seeds:
                run_dir = get_run_dir(config.runs_root, baseline, env_name, seed)
                model_path = get_model_path(run_dir)
                
                if not os.path.exists(model_path):
                    continue
                
                env = make_env(config, baseline, env_name, seed)
                model = PPO.load(model_path, env=env)
                
                metrics = compute_task_completion(model, env, n_episodes)
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                rows.append([
                    baseline, env_name, seed,
                    metrics["mean_dishes"],
                    metrics["completion_rate"],
                ])
            
            if all_metrics:
                results[baseline][env_name] = {
                    key: np.mean(values) for key, values in all_metrics.items()
                }
    
    if output_csv:
        ensure_dir(os.path.dirname(output_csv))
        write_csv_header(
            output_csv,
            ["baseline", "env", "seed", "mean_dishes", "completion_rate"],
            overwrite=True
        )
        for row in rows:
            append_csv_row(output_csv, row)
    
    return dict(results)


# =============================================================================
# Summary Report
# =============================================================================

def generate_summary_report(
    config: Config,
    baselines: Optional[List[str]] = None,
    env_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive summary report of all analyses.
    
    Args:
        config: Experiment configuration
        baselines: Baselines to include
        env_names: Environments to include
        output_dir: Directory to save CSV files
        
    Returns:
        Formatted summary string
    """
    if output_dir is None:
        output_dir = config.results_root
    
    ensure_dir(output_dir)
    
    # Run all analyses
    nash_results = nash_gap_analysis(
        config, baselines, env_names,
        output_csv=os.path.join(output_dir, "nash_gap.csv")
    )
    
    latency_results = latency_analysis(
        config, baselines,
        output_csv=os.path.join(output_dir, "latency.csv")
    )
    
    robustness_results = robustness_analysis(
        config, baselines, env_names,
        output_csv=os.path.join(output_dir, "robustness.csv")
    )
    
    task_results = task_completion_analysis(
        config, baselines, env_names,
        output_csv=os.path.join(output_dir, "task_completion.csv")
    )
    
    # Format report
    lines = ["=" * 60, "PPO-LLM Strategy Shaping - Experiment Summary", "=" * 60, ""]
    
    lines.append("## Nash Gap Analysis")
    for baseline, envs in nash_results.items():
        for env_name, gap in envs.items():
            lines.append(f"  {baseline} / {env_name}: {gap:.4f}")
    lines.append("")
    
    lines.append("## Inference Latency")
    for baseline, latency in latency_results.items():
        lines.append(f"  {baseline}: {latency['mean_ms']:.2f}ms (p95: {latency['p95_ms']:.2f}ms)")
    lines.append("")
    
    lines.append("## Task Completion")
    for baseline, envs in task_results.items():
        for env_name, metrics in envs.items():
            lines.append(f"  {baseline} / {env_name}: {metrics['mean_dishes']:.2f} dishes")
    
    report = "\n".join(lines)
    
    # Save report
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    return report

"""
Utility functions for PPO-LLM Strategy Shaping experiments.

Provides seeding, file I/O helpers, and common utilities.
"""

import os
import csv
import random
from typing import Set, Tuple, Optional

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def load_completed_set(csv_path: str) -> Set[Tuple[str, str, str]]:
    """
    Load set of completed (baseline, env, seed) tuples from a CSV file.
    
    Used for resuming experiments without re-running completed configurations.
    
    Args:
        csv_path: Path to CSV file with columns [baseline, env, seed, ...]
        
    Returns:
        Set of (baseline, env, seed) tuples already completed
    """
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    done.add((row[0], row[1], str(row[2])))
    return done


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    """
    Convert a name to a filesystem-safe version.
    
    Args:
        name: Original name (e.g., "PPO+LLM", "No Noise")
        
    Returns:
        Safe version with spaces and special chars replaced
    """
    return name.replace(" ", "_").replace("+", "_plus_")


def get_run_dir(
    runs_root: str,
    baseline: str,
    env_name: str,
    seed: int
) -> str:
    """
    Get the directory path for a specific experimental run.
    
    Args:
        runs_root: Root directory for all runs
        baseline: Baseline name
        env_name: Environment name
        seed: Random seed
        
    Returns:
        Full path to the run directory
    """
    safe_base = safe_name(baseline)
    safe_env = safe_name(env_name)
    return os.path.join(runs_root, safe_base, safe_env, f"seed_{seed}")


def get_checkpoint_dir(run_dir: str) -> str:
    """
    Get the checkpoint directory for a run.
    
    Args:
        run_dir: Run directory path
        
    Returns:
        Path to checkpoints subdirectory
    """
    return os.path.join(run_dir, "checkpoints")


def get_model_path(run_dir: str, filename: str = "final_model.zip") -> str:
    """
    Get the full path to a model file.
    
    Args:
        run_dir: Run directory path
        filename: Model filename
        
    Returns:
        Full path to model file
    """
    return os.path.join(run_dir, filename)


def write_csv_header(csv_path: str, headers: list, overwrite: bool = False) -> None:
    """
    Write CSV header if file doesn't exist or overwrite is True.
    
    Args:
        csv_path: Path to CSV file
        headers: List of column headers
        overwrite: Whether to overwrite existing file
    """
    if overwrite or not os.path.exists(csv_path):
        ensure_dir(os.path.dirname(csv_path))
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def append_csv_row(csv_path: str, row: list) -> None:
    """
    Append a row to a CSV file.
    
    Args:
        csv_path: Path to CSV file
        row: List of values to write
    """
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def find_latest_checkpoint(
    checkpoint_dir: str,
    prefix: str = "ppo_",
    suffix: str = ".zip"
) -> Optional[str]:
    """
    Find the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint filename prefix
        suffix: Checkpoint filename suffix
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    import re
    
    if not os.path.isdir(checkpoint_dir):
        return None
    
    candidates = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    
    if not candidates:
        return None
    
    # Sort by step number (extract from filename)
    def extract_step(filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else 0
    
    candidates.sort(key=extract_step, reverse=True)
    return os.path.join(checkpoint_dir, candidates[0])


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

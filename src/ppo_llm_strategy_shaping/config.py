"""
Configuration module for PPO-LLM Strategy Shaping experiments.

Centralizes all hyperparameters and paths for reproducibility.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    """Configuration for PPO-LLM Strategy Shaping experiments."""
    
    layout: str = "asymmetric_advantages"
    horizon: int = 400
    num_actions: int = 6
    
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    seeds: List[int] = field(default_factory=lambda: [1001, 2002, 3003, 4004, 5005])
    checkpoint_every_steps: int = 50_000
    log_every_steps: int = 2_048
    eval_episodes: int = 10
    
    n_steps: int = 2048
    batch_size: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    llm_model_name: str = "EleutherAI/gpt-neo-1.3B"
    llm_bonus: float = 0.2
    
    noise_std: float = 0.01
    delay_noise_prob: float = 0.2
    delay_penalty: float = 0.5
    
    baseline_steps: Dict[str, int] = field(default_factory=lambda: {
        "Baseline": 1_000_000,
        "CC_PPO": 1_000_000,
        "SP_PPO": 1_000_000,
        "HARL": 1_000_000,
        "PPO+LLM": 600_000,
        "PBT_PPO": 1_000_000,
    })
    
    br_train_steps: int = 200_000
    br_eval_episodes: int = 20
    
    base_dir: str = "/content/drive/MyDrive"
    
    @property
    def runs_dir(self) -> str:
        return os.path.join(self.base_dir, "runs")
    
    @property
    def results_csv(self) -> str:
        return os.path.join(self.base_dir, "results_combined_new.csv")
    
    @property
    def br_results_csv(self) -> str:
        return os.path.join(self.base_dir, "br_nash_results.csv")
    
    @property
    def latency_csv(self) -> str:
        return os.path.join(self.base_dir, "latency_results.csv")
    
    @property
    def robustness_per_seed_csv(self) -> str:
        return os.path.join(self.base_dir, "robustness_deltas_per_seed.csv")
    
    @property
    def robustness_agg_csv(self) -> str:
        return os.path.join(self.base_dir, "robustness_deltas_agg.csv")
    
    @property
    def task_per_seed_csv(self) -> str:
        return os.path.join(self.base_dir, "task_completion_per_seed.csv")
    
    @property
    def task_agg_csv(self) -> str:
        return os.path.join(self.base_dir, "task_completion_agg.csv")

DEFAULT_CONFIG = Config()

BASELINES = ["Baseline", "PPO+LLM", "CC_PPO", "SP_PPO", "HARL", "PBT_PPO"]
ENV_NAMES = ["No Noise", "Noise", "Delay", "Combo"]

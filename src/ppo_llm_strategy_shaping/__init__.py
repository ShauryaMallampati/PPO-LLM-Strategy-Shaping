"""
PPO-LLM Strategy Shaping: LLM-guided reward shaping for multi-agent coordination.

This package provides tools for training PPO agents with LLM-based reward shaping
in the Overcooked cooperative cooking environment.
"""

from .config import Config
from .env_wrappers import (
    OCWrapper,
    OCWrapperNoise,
    OCWrapperDelay,
    OCWrapperCombo,
    OCWrapperLLM,
    HARL,
    make_env,
    make_train_env,
)
from .llm_shaping import get_llm, is_good
from .train import train_one_run, train_all
from .evaluation import (
    evaluate,
    compute_nash_gap,
    measure_latency,
    compute_robustness_deltas,
    compute_task_completion,
)
from .utils import set_global_seed, load_completed_set

__version__ = "1.0.0"
__all__ = [
    "Config",
    "OCWrapper",
    "OCWrapperNoise", 
    "OCWrapperDelay",
    "OCWrapperCombo",
    "OCWrapperLLM",
    "HARL",
    "make_env",
    "make_train_env",
    "get_llm",
    "is_good",
    "train_one_run",
    "train_all",
    "evaluate",
    "compute_nash_gap",
    "measure_latency",
    "compute_robustness_deltas",
    "compute_task_completion",
    "set_global_seed",
    "load_completed_set",
]

"""
Environment wrappers for Overcooked multi-agent coordination experiments.

This module provides Gymnasium-compatible wrappers for the Overcooked environment
with various perturbation regimes (noise, delay, combo) and reward shaping variants.
"""

import random
import numpy as np
import gymnasium as gym
import torch
from typing import Optional, Tuple, Dict, Any

from stable_baselines3.common.monitor import Monitor
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

from .config import DEFAULT_CONFIG

# Number of actions in Overcooked
NUM_ACTIONS = len(Action.ALL_ACTIONS)


class OCWrapper(gym.Env):
    """
    True 2-agent Overcooked wrapper.
    
    This wrapper provides:
    - observation: global featurized state (flattened)
    - action_space: MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS]) for joint actions
    - reward: shared team reward
    
    Args:
        layout: Name of the Overcooked layout to use
        horizon: Maximum episode length
    """
    metadata = {"render.modes": []}

    def __init__(
        self, 
        layout: str = DEFAULT_CONFIG.layout,
        horizon: int = DEFAULT_CONFIG.horizon
    ):
        super().__init__()
        self.layout = layout
        self.horizon = horizon
        
        mdp = OvercookedGridworld.from_layout_name(layout)
        self.oc = OvercookedEnv.from_mdp(mdp, horizon=horizon)
        
        o0, _ = self.oc.featurize_state_mdp(self.oc.state)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=o0.flatten().shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS])

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        self.oc.reset()
        o0, _ = self.oc.featurize_state_mdp(self.oc.state)
        return o0.flatten().astype(np.float32), {}

    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute joint action in the environment.
        
        Args:
            action: Array of [action_agent_0, action_agent_1]
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        a0, a1 = int(action[0]), int(action[1])
        joint = [Action.ALL_ACTIONS[a0], Action.ALL_ACTIONS[a1]]
        state, r, done, info = self.oc.step(joint)
        o0, _ = self.oc.featurize_state_mdp(state)
        return o0.flatten().astype(np.float32), float(r), bool(done), False, info


class OCWrapperNoise(OCWrapper):
    """
    Overcooked wrapper with observation noise.
    
    Adds Gaussian noise to observations to test robustness to sensory perturbations.
    
    Args:
        layout: Name of the Overcooked layout
        horizon: Maximum episode length
        noise_std: Standard deviation of Gaussian noise added to observations
    """
    
    def __init__(
        self, 
        layout: str = DEFAULT_CONFIG.layout,
        horizon: int = DEFAULT_CONFIG.horizon,
        noise_std: float = DEFAULT_CONFIG.noise_std
    ):
        super().__init__(layout, horizon)
        self.noise_std = noise_std
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, r, term, trunc, info = super().step(action)
        obs = (obs + np.random.normal(0, self.noise_std, size=obs.shape)).astype(np.float32)
        return obs, r, term, trunc, info


class OCWrapperDelay(OCWrapper):
    """
    Overcooked wrapper with stochastic reward delays/penalties.
    
    Simulates communication delays or action execution failures by randomly
    penalizing rewards.
    
    Args:
        layout: Name of the Overcooked layout
        horizon: Maximum episode length
        noise_prob: Probability of applying delay penalty per step
        delay_penalty: Penalty subtracted from reward when delay occurs
    """
    
    def __init__(
        self, 
        layout: str = DEFAULT_CONFIG.layout,
        horizon: int = DEFAULT_CONFIG.horizon,
        noise_prob: float = DEFAULT_CONFIG.delay_noise_prob,
        delay_penalty: float = DEFAULT_CONFIG.delay_penalty
    ):
        super().__init__(layout, horizon)
        self.noise_prob = noise_prob
        self.delay_penalty = delay_penalty
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, r, term, trunc, info = super().step(action)
        if np.random.rand() < self.noise_prob:
            r -= self.delay_penalty
        return obs, r, term, trunc, info


class OCWrapperCombo(OCWrapper):
    """
    Overcooked wrapper with combined noise and delay perturbations.
    
    Combines observation noise and stochastic reward penalties for maximum
    environmental challenge.
    
    Args:
        layout: Name of the Overcooked layout
        horizon: Maximum episode length
        noise_std: Standard deviation of observation noise
        noise_prob: Probability of delay penalty
        delay_penalty: Penalty amount when delay occurs
    """
    
    def __init__(
        self, 
        layout: str = DEFAULT_CONFIG.layout,
        horizon: int = DEFAULT_CONFIG.horizon,
        noise_std: float = DEFAULT_CONFIG.noise_std,
        noise_prob: float = DEFAULT_CONFIG.delay_noise_prob,
        delay_penalty: float = DEFAULT_CONFIG.delay_penalty
    ):
        super().__init__(layout, horizon)
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.delay_penalty = delay_penalty
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, r, term, trunc, info = super().step(action)
        obs = (obs + np.random.normal(0, self.noise_std, size=obs.shape)).astype(np.float32)
        if np.random.rand() < self.noise_prob:
            r -= self.delay_penalty
        return obs, r, term, trunc, info


class OCWrapperLLM(OCWrapper):
    """
    Overcooked wrapper with LLM-based reward shaping.
    
    Uses a language model to evaluate joint actions and provide bonus rewards
    for actions deemed "cooperative" or "helpful".
    
    Args:
        layout: Name of the Overcooked layout
        horizon: Maximum episode length
        llm_bonus: Reward bonus added when LLM judges actions as "good"
    """
    
    def __init__(
        self, 
        layout: str = DEFAULT_CONFIG.layout,
        horizon: int = DEFAULT_CONFIG.horizon,
        llm_bonus: float = DEFAULT_CONFIG.llm_bonus
    ):
        super().__init__(layout, horizon)
        self.llm_bonus = llm_bonus
        # Import here to avoid circular dependency
        from .llm_shaping import is_good
        self._is_good = is_good
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, r, term, trunc, info = super().step(action)
        a0, a1 = int(action[0]), int(action[1])
        act0 = Action.ACTION_TO_CHAR[Action.ALL_ACTIONS[a0]]
        act1 = Action.ACTION_TO_CHAR[Action.ALL_ACTIONS[a1]]
        
        prompt = (
            f"In cooperative cooking, are joint actions '{act0}' and '{act1}' helpful? "
            f"Answer good or bad."
        )
        
        if self._is_good(prompt):
            r += self.llm_bonus
        
        return obs, r, term, trunc, info


# Combined LLM + perturbation wrappers using multiple inheritance
class LLMNoise(OCWrapperLLM, OCWrapperNoise):
    """LLM shaping with observation noise."""
    
    def __init__(self, layout: str = DEFAULT_CONFIG.layout, **kwargs):
        OCWrapperLLM.__init__(self, layout, **kwargs)
        self.noise_std = kwargs.get('noise_std', DEFAULT_CONFIG.noise_std)


class LLMDelay(OCWrapperLLM, OCWrapperDelay):
    """LLM shaping with stochastic delays."""
    
    def __init__(self, layout: str = DEFAULT_CONFIG.layout, **kwargs):
        OCWrapperLLM.__init__(self, layout, **kwargs)
        self.noise_prob = kwargs.get('noise_prob', DEFAULT_CONFIG.delay_noise_prob)
        self.delay_penalty = kwargs.get('delay_penalty', DEFAULT_CONFIG.delay_penalty)


class LLMCombo(OCWrapperLLM, OCWrapperCombo):
    """LLM shaping with combined noise and delays."""
    
    def __init__(self, layout: str = DEFAULT_CONFIG.layout, **kwargs):
        OCWrapperLLM.__init__(self, layout, **kwargs)
        self.noise_std = kwargs.get('noise_std', DEFAULT_CONFIG.noise_std)
        self.noise_prob = kwargs.get('noise_prob', DEFAULT_CONFIG.delay_noise_prob)
        self.delay_penalty = kwargs.get('delay_penalty', DEFAULT_CONFIG.delay_penalty)


class HARL(OCWrapper):
    """
    Hierarchical Agent RL baseline wrapper.
    
    Adds task-completion-based reward shaping based on remaining orders.
    """
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = super().step(action)
        orders_remaining = info.get("orders_remaining", 0)
        
        if orders_remaining == 0:
            reward += 1.0
        elif orders_remaining < 3:
            reward += 0.5
        
        return obs, reward, term, trunc, info


# HARL + perturbation combinations
class HARLNoise(HARL, OCWrapperNoise):
    """HARL shaping with observation noise."""
    pass


class HARLDelay(HARL, OCWrapperDelay):
    """HARL shaping with stochastic delays."""
    pass


class HARLCombo(HARL, OCWrapperCombo):
    """HARL shaping with combined perturbations."""
    pass


class SP_PPO(OCWrapper):
    """Self-Play PPO baseline (standard OCWrapper with no modifications)."""
    pass


def make_env(
    env_name: str, 
    layout: str = DEFAULT_CONFIG.layout,
    horizon: int = DEFAULT_CONFIG.horizon
) -> Monitor:
    """
    Create an evaluation environment (no reward shaping).
    
    Args:
        env_name: One of "No Noise", "Noise", "Delay", "Combo"
        layout: Overcooked layout name
        horizon: Maximum episode length
        
    Returns:
        Monitored Gymnasium environment
    """
    e = env_name.lower()
    mapping = {
        "no noise": OCWrapper,
        "noise": OCWrapperNoise,
        "delay": OCWrapperDelay,
        "combo": OCWrapperCombo,
    }
    
    if e not in mapping:
        raise ValueError(f"Unknown env_name: {env_name}. Choose from {list(mapping.keys())}")
    
    return Monitor(mapping[e](layout, horizon))


def make_train_env(
    baseline: str, 
    layout: str = DEFAULT_CONFIG.layout,
    env_name: str = "No Noise",
    horizon: int = DEFAULT_CONFIG.horizon
) -> Monitor:
    """
    Create a training environment with appropriate reward shaping.
    
    Args:
        baseline: Training method name (e.g., "PPO+LLM", "HARL", "Baseline")
        layout: Overcooked layout name
        env_name: Perturbation regime ("No Noise", "Noise", "Delay", "Combo")
        horizon: Maximum episode length
        
    Returns:
        Monitored Gymnasium environment with appropriate wrappers
    """
    b = baseline.lower()
    e = env_name.lower()
    
    if b == "ppo+llm":
        cls_map = {
            "no noise": OCWrapperLLM,
            "noise": LLMNoise,
            "delay": LLMDelay,
            "combo": LLMCombo,
        }
        return Monitor(cls_map[e](layout, horizon=horizon))
    
    if b == "harl":
        cls_map = {
            "no noise": HARL,
            "noise": HARLNoise,
            "delay": HARLDelay,
            "combo": HARLCombo,
        }
        return Monitor(cls_map[e](layout, horizon=horizon))
    
    if b == "sp_ppo":
        return Monitor(SP_PPO(layout, horizon))
    
    # Baseline, CC_PPO, PBT_PPO use standard env
    return make_env(env_name, layout, horizon)


def warmup_mlam(layout: str = DEFAULT_CONFIG.layout, horizon: int = DEFAULT_CONFIG.horizon):
    """
    Pre-warm the MediumLevelActionManager to avoid pickle errors in parallel workers.
    
    Call this once on the main process before spawning parallel training jobs.
    
    Args:
        layout: Overcooked layout name
        horizon: Maximum episode length
    """
    print("Prewarming MLAM planner...")
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    _ = env.featurize_state_mdp(env.state)
    print("MLAM prewarm complete.")

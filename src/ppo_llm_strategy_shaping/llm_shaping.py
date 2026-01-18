"""
LLM-based reward shaping module.

This module provides functionality for using language models to evaluate
joint actions and provide reward bonuses for cooperative behavior.
"""

import torch
from typing import Tuple, Optional, Any

from .config import DEFAULT_CONFIG

# Global LLM instances (lazy loaded)
_GLOBAL_LLM: Optional[Any] = None
_GLOBAL_TOK: Optional[Any] = None


def get_llm(
    model_name: str = DEFAULT_CONFIG.llm_model_name,
    device: str = DEFAULT_CONFIG.device
) -> Tuple[Any, Any]:
    """
    Lazy-load the LLM and tokenizer.
    
    Uses global caching to avoid reloading the model multiple times.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "EleutherAI/gpt-neo-1.3B")
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Tuple of (model, tokenizer). Model may be False if loading failed.
    """
    global _GLOBAL_LLM, _GLOBAL_TOK
    
    if _GLOBAL_LLM is None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading LLM: {model_name} on {device}")
            _GLOBAL_TOK = AutoTokenizer.from_pretrained(model_name)
            _GLOBAL_LLM = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(device).eval()
            print("LLM loaded successfully")
            
        except Exception as e:
            print(f"Failed to load LLM {model_name}: {e}")
            _GLOBAL_LLM = False
            _GLOBAL_TOK = None
    
    return _GLOBAL_LLM, _GLOBAL_TOK


def reset_llm():
    """Reset the global LLM cache to allow reloading with different settings."""
    global _GLOBAL_LLM, _GLOBAL_TOK
    _GLOBAL_LLM = None
    _GLOBAL_TOK = None


@torch.no_grad()
def is_good(
    prompt: str,
    model_name: str = DEFAULT_CONFIG.llm_model_name,
    device: str = DEFAULT_CONFIG.device
) -> bool:
    """
    Evaluate whether an action is "good" using LLM logit comparison.
    
    Compares the logit of " good" vs " bad" tokens given the prompt.
    If logit(good) > logit(bad), returns True.
    
    Args:
        prompt: The prompt describing the action to evaluate
        model_name: HuggingFace model identifier
        device: Device to run inference on
        
    Returns:
        True if LLM judges the action as "good", False otherwise
    """
    llm, tok = get_llm(model_name, device)
    
    if not llm:
        return False
    
    try:
        # Tokenize prompt
        enc = tok(prompt, return_tensors="pt").to(device)
        
        # Get logits for next token
        logits = llm(**enc).logits[0, -1]
        
        # Get token IDs for " good" and " bad"
        def get_token_idx(tok_str: str) -> Optional[int]:
            ids = tok.encode(tok_str, add_special_tokens=False)
            return ids[0] if ids else None
        
        good_idx = get_token_idx(" good")
        bad_idx = get_token_idx(" bad")
        
        if good_idx is None or bad_idx is None:
            return False
        
        # Compare logits
        return logits[good_idx].item() > logits[bad_idx].item()
    
    except Exception as e:
        print(f"LLM inference failed: {e}")
        return False


def compute_llm_reward_bonus(
    action_0_char: str,
    action_1_char: str,
    bonus: float = DEFAULT_CONFIG.llm_bonus,
    model_name: str = DEFAULT_CONFIG.llm_model_name,
    device: str = DEFAULT_CONFIG.device
) -> float:
    """
    Compute the LLM-based reward bonus for a joint action.
    
    Args:
        action_0_char: Character representation of agent 0's action
        action_1_char: Character representation of agent 1's action
        bonus: Reward bonus to add if action is deemed "good"
        model_name: HuggingFace model identifier
        device: Device to run inference on
        
    Returns:
        Reward bonus (0.0 or the specified bonus value)
    """
    prompt = (
        f"In cooperative cooking, are joint actions '{action_0_char}' and '{action_1_char}' helpful? "
        f"Answer good or bad."
    )
    
    if is_good(prompt, model_name, device):
        return bonus
    return 0.0


def get_available_models() -> dict:
    """
    Return a dictionary of available LLM models for experiments.
    
    Returns:
        Dict mapping model tags to HuggingFace model names
    """
    return {
        "gpt-neo-125M": "EleutherAI/gpt-neo-125M",
        "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
        "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
    }

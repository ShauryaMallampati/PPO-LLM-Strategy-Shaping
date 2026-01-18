# PPO-LLM Strategy Shaping

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLM-guided reward shaping for multi-agent coordination in Overcooked**

This repository contains the code for reproducing experiments on using Large Language Models (LLMs) to provide reward shaping signals for Proximal Policy Optimization (PPO) agents in the Overcooked multi-agent environment.

## Overview

We investigate whether LLM-based reward shaping can accelerate learning and improve coordination in multi-agent settings. Our approach:

1. **PPO+LLM**: Uses GPT-Neo to evaluate state descriptions and provide auxiliary reward signals based on whether the current state appears "good" or "bad"
2. **Robustness Testing**: Evaluates policies under observation noise, action delays, and combined perturbations
3. **Comprehensive Baselines**: Compares against CC-PPO, SP-PPO, HARL, and PBT-PPO

## Installation

```bash
# Clone the repository
git clone https://github.com/ShauryaMallampati/PPO-LLM-Strategy-Shaping.git
cd PPO-LLM-Strategy-Shaping

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Project Structure

```
PPO-LLM-Strategy-Shaping/
├── src/
│   └── ppo_llm_strategy_shaping/
│       ├── __init__.py          # Package exports
│       ├── config.py            # Configuration dataclass
│       ├── env_wrappers.py      # Gymnasium environment wrappers
│       ├── llm_shaping.py       # LLM reward shaping logic
│       ├── train.py             # Training loops and callbacks
│       ├── evaluation.py        # Analysis functions
│       └── utils.py             # Utility functions
├── notebooks/
│   └── experiments.ipynb        # Main experiment notebook
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### Python API

```python
from ppo_llm_strategy_shaping import Config, train_all, evaluate

# Create configuration
config = Config(
    layout="cramped_room",
    horizon=400,
    seeds=[42, 123, 456],
)

# Train all baselines
results = train_all(config, n_jobs=4, verbose=1)

# Evaluate
from ppo_llm_strategy_shaping import nash_gap_analysis, latency_analysis
nash_results = nash_gap_analysis(config)
latency_results = latency_analysis(config)
```

### Using the Notebook

Open `notebooks/experiments.ipynb` for an interactive walkthrough of:
- Training all baselines
- Nash gap analysis
- Latency measurements
- Robustness evaluation
- Task completion metrics

## Baselines

| Baseline | Description | Training Steps |
|----------|-------------|----------------|
| Baseline | Vanilla PPO | 1,000,000 |
| PPO+LLM | PPO with LLM reward shaping | 600,000 |
| CC_PPO | Centralized Critic PPO | 1,000,000 |
| SP_PPO | Self-Play PPO | 1,000,000 |
| HARL | Heterogeneous-Agent RL | 1,000,000 |
| PBT_PPO | Population-Based Training | 1,000,000 |

## Environment Perturbations

We test robustness under four regimes:

- **No Noise**: Standard environment
- **Noise**: Gaussian observation noise (std=0.01)
- **Delay**: 20% probability of action delay with penalty
- **Combo**: Combined noise and delay

## Configuration

Key hyperparameters in `Config`:

```python
Config(
    # Environment
    layout="cramped_room",
    horizon=400,
    
    # PPO
    lr=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    
    # LLM Shaping
    llm_model="EleutherAI/gpt-neo-1.3B",
    llm_reward_scale=0.1,
    
    # Perturbations
    noise_std=0.01,
    delay_prob=0.2,
    delay_penalty=-0.5,
)
```

## Results

Our experiments show that:

1. PPO+LLM achieves comparable performance with 40% fewer training steps
2. LLM-shaped policies show improved robustness to perturbations
3. Inference latency remains practical for real-time deployment

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mallampati2025ppollm,
  title={PPO-LLM Strategy Shaping: LLM-Guided Reward Shaping for Multi-Agent Coordination},
  author={Mallampati, Shaurya},
  year={2025},
  url={https://github.com/ShauryaMallampati/PPO-LLM-Strategy-Shaping}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Overcooked-AI environment (https://github.com/HumanCompatibleAI/overcooked_ai)
- Stable-Baselines3 for PPO implementation (https://github.com/DLR-RM/stable-baselines3)
- EleutherAI for GPT-Neo models (https://www.eleuther.ai/)

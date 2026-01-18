# PPO-LLM Strategy Shaping

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM-guided reward shaping for multi-agent coordination in Overcooked.

Code for reproducing experiments on using Large Language Models to provide reward shaping signals for PPO agents in the Overcooked multi-agent environment.

## Overview

Investigates whether LLM-based reward shaping can accelerate learning and improve coordination in multi-agent settings. The experiments include:

1. **PPO+LLM**: Uses GPT-Neo to evaluate state descriptions and provide auxiliary reward signals
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
├── src/ppo_llm_strategy_shaping/
│   ├── config.py              # Configuration dataclass
│   ├── env_wrappers.py        # Gymnasium environment wrappers
│   ├── llm_shaping.py         # LLM reward shaping
│   ├── train.py               # Training and callbacks
│   ├── evaluation.py          # Analysis functions
│   └── utils.py               # Utilities
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_training.ipynb
│   ├── 03_nash_analysis.ipynb
│   ├── 04_latency_analysis.ipynb
│   ├── 05_robustness_analysis.ipynb
│   ├── 06_task_completion.ipynb
│   └── 07_llm_sensitivity.ipynb
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

Jupyter notebooks are provided in `notebooks/` for interactive experimentation. Each notebook is self-contained and can run independently in Google Colab:

1. `01_setup.ipynb` - Environment and dependency setup
2. `02_training.ipynb` - Train all baselines across perturbation regimes
3. `03_nash_analysis.ipynb` - Best response analysis
4. `04_latency_analysis.ipynb` - LLM inference latency measurement
5. `05_robustness_analysis.ipynb` - Robustness under perturbations
6. `06_task_completion.ipynb` - Task completion metrics
7. `07_llm_sensitivity.ipynb` - Model size comparison (125M vs 1.3B)

## Baselines

| Baseline | Training Steps |
|----------|----------------|
| Baseline | 1,000,000 |
| PPO+LLM | 600,000 |
| CC_PPO | 1,000,000 |
| SP_PPO | 1,000,000 |
| HARL | 1,000,000 |
| PBT_PPO | 1,000,000 |

## Environment Perturbations

Robustness tested under four regimes:

- No Noise: Standard environment
- Noise: Gaussian observation noise (std=0.01)
- Delay: 20% probability action delay with penalty
- Combo: Combined noise and delay

## Configuration

Key hyperparameters in `Config`:

```python
Config(
    layout="asymmetric_advantages",
    horizon=400,
    
    # PPO
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=2048,
    gamma=0.99,
    
    # LLM shaping
    llm_model_name="EleutherAI/gpt-neo-1.3B",
    llm_bonus=0.2,
    
    # Perturbations
    noise_std=0.01,
    delay_noise_prob=0.2,
    delay_penalty=0.5,
)
```

## Results

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

# PPO-LLM Strategy Shaping

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.02002-b31b1b.svg)](https://arxiv.org/abs/2507.02002)


**Note:** Developed in Google Colab. Local execution may require syntax adjustments for file paths, GPU configuration, and environment setup.

## Overview

Evaluates LLM-based reward shaping for accelerating learning and improving coordination in multi-agent PPO training. The approach uses language models to evaluate state descriptions and provide auxiliary reward signals. Experiments include:

- **PPO+LLM**: Reward shaping with GPT-Neo
- **Robustness Evaluation**: Observation noise, action delays, combined perturbations
- **Baselines**: CC-PPO, SP-PPO, HARL, PBT-PPO

## Installation

```bash
git clone https://github.com/ShauryaMallampati/PPO-LLM-Strategy-Shaping.git
cd PPO-LLM-Strategy-Shaping
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
PPO-LLM-Strategy-Shaping/
├── src/ppo_llm_strategy_shaping/
│   ├── config.py
│   ├── env_wrappers.py
│   ├── llm_shaping.py
│   ├── train.py
│   ├── evaluation.py
│   └── utils.py
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

## Usage

Notebook-based interface for all experiments:

| Notebook | Purpose |
|----------|---------|
| 01_setup.ipynb | Environment setup |
| 02_training.ipynb | Train baselines across perturbation regimes |
| 03_nash_analysis.ipynb | Best response analysis |
| 04_latency_analysis.ipynb | LLM inference latency |
| 05_robustness_analysis.ipynb | Perturbation robustness |
| 06_task_completion.ipynb | Task completion metrics |
| 07_llm_sensitivity.ipynb | Model size comparison |

## Experimental Setup

**Baselines:** Baseline, PPO+LLM, CC-PPO, SP-PPO, HARL, PBT-PPO

**Environment:** Overcooked with asymmetric advantages layout (horizon 400)

**Perturbation Regimes:**
- Standard (no noise)
- Observation noise (σ=0.01)
- Action delay (20% probability, penalty 0.5)
- Combined noise and delay

**LLM Configuration:** GPT-Neo 1.3B with reward bonus 0.2

## Results

- PPO+LLM achieves comparable performance with 40% fewer training steps
- Improved robustness under observation noise, action delays, and combined perturbations
- Practical inference latency for real-time deployment

## Citation

Mallampati, S., Shelim, R., Saad, W., & Ramakrishnan, N. (2025). Dynamic strategy adaptation in multi-agent environments with large language models. *arXiv preprint arXiv:2507.02002*.

```bibtex
@misc{mallampati2025dynamicstrategyadaptationmultiagent,
      title={Dynamic Strategy Adaptation in Multi-Agent Environments with Large Language Models}, 
      author={Shaurya Mallampati and Rashed Shelim and Walid Saad and Naren Ramakrishnan},
      year={2025},
      eprint={2507.02002},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2507.02002}
}
```

## References

- [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [EleutherAI GPT-Neo](https://www.eleuther.ai/)

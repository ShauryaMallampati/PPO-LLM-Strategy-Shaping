# PPO-LLM Strategy Shaping

Dynamic Strategy Adaptation in Multi-Agent Environments using PPO with LLM-based reward shaping.

## Notebooks

Run these notebooks **in order** on Google Colab:

| Notebook | Description |
|----------|-------------|
| [01_setup.ipynb](notebooks/01_setup.ipynb) | Setup dependencies, clone overcooked_ai, mount Google Drive |
| [02_training.ipynb](notebooks/02_training.ipynb) | Train all baselines (Baseline, PPO+LLM, CC_PPO, SP_PPO, HARL, PBT_PPO) |
| [03_nash_analysis.ipynb](notebooks/03_nash_analysis.ipynb) | Compute Nash equilibrium gaps via best-response training |
| [04_latency_analysis.ipynb](notebooks/04_latency_analysis.ipynb) | Measure per-step decision latency (ms) |
| [05_robustness_analysis.ipynb](notebooks/05_robustness_analysis.ipynb) | Compute robustness deltas across environment perturbations |
| [06_task_completion.ipynb](notebooks/06_task_completion.ipynb) | Compute sparse reward task completion metrics |
| [07_llm_sensitivity.ipynb](notebooks/07_llm_sensitivity.ipynb) | Train PPO+LLM with GPT-Neo 125M for LLM size sensitivity |

## Environment Regimes

- **No Noise**: Clean environment
- **Noise**: Gaussian observation noise (σ=0.01)
- **Delay**: Stochastic reward delays (20% prob, -0.5 penalty)
- **Combo**: Both noise and delay

## Baselines

- **Baseline**: Vanilla PPO
- **PPO+LLM**: PPO with LLM-based reward shaping (GPT-Neo 1.3B)
- **CC_PPO**: Centralized Critic PPO
- **SP_PPO**: Self-Play PPO
- **HARL**: Hierarchical Attention RL
- **PBT_PPO**: Population-Based Training PPO

## Configuration

- Layout: `asymmetric_advantages`
- Horizon: 400 steps
- Seeds: [1001, 2002, 3003, 4004, 5005]
- PPO+LLM: 600k steps
- Other baselines: 1M steps

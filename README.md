# PPO-LLM Strategy Shaping

PPO in Overcooked-AI with frozen-LLM binary reward shaping, plus post-training unilateral-deviation diagnostics.

## Notebooks

Run these notebooks in order on Google Colab:

| Notebook | Description |
|----------|-------------|
| [01_setup.ipynb](notebooks/01_setup.ipynb) | Setup dependencies, clone `overcooked_ai`, and mount Google Drive |
| [02_training.ipynb](notebooks/02_training.ipynb) | Train the main baselines on `asymmetric_advantages`; PPO+LLM uses frozen-LLM binary reward shaping |
| [03_nash_analysis.ipynb](notebooks/03_nash_analysis.ipynb) | Run the post-training unilateral-deviation diagnostic via a fixed-budget best-response PPO probe (legacy filename retained for compatibility) |
| [04_latency_analysis.ipynb](notebooks/04_latency_analysis.ipynb) | Measure per-step decision latency (ms) |
| [05_robustness_analysis.ipynb](notebooks/05_robustness_analysis.ipynb) | Compute robustness deltas across environment perturbations |
| [06_task_completion.ipynb](notebooks/06_task_completion.ipynb) | Compute sparse reward task-completion metrics |
| [07_llm_sensitivity.ipynb](notebooks/07_llm_sensitivity.ipynb) | Sensitivity sweep for PPO+LLM with frozen GPT-Neo 125M |

## Environment Regimes

- No Noise: clean environment
- Noise: Gaussian observation noise (`σ=0.01`)
- Delay: stochastic reward delays (`20%` probability, `-0.5` penalty)
- Combo: both noise and delay

## Baselines

- Baseline: vanilla PPO
- PPO+LLM: PPO with frozen GPT-Neo binary reward shaping
- CC_PPO: centralized-critic PPO
- SP_PPO: self-play PPO
- HARL: hierarchical attention RL
- PBT_PPO: population-based training PPO

## Paper-Tracked Configuration

The public notebook defaults used for the main paper setup are:

- Main layout: `asymmetric_advantages`
- Horizon: `400`
- Seeds: `[1001, 2002, 3003, 4004, 5005]`
- PPO hyperparameters: `n_steps=2048`, `batch_size=2048`, `learning_rate=3e-4`, `gamma=0.99`
- Frozen-LLM shaping bonus: `LLM_BONUS = 0.2`
- PPO+LLM training budget: `600_000` steps
- Other baseline budgets: `1_000_000` steps
- Final policy evaluation in training notebooks: `10` rollout episodes
- Post-training unilateral-deviation diagnostic: `200_000` PPO steps for the best-response probe and `20` evaluation rollouts

## Reproducibility And Terminology Notes

- The main paper evaluation layout is `asymmetric_advantages`.
- Burger Kitchen is illustrative/qualitative only; it is not the main evaluation layout in this public repo.
- The method is frozen-LLM binary reward shaping: during PPO training, the frozen language model contributes a fixed `+0.2` bonus when the prompt-conditioned comparison favors `good` over `bad`.
- PPO training uses discounted returns internally with `gamma=0.99`.
- The unilateral-deviation diagnostic reports undiscounted episodic team return.
- The best-response probe is single-round, fixed-budget PPO training. It is not iterated best response.
- The unilateral-deviation gap is the one-sided best-response gain under the tested policy class and training budget.
- The diagnostic is an empirical robustness probe. This repo does not claim equilibrium computation or convergence proofs.
- Older filenames or helper names may still mention `nash` for backward compatibility, but the public-facing term is unilateral deviation.

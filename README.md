# PPO-LLM Strategy Shaping

Research code for the PPO and frozen language-model reward-shaping experiments in Overcooked-AI.

## Summary

This repository contains notebooks for training and analyzing PPO agents with an optional frozen GPT-Neo reward-shaping signal. The final paper tracks four method labels: Baseline, PPO+LLM, HARL, and PBT_PPO.

PPO performs action selection. PPO+LLM adds a training-time reward bonus from a frozen GPT-Neo evaluator. The evaluator receives only the joint primitive action pair, compares the next-token logits for `good` and `bad`, and adds `+0.2` to the team reward when `good` scores higher.

## Repository Structure

| Path | Description |
|------|-------------|
| `notebooks/01_setup.ipynb` | Dependency setup, Overcooked-AI clone, and Google Drive mounting. |
| `notebooks/02_training.ipynb` | Main training notebook for the final method labels. |
| `notebooks/03_fixed_partner_retraining_diagnostic.ipynb` | Fixed-partner PPO retraining diagnostic notebook. |
| `notebooks/04_latency_analysis.ipynb` | Evaluation-time PPO/environment latency analysis. |
| `notebooks/05_robustness_analysis.ipynb` | Return-drop analysis across perturbation regimes. |
| `notebooks/06_task_completion.ipynb` | Return-normalization analysis relative to the expected penalty baseline. |
| `notebooks/07_llm_sensitivity.ipynb` | Frozen-LLM scale sensitivity analysis using GPT-Neo 125M. |

## Method Labels

| Label | Description |
|-------|-------------|
| Baseline | PPO trained with the environment reward only. |
| PPO+LLM | PPO with training-time frozen GPT-Neo reward shaping. |
| HARL | PPO-based comparison label that reduces to a constant per-step reward offset in the released code. |
| PBT_PPO | PPO with a lightweight learning-rate adaptation callback. |

## Environment Regimes

| Regime | Description |
|--------|-------------|
| No Noise | Base environment without observation noise or reward penalty. |
| Noise | Adds Gaussian observation noise with standard deviation `0.01`. |
| Delay | Stochastic per-step reward penalty: subtracts `0.5` with probability `0.2` each step. |
| Combo | Applies both Gaussian observation noise and the stochastic per-step reward penalty. |

The reward-penalty condition has an expected cumulative penalty of approximately `-40` over the 400-step horizon.

## Configuration

- Layout: `asymmetric_advantages`
- Horizon: `400`
- Seeds: `[1001, 2002, 3003, 4004, 5005]`
- PPO hyperparameters: `n_steps=2048`, `batch_size=2048`, `learning_rate=3e-4`, `gamma=0.99`
- PPO+LLM shaping bonus: `LLM_BONUS = 0.2`
- PPO+LLM training budget: `600_000` environment steps
- Baseline, HARL, and PBT_PPO training budgets: `1_000_000` environment steps
- Fixed-partner PPO retraining diagnostic: `200_000` retraining steps and `20` evaluation rollouts

## Running the Notebooks

1. `01_setup.ipynb`
2. `02_training.ipynb`
3. `03_fixed_partner_retraining_diagnostic.ipynb`
4. `04_latency_analysis.ipynb`
5. `05_robustness_analysis.ipynb`
6. `06_task_completion.ipynb`
7. `07_llm_sensitivity.ipynb`

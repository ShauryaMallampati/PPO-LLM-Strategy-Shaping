# PPO-LLM Strategy Shaping

Research code for **Dynamic Strategy Adaptation in Multi-Agent Environments with Large Language Models**.

This repository contains the notebooks used to train and analyze PPO agents in Overcooked-AI with an optional frozen language-model reward-shaping signal. The final paper tracks four method labels: Baseline, PPO+LLM, HARL, and PBT_PPO.

## Method Summary

PPO is the only trained action-selection policy. In the PPO+LLM condition, a frozen GPT-Neo evaluator is queried only during training. The evaluator receives a prompt containing only the two primitive action symbols selected by PPO, compares the next-token logits for `good` and `bad`, and adds a bounded `+0.2` team-reward bonus when `good` scores higher. The language model is not updated, does not select actions, and is not queried during evaluation or deployment.

Because the prompt contains only the joint primitive-action pair, the frozen evaluator acts as a fixed action-pair scorer over the 36 possible two-agent primitive-action pairs. The shaping signal is an action-pair reward bonus, not potential-based reward shaping.

## Repository Structure

| Path | Description |
|------|-------------|
| `notebooks/01_setup.ipynb` | Dependency setup, Overcooked-AI clone, and Google Drive mounting. |
| `notebooks/02_training.ipynb` | Main training notebook for the final method labels. |
| `notebooks/03_nash_analysis.ipynb` | Legacy filename for the fixed-partner PPO retraining diagnostic notebook. The filename is retained for artifact compatibility. |
| `notebooks/04_latency_analysis.ipynb` | Evaluation-time PPO/environment latency analysis. |
| `notebooks/05_robustness_analysis.ipynb` | Return-drop analysis across perturbation regimes. |
| `notebooks/06_task_completion.ipynb` | Return-normalization analysis relative to the expected penalty baseline. |
| `notebooks/07_llm_sensitivity.ipynb` | Frozen-LLM scale sensitivity analysis using GPT-Neo 125M. |

## Final Method Labels

| Label | Description |
|-------|-------------|
| Baseline | PPO trained with the environment reward only. |
| PPO+LLM | PPO with training-time frozen GPT-Neo reward shaping. |
| HARL | PPO-based comparison label that reduces to a constant per-step reward offset in the released code. |
| PBT_PPO | PPO with a lightweight learning-rate adaptation callback; no maintained population is used. |

## Environment Regimes

| Regime | Description |
|--------|-------------|
| No Noise | Base environment without observation noise or reward penalty. |
| Noise | Adds i.i.d. Gaussian observation noise with standard deviation `0.01`. |
| Delay | Legacy regime name for a stochastic per-step reward penalty: subtracts `0.5` with probability `0.2` each step. |
| Combo | Applies both Gaussian observation noise and the stochastic per-step reward penalty. |

Over a 400-step episode, the reward-penalty condition has an expected cumulative penalty of approximately `-40`. This value is an expected penalty baseline, not a hard reward floor.

## Paper-Tracked Configuration

The main paper configuration uses:

- Layout: `asymmetric_advantages`
- Horizon: `400`
- Seeds: `[1001, 2002, 3003, 4004, 5005]`
- PPO hyperparameters: `n_steps=2048`, `batch_size=2048`, `learning_rate=3e-4`, `gamma=0.99`
- PPO+LLM shaping bonus: `LLM_BONUS = 0.2`
- PPO+LLM training budget: `600_000` environment steps
- Baseline, HARL, and PBT_PPO training budgets: `1_000_000` environment steps
- Final policy evaluation in the training notebook: `10` rollout episodes
- Fixed-partner PPO retraining diagnostic: `200_000` retraining steps and `20` evaluation rollouts

The wall-clock training times reported in the paper are budget-specific run times. They should not be interpreted as a matched-budget estimate of training-time LLM-query overhead.

## Running the Notebooks

The notebooks are designed for Google Colab with Google Drive paths. Run them in order when reproducing the full workflow:

1. `01_setup.ipynb`
2. `02_training.ipynb`
3. `03_nash_analysis.ipynb` for the fixed-partner PPO retraining diagnostic
4. `04_latency_analysis.ipynb`
5. `05_robustness_analysis.ipynb`
6. `06_task_completion.ipynb`
7. `07_llm_sensitivity.ipynb`

The legacy filename `03_nash_analysis.ipynb` is retained to preserve compatibility with existing artifact paths. The diagnostic itself should be interpreted only as a fixed-budget PPO retraining check against a fixed partner, not as a Nash equilibrium, best-response, unilateral-deviation, or convergence analysis.

## Reproducibility Notes

This repository provides the paper notebooks and configuration details used for the reported experiments. Some environment paths are Colab/Google-Drive-specific, and exact reruns may depend on package versions, hardware, and the Overcooked-AI revision available at setup time. The paper's latency measurements refer to evaluation-time PPO policy and environment-step latency; they do not include training-time language-model query cost.

## Citation

If you use this repository, cite the associated paper:

```bibtex
@article{mallampati2026dynamic,
  title={Dynamic Strategy Adaptation in Multi-Agent Environments with Large Language Models},
  author={Mallampati, Shaurya and Shelim, Rashed and Saad, Walid and Ramakrishnan, Naren},
  year={2026}
}
```

# PPO-LLM Strategy Shaping

Research code for PPO and frozen language-model reward-shaping experiments in Overcooked-AI.

## Notebooks

1. `notebooks/01_setup.ipynb`
2. `notebooks/02_training.ipynb`
3. `notebooks/03_fixed_partner_retraining_diagnostic.ipynb` — Fixed-partner PPO retraining diagnostic used to measure the return change after retraining one side against the other side held fixed under a fixed PPO budget.
4. `notebooks/04_latency_analysis.ipynb`
5. `notebooks/05_robustness_analysis.ipynb`
6. `notebooks/06_task_completion.ipynb`
7. `notebooks/07_llm_sensitivity.ipynb`

## Method Labels

- Baseline
- PPO+LLM
- HARL
- PBT_PPO

## Environment Regimes

- No Noise
- Noise
- Delay
- Combo

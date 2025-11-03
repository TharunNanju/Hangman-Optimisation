# Hangman-Optimisation

This repository contains an interactive Jupyter notebook that builds a Hangman-playing AI by combining a probabilistic Hidden Markov Model (HMM) with a Deep Q-Network (DQN) reinforcement learning agent.

## Requirements

- Python 3.11+
- Dependencies installed in the notebook kernel, including:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `torch`
  - `tqdm`

## Data

The `Data/` directory includes two text files:

- `corpus.txt` – training words for the HMM and RL environment.
- `test.txt` – evaluation set used for final scoring.

## Notebook Overview

Open `hangman_ai_solution.ipynb` to explore the complete pipeline:

1. **Data loading and analysis** – visualize corpus/test word-length distributions.
2. **Advanced HMM** – train length-specific probabilistic models for letter prediction.
3. **Hangman environment** – simulate the Hangman game for reinforcement learning.
4. **DQN agent** – combine HMM-derived features with a neural network policy.
5. **Training & evaluation** – train the agent and measure performance on the held-out test set.
6. **Visualizations & reports** – export charts and serialized artifacts for later inspection.

## Running the Notebook

1. Launch Jupyter Lab or open the notebook in VS Code.
2. (Optional) Set `FAST_MODE=1` in the environment to run a shortened training schedule.
3. Execute the cells sequentially. The notebook will create artifacts in the `artifacts/` directory:
   - `dqn_agent.pth`
   - `hmm_model.pkl`
   - `results_summary.pkl`
   - `training_progress.png`
   - `evaluation_results.png`
   - `final_score_report.txt`

## Notes

- Full training is computationally intensive; fast mode is helpful for quick smoke tests.
- The serialized HMM file stores frequency tables instead of the raw object to remain picklable.

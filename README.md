# Hybrid RL–GA Framework for Blackjack Strategy

This project is a blackjack AI experimentation framework that combines:

- A custom blackjack simulator (`blackjack.py`, `jack.py`)
- Tabular RL agents (Q-learning, QV-learning) in `agentRL.py`
- A Genetic Algorithm over full decision tables (`GA.py`, `agentGA.py`)
- A hybrid GA-over-RL module that evolves RL hyperparameters (`GA_RL_hp.py`)

The main entrypoint is `run_blackjack_edina.py`, which provides a menu to run:
- Single-agent RL experiments
- Multi-agent RL experiments (Q-learning or QV-learning)
- GA over full hit/stand/double/split policies
- GA-over-RL hyperparameter search (optimizing α, γ, ε, and algorithm type)

## Key Results (example)

- Q-learning agents reach winrates of about **~41.6%** after training.
- QV-learning agents reach about **~42.2%**, empirically outperforming Q-learning and approaching optimal basic-strategy performance.
- The GA-over-RL module finds strong QV-learning hyperparameter settings automatically via repeated training/evaluation cycles.

*(Exact numbers depend on seeds and training budgets.)*

## Setup

```bash
git clone https://github.com/<your-username>/blackjack-rl-ga-framework.git
cd blackjack-rl-ga-framework
pip install -r requirements.txt

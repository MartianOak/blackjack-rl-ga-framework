# Hybrid RL–GA Blackjack Framework

This project is a blackjack strategy experimentation framework that combines:

- A custom blackjack simulator (`blackjack.py`, `jack.py`)
- Tabular RL agents (Q-learning, QV-learning) in `agentRL.py`
- A Genetic Algorithm over full decision tables (`GA.py`, `agentGA.py`)
- A hybrid GA-over-RL module that optimizes RL hyperparameters (`GA_RL_hp.py`)

The main entrypoint is `run_blackjack_edina.py`, which provides a menu to run:

1. Single-agent RL (Q-learning)
2. Multi-agent RL (Q-learning)
3. Multi-agent RL (QV-learning)
4. GA over full hit/stand/double/split policies
5. GA-over-RL hyperparameter search (α, γ, ε, algorithm type)

---

## Project Highlights

- Implemented **Q-learning** and **QV-learning** agents in a custom blackjack simulator.
- Evolved complete strategy tables using a **Genetic Algorithm** over policy space.
- Built a **GA-over-RL hyperparameter search** that repeatedly trains RL agents and uses their winrates as fitness.
- Fixed critical RL stability bugs (e.g. split-state QV updates) and integrated everything into a single experiment driver.

**Example result:**  
QV-learning configurations found by the GA-over-RL search achieve around **~42.2% winrate** vs **~41.6%** for tuned Q-learning, approaching optimal basic-strategy performance.

---

## Installation

```bash
git clone https://github.com/MartianOak/blackjack-rl-ga-framework.git
cd blackjack-rl-ga-framework
pip install -r requirements.txt

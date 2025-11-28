#!/usr/bin/env python3
"""
Run Blackjack RL & GA end-to-end on EDINA.

Usage:
    python3 run_blackjack_edina.py
"""

import os
import textwrap
import importlib
import random
import numpy as np

# -----------------------------
# 1. Basic setup & patching
# -----------------------------

def safe_replace_in_file(path, old, new):
    """Replace 'old' with 'new' in a file, if 'old' is found."""
    if not os.path.exists(path):
        print(f"[WARN] {path} not found; skipping patch.")
        return
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        print(f"[INFO] Pattern not found in {path}; patch may already be applied.")
        return
    text = text.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[PATCHED] {path}")


def setup_project():
    """Run once at the start: ensure dirs, stub files, and patches."""
    # Work in the directory of this script
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    print("Project root:", project_root)

    # --- Ensure PSO.py exists (stub if needed) ---
    if not os.path.exists("PSO.py"):
        with open("PSO.py", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""\
            # PSO.py – minimal stub so `import PSO` works

            class PSO:
                def __init__(self):
                    self.gen = 0
                    self.n = 0

                def pso(self):
                    raise NotImplementedError("PSO.pso() is not implemented in this stub.")
            """))
        print("[INFO] Created PSO.py stub.")
    else:
        print("[INFO] PSO.py already exists.")

    # --- Ensure Tables/ and SplitTables/ exist ---
    os.makedirs("Tables", exist_ok=True)
    os.makedirs("SplitTables", exist_ok=True)
    print("[INFO] Ensured Tables/ and SplitTables/ directories exist.")

    # --- Patch known issues --------------------------------------
    # 1) AgentRL.getActionEGreedy dealerHand bug (if still present)
    agent_rl_bug = (
        "        else:\n"
        "            #choose greedily\n"
        "            return self.getActionGreedy(playerHand, dealerHand[0])"
    )
    agent_rl_fix = (
        "        else:\n"
        "            # choose greedily\n"
        "            return self.getActionGreedy(playerHand, dealerHand)"
    )
    safe_replace_in_file("agentRL.py", agent_rl_bug, agent_rl_fix)

    # 2) GA multiprocessing: make sure we always have at least 1 worker
    ga_bug = "mp.cpu_count() - 2"
    ga_fix = "max(1, mp.cpu_count() - 1)"  # always >= 1
    safe_replace_in_file("GA.py", ga_bug, ga_fix)

    print("[INFO] Setup & patching complete.")


# -----------------------------
# 2. RL: Train a single agent
# -----------------------------

def run_rl():
    from agentRL import AgentRL
    import jack

    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    rl_agent = AgentRL()

    # Algorithm: 1 = Q-learning, 3 = QV-learning (check your agentRL implementation)
    rl_agent.alg = 1  # Q-learning

    # Policy: 1 = Greedy, 2 = e-Greedy, 3 = Softmax
    rl_agent.pol = 2  # ε-greedy

    # Hyperparameters – tweak as you like
    rl_agent.alpha = 0.4       # learning rate
    rl_agent.gamma = 0.9       # discount
    rl_agent.startEpsilon = 0.05
    rl_agent.epsilon = rl_agent.startEpsilon

    # Number of training games
    rl_agent.epochs = 1_000_000  # reduce if you want quick tests

    print("\n[RL] Training RL agent with Q-learning...")
    trained_agent = jack.game(rl_agent)  # your jack.game(agent) function
    print("[RL] Training complete.\n")

    # If jack.game returns the same object, this is redundant but harmless.
    agent_to_report = trained_agent if trained_agent is not None else rl_agent

    # Print whatever your Agent class exposes
    if hasattr(agent_to_report, "printResults"):
        agent_to_report.printResults()
    else:
        print("[RL] No printResults() method found on agent – training finished, but nothing to print.")

    print("[RL] If your agent code saves tables, they should be in ./Tables and ./SplitTables now.")


def run_rl_multi(num_agents=10, alg=1, pol=2):
    """
    Train multiple RL agents and report average winrates.

    alg: 1 = Q-learning, 3 = QV-learning
    pol: 1 = Greedy, 2 = ε-Greedy, 3 = Softmax
    """
    from agentRL import AgentRL
    import jack
    import importlib

    importlib.reload(jack)

    results_overall = []
    results_tail = []

    for k in range(num_agents):
        print(f"\n[RL multi] Training agent {k+1}/{num_agents} (alg={alg}, pol={pol})...")

        # different seed per agent
        random.seed(k + 1)
        np.random.seed(k + 1)

        rl_agent = AgentRL()

        # algorithm (1 = Q-learning, 3 = QV-learning)
        rl_agent.alg = alg

        # policy: 1 = Greedy, 2 = ε-Greedy, 3 = Softmax
        rl_agent.pol = pol

        # hyperparameters (match thesis)
        rl_agent.alpha = 0.4
        rl_agent.gamma = 0.9
        rl_agent.startEpsilon = 0.05
        rl_agent.epsilon = rl_agent.startEpsilon
        rl_agent.epochs = 1_000_000  # 1M epochs, like in the thesis

        # reset tracking
        rl_agent.wins = 0
        rl_agent.losses = 0
        rl_agent.memory = []
        rl_agent.winrates = []

        jack.game(rl_agent)

        total_games = rl_agent.wins + rl_agent.losses
        overall_wr = (rl_agent.wins / total_games * 100.0) if total_games > 0 else 0.0
        results_overall.append(overall_wr)

        # tail estimate: last tracked winrate from memory (if any)
        if hasattr(rl_agent, "winrates") and rl_agent.winrates:
            tail_wr = rl_agent.winrates[-1]
        else:
            tail_wr = None
        results_tail.append(tail_wr)

        print(
            f"[RL multi] Agent {k+1}: overall winrate = {overall_wr:.4f}%"
            + (f", tail winrate ≈ {tail_wr:.4f}%" if tail_wr is not None else "")
        )

    # summary
    print("\n[RL multi] Summary over", num_agents, "agents (alg =", alg, ", pol =", pol, ")")
    if results_overall:
        mean_overall = sum(results_overall) / len(results_overall)
        print(
            f"  Overall winrate: mean = {mean_overall:.4f}%, "
            f"min = {min(results_overall):.4f}%, max = {max(results_overall):.4f}%"
        )
    if any(r is not None for r in results_tail):
        valid_tail = [r for r in results_tail if r is not None]
        if valid_tail:
            mean_tail = sum(valid_tail) / len(valid_tail)
            print(
                f"  Tail winrate (memory-based): mean = {mean_tail:.4f}%, "
                f"min = {min(valid_tail):.4f}%, max = {max(valid_tail):.4f}%"
            )


# -----------------------------
# 3. GA: Evolve full strategies
# -----------------------------

def run_ga():
    import GA
    import importlib

    importlib.reload(GA)

    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    genetics = GA.genetics()

    # === Thesis configuration (heavy, but faithful) ===
    # You can temporarily scale these down if runs are too long.
    genetics.nrAgents = 100          # number of agents in the population
    genetics.nrGens = 50             # number of generations
    genetics.epochs = 50_000         # games per agent per generation
    genetics.randomSeed = 1

    # GA-specific settings as in the paper:
    # par = 1 → Ranked parent selection
    # inher = 1 → Single point crossover
    genetics.par = 1
    genetics.inher = 1

    # Mutation chance starts at 0.05 and decays linearly over generations:
    # mutateChance = mutateChance - mutateStart / generations
    genetics.mutateChanceStart = 0.05
    genetics.mutateChance = genetics.mutateChanceStart

    print(
        "\n[GA] Running Genetic Algorithm with:\n"
        f"    nrAgents = {genetics.nrAgents}\n"
        f"    nrGens   = {genetics.nrGens}\n"
        f"    epochs   = {genetics.epochs} per agent per generation\n"
    )

    best_agent = genetics.GA()
    print("[GA] GA complete.\n")

    if hasattr(best_agent, "winrate"):
        print(f"[GA] Best agent winrate: {best_agent.winrate}")
    else:
        print("[GA] GA returned an agent, but it has no 'winrate' attribute.")

    print("[GA] Check for GA learning curve PNGs and tables in the project directory.")


# -----------------------------
# 4. Hybrid GA over RL hyperparameters
# -----------------------------

def run_ga_rl_hyperparams():
    import GA_RL_hp
    import importlib

    importlib.reload(GA_RL_hp)

    print("\n[Hybrid GA→RL] Running Genetic Algorithm over RL hyperparameters...\n")
    best = GA_RL_hp.run_ga_over_rl(
        pop_size=20,
        generations=10,
        episodes_per_agent=100_000
    )

    print("\n[Hybrid GA→RL] Best hyperparameters found:")
    print(f"  fitness    = {best.fitness:.4f}%")
    print(f"  alpha      = {best.alpha:.4f}")
    print(f"  gamma      = {best.gamma:.4f}")
    print(f"  eps_start  = {best.eps_start:.4f}")
    print(f"  alg        = {best.alg} (1=Q-learning, 3=QV-learning)")

    # Optional: save to file
    with open("GA_RL_hp_best.txt", "w") as f:
        f.write("Best RL Hyperparameters Found by GA:\n")
        f.write(f"alpha={best.alpha}\n")
        f.write(f"gamma={best.gamma}\n")
        f.write(f"epsilon_start={best.eps_start}\n")
        f.write(f"algorithm={best.alg}\n")
        f.write(f"fitness={best.fitness}\n")

    print("\n[Hybrid GA→RL] Saved results to GA_RL_hp_best.txt")


def ask_int(prompt, valid_options):
    """Ask for an int until the user gives one in valid_options."""
    while True:
        try:
            val = int(input(prompt))
            if val in valid_options:
                return val
            print(f"Please enter one of: {sorted(valid_options)}")
        except ValueError:
            print("Please enter a valid integer.")


# -----------------------------
# 5. Main entry
# -----------------------------

def main():
    setup_project()

    # Import core modules after patching, so we see patched code
    import agent
    import agentRL
    import agentGA
    import blackjack
    import GA
    import jack
    import GA_RL_hp

    # Reload them just to be sure
    importlib.reload(agent)
    importlib.reload(agentRL)
    importlib.reload(agentGA)
    importlib.reload(blackjack)
    importlib.reload(GA)
    importlib.reload(jack)
    importlib.reload(GA_RL_hp)

    print("\n[MAIN] Modules imported successfully.")

    # Simple text menu
    while True:
        print("\n=== Blackjack RL–GA Experiment Menu ===")
        print("1: Run single-agent RL (Q-learning)")
        print("2: Run multi-agent RL (Q-learning)")
        print("3: Run multi-agent RL (QV-learning)")
        print("4: Run GA over full policies")
        print("5: Run GA over RL hyperparameters (hybrid)")
        print("0: Exit")
        choice = ask_int("Select an option: ", {0, 1, 2, 3, 4, 5})

        if choice == 0:
            print("\n[MAIN] Exiting. All done.")
            break

        elif choice == 1:
            print("\n[MAIN] Option 1 selected: single-agent RL (Q-learning).")
            run_rl()

        elif choice == 2:
            print("\n[MAIN] Option 2 selected: multi-agent RL (Q-learning).")
            # You can tweak num_agents if desired
            run_rl_multi(num_agents=10, alg=1, pol=2)

        elif choice == 3:
            print("\n[MAIN] Option 3 selected: multi-agent RL (QV-learning).")
            run_rl_multi(num_agents=10, alg=3, pol=2)

        elif choice == 4:
            print("\n[MAIN] Option 4 selected: GA over full policies.")
            run_ga()

        elif choice == 5:
            print("\n[MAIN] Option 5 selected: GA over RL hyperparameters (hybrid).")
            run_ga_rl_hyperparams()



if __name__ == "__main__":
    main()

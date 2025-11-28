"""
GA_RL_hp.py

Genetic Algorithm over RL hyperparameters (hybrid model).

Each GA individual encodes a set of hyperparameters for AgentRL:
    - alpha (learning rate)
    - gamma (discount factor)
    - startEpsilon (initial epsilon for epsilon-greedy)
    - alg (1 = Q-learning, 3 = QV-learning)

For each individual:
    1. We create an AgentRL with these hyperparameters.
    2. Train it for EPISODES_PER_AGENT episodes using jack.game(agent).
    3. Use the resulting winrate as the fitness of the individual.

This lets the GA search for good RL hyperparameters, instead of evolving
full policy tables directly.
"""

import random
import math
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import importlib

import agentRL
import jack

# Reload to ensure freshness
importlib.reload(agentRL)
importlib.reload(jack)

# GA CONFIG
POP_SIZE = 20
GENERATIONS = 10
TOURNAMENT_SIZE = 3
ELITISM = 2

# RL training budget
EPISODES_PER_AGENT = 100_000

# RL hyperparameter bounds
ALPHA_MIN, ALPHA_MAX = 0.1, 0.6
GAMMA_MIN, GAMMA_MAX = 0.7, 0.99
EPS_MIN, EPS_MAX     = 0.01, 0.2

# Mutation settings
MUTATION_PROB = 0.3
MUTATION_STD_FRAC = 0.15
ALG_MUTATION_PROB = 0.1


@dataclass
class Individual:
    alpha: float
    gamma: float
    eps_start: float
    alg: int  # 1 = Q learning, 3 = QV learning

    fitness: Optional[float] = field(default=None, compare=False)

    def clone(self):
        return copy.deepcopy(self)


# ------------------------------------------------------
# CREATE RANDOM INDIVIDUAL
# ------------------------------------------------------
def random_individual():
    return Individual(
        alpha=random.uniform(ALPHA_MIN, ALPHA_MAX),
        gamma=random.uniform(GAMMA_MIN, GAMMA_MAX),
        eps_start=random.uniform(EPS_MIN, EPS_MAX),
        alg=random.choice([1, 3])
    )


def clip(val, lo, hi):
    return max(lo, min(hi, val))


# ------------------------------------------------------
# MUTATION
# ------------------------------------------------------
def mutate(ind: Individual) -> Individual:
    child = ind.clone()

    if random.random() < MUTATION_PROB:
        std = (ALPHA_MAX - ALPHA_MIN) * MUTATION_STD_FRAC
        child.alpha = clip(child.alpha + random.gauss(0, std), ALPHA_MIN, ALPHA_MAX)

    if random.random() < MUTATION_PROB:
        std = (GAMMA_MAX - GAMMA_MIN) * MUTATION_STD_FRAC
        child.gamma = clip(child.gamma + random.gauss(0, std), GAMMA_MIN, GAMMA_MAX)

    if random.random() < MUTATION_PROB:
        std = (EPS_MAX - EPS_MIN) * MUTATION_STD_FRAC
        child.eps_start = clip(child.eps_start + random.gauss(0, std), EPS_MIN, EPS_MAX)

    if random.random() < ALG_MUTATION_PROB:
        child.alg = 1 if child.alg == 3 else 3

    child.fitness = None
    return child


# ------------------------------------------------------
# CROSSOVER
# ------------------------------------------------------
def crossover(p1: Individual, p2: Individual) -> Individual:
    return Individual(
        alpha=p1.alpha if random.random() < 0.5 else p2.alpha,
        gamma=p1.gamma if random.random() < 0.5 else p2.gamma,
        eps_start=p1.eps_start if random.random() < 0.5 else p2.eps_start,
        alg=p1.alg if random.random() < 0.5 else p2.alg
    )


# ------------------------------------------------------
# EVALUATION (RUN RL TRAINING)
# ------------------------------------------------------
def evaluate_individual(ind: Individual, seed=1) -> float:
    if ind.fitness is not None:
        return ind.fitness

    random.seed(seed)
    np.random.seed(seed)

    agent = agentRL.AgentRL()
    agent.alg = ind.alg
    agent.pol = 2  # epsilon-greedy

    agent.alpha = ind.alpha
    agent.gamma = ind.gamma
    agent.startEpsilon = ind.eps_start
    agent.epsilon = ind.eps_start
    agent.epochs = EPISODES_PER_AGENT

    agent.wins = 0
    agent.losses = 0
    agent.memory = []
    agent.winrates = []

    jack.game(agent)

    total = agent.wins + agent.losses
    overall = (agent.wins / total * 100) if total > 0 else 0

    if agent.winrates:
        ind.fitness = agent.winrates[-1]
    else:
        ind.fitness = overall

    return ind.fitness


# ------------------------------------------------------
# SELECTION
# ------------------------------------------------------
def tournament_select(pop):
    cand = random.sample(pop, TOURNAMENT_SIZE)
    cand.sort(key=lambda x: x.fitness, reverse=True)
    return cand[0]


# ------------------------------------------------------
# MAIN GA LOOP
# ------------------------------------------------------
def run_ga_over_rl(pop_size=POP_SIZE, generations=GENERATIONS,
                   episodes_per_agent=EPISODES_PER_AGENT,
                   base_seed=42):

    global EPISODES_PER_AGENT
    EPISODES_PER_AGENT = episodes_per_agent

    print(f"[GA_RL_hp] Starting GA (pop={pop_size}, gens={generations})")

    population = [random_individual() for _ in range(pop_size)]
    best_overall = None

    for gen in range(generations):
        print(f"\n[GA_RL_hp] Generation {gen+1}/{generations}")

        # Evaluate
        for i, ind in enumerate(population):
            if ind.fitness is None:
                seed = base_seed + i + gen * pop_size
                fitness = evaluate_individual(ind, seed)
                print(
                    f"  Ind {i:02d}: fitness={fitness:.4f}% "
                    f"(alpha={ind.alpha:.3f}, gamma={ind.gamma:.3f}, eps={ind.eps_start:.3f}, alg={ind.alg})"
                )

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Track best ever
        if best_overall is None or population[0].fitness > best_overall.fitness:
            best_overall = population[0].clone()

        print(
            f"  -> Best this gen: {population[0].fitness:.4f}% "
            f"(alpha={population[0].alpha:.3f}, gamma={population[0].gamma:.3f}, eps={population[0].eps_start:.3f}, alg={population[0].alg})"
        )
        print(
            f"  -> Best overall: {best_overall.fitness:.4f}% "
            f"(alpha={best_overall.alpha:.3f}, gamma={best_overall.gamma:.3f}, eps={best_overall.eps_start:.3f}, alg={best_overall.alg})"
        )

        # --- Next generation ---
        new_pop = []

        # Elitism
        elites = [population[i].clone() for i in range(min(ELITISM, pop_size))]
        for e in elites:
            e.fitness = None
        new_pop.extend(elites)

        # Fill rest
        while len(new_pop) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    # Ensure best has fitness
    if best_overall.fitness is None:
        evaluate_individual(best_overall)

    print("\n[GA_RL_hp] Complete.")
    print(
        f"Best: fitness={best_overall.fitness:.4f}%, "
        f"alpha={best_overall.alpha:.4f}, gamma={best_overall.gamma:.4f}, "
        f"eps_start={best_overall.eps_start:.4f}, alg={best_overall.alg}"
    )

    return best_overall


if __name__ == "__main__":
    run_ga_over_rl(pop_size=10, generations=5, episodes_per_agent=50_000)

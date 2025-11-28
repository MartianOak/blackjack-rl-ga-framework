import random
import numpy as np
import math
import matplotlib.pyplot as plt
import agent
import agentGA
from numpy import argmax

class AgentPSO (agentGA.AgentGA):

    def __init__(self):
        super().__init__()
        self.alg = 5
        total_dims = sum(len(self.states[i][j]) for i in range(len(self.states)) for j in range(len(self.states[0]))) + \
            sum(len(self.splitStates[i][j]) for i in range(len(self.splitStates)) for j in range(len(self.splitStates[0])))
        self.v_vec = [random.uniform(-0.5,0.5) for x in range(total_dims)]
        self.p_vec = [0 for x in range(total_dims)]
        self.p_best = list(self.p_vec)
        self.win_best = 0

    def update(self, g_best, vel_const, p_lowerBounds, p_upperBounds):
        #update best personal position
        if self.winrate > self.win_best:
            self.win_best = self.winrate
            self.p_best = list(self.p_vec)

        #update velocity vector
        rand1 = random.random()
        rand2 = random.random()
        for i in range(len(self.v_vec)):
            self.v_vec[i] += vel_const * rand1 * (self.p_best[i] - self.p_vec[i]) + vel_const * rand2 * (g_best[i] - self.p_vec[i])
        #update position
        for i in range(len(self.p_vec)):
            self.p_vec[i] += self.v_vec[i]
        #check if within boundaries, using absorbing (80% dampening) + reflecting wall
        for i in range(len(self.p_vec)):
            if self.p_vec[i] < p_lowerBounds[i]:
                dif = self.p_vec[i] - p_lowerBounds[i]
                self.p_vec[i] -= 1.2 * dif
            if self.p_vec[i] > p_upperBounds[i]:
                dif = self.p_vec[i] - p_upperBounds[i]
                self.p_vec[i] -= 1.2 * dif
        #print(self.v_vec)
        self.convertToStates(p_upperBounds)

    def randomInit(self, p_upperBounds, p_lowerBounds):
        #initialize a random position in the search space
        for i in range(len(self.p_vec)):
            self.p_vec[i] = random.uniform(p_lowerBounds[i], p_upperBounds[i])

    def convertToStates(self, p_upperBounds):
        # Map the flat position vector back to discrete actions for each state/split-state
        idx = 0
        for i in range(len(self.states)):
            for j in range(len(self.states[i])):
                max_idx = len(self.states[i][j]) - 1
                rounded = int(round(self.p_vec[idx]))
                ub = p_upperBounds[idx] if idx < len(p_upperBounds) else max_idx + 0.5
                if rounded > ub - 0.5:
                    rounded = max_idx
                if rounded < -0.5:
                    rounded = 0
                for m in range(len(self.states[i][j])):
                    self.states[i][j][m] = 1 if m == rounded else 0
                idx += 1
        for k in range(len(self.splitStates)):
            for l in range(len(self.splitStates[k])):
                max_idx = len(self.splitStates[k][l]) - 1
                rounded = int(round(self.p_vec[idx]))
                ub = p_upperBounds[idx] if idx < len(p_upperBounds) else max_idx + 0.5
                if rounded > ub - 0.5:
                    rounded = max_idx
                if rounded < -0.5:
                    rounded = 0
                for n in range(len(self.splitStates[k][l])):
                    self.splitStates[k][l][n] = 1 if n == rounded else 0
                idx += 1

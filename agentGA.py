import random
import numpy as np
import math
import matplotlib.pyplot as plt
import agent
from numpy import argmax

class AgentGA(agent.Agent):

    def __init__(self):
        super().__init__()
        self.alg = 2
        self.winrate = 0

    def reset(self):
        self.wins = 0
        self.naturalWins = 0
        self.losses = 0
        self.winrate = 0

    def mutate(self, mutateChance):
        #add some random mutations based on the mutationchance
        for i in range(len(self.states)):
            for j in range(len(self.states[i])):
                #mutate regular states
                mutate = random.random()
                if mutate < mutateChance:
                    z = random.randint(0,len(self.states[i][j]) - 1)
                    for k in range(len(self.states[i][j])):
                        if k == z:
                            self.states[i][j][k] = 1
                        else:
                            self.states[i][j][k] = 0
                #mutation for splitstates
                if j < len(self.splitStates[i]):
                    mutate = random.random()
                    if mutate < mutateChance:
                        z = random.randint(0,len(self.splitStates[i][j]) - 1)
                        for k in range(len(self.splitStates[i][j])):
                            if k == z:
                                self.splitStates[i][j][k] = 1
                            else:
                                self.splitStates[i][j][k] = 0

    def initializeRandom(self):
        #set one random action at each state to 1 (choosing the policy to follow)
        for i in range(len(self.states)):
            for j in range(len(self.states[i])):
                x = random.random()
                val = 1 / len(self.states[i][j])
                val2 = 0
                for k in range(len(self.states[i][j])):
                    val2 += val
                    if x < val2:
                        self.states[i][j][k] = 1
                        break
        
        #do the same for splitstates
        for i in range(len(self.splitStates)):
            for j in range(len(self.splitStates[i])):
                x = random.random()
                val = 1 / len(self.splitStates[i][j])
                val2 = 0
                for k in range(len(self.splitStates[i][j])):
                    val2 += val
                    if x < val2:
                        self.splitStates[i][j][k] = 1
                        break

    #return the action which has a 1 for that state
    def getActionGreedy(self, playerHand, dealerHand):
        if playerHand.size == 2 and playerHand[0] == playerHand[1]:
            y = argmax(self.splitStates[dealerHand][playerHand[0]])
            if y == 2:
                return 3
            elif y == 3:
                return 2
            else:
                return y
        return argmax(self.states[dealerHand][np.sum(playerHand)])

    def calcWinrate(self):
        if self.wins == 0:
            self.winrate = 0
        else:
            self.winrate = (self.wins / (self.losses + self.wins)) * 100

    def printResults(self):
        print("------------| Results |------------")
        print("Agent's wins: " + str(self.wins))
        print("Agent's natural wins: " + str(self.naturalWins))
        print("Agent's losses: " + str(self.losses))
        if self.wins + self.losses == 0:
            print("Agent's winrate: Undefined, no games played.")
        else:
            print("Agent's winrate: " + str(self.wins / (self.wins + self.losses) * 100) + "%")
        print("Agent's splits: " + str(self.splits))
        print("Agent's games: " + str(self.wins + self.losses))
        print("Natural games: " + str(self.wins + self.losses - self.splits))
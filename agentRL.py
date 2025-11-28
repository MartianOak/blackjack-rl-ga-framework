import random
import numpy as np
import math
import matplotlib.pyplot as plt
import agent
from numpy import argmax

class AgentRL(agent.Agent):

    def __init__(self):
        super().__init__()
        #Fill states Value array with empty state/action pairs: [dealer hand[own hand]]
        self.vStates = [[0 for x in range(22)] for y in range(12)]
        #State Value array for the hands in which you are allowed to split. [dealer hand[own pair of cards]]
        self.vSplitStates = [[0 for x in range(12)] for y in range(12)]
        self.memory = []
        self.winrates = []
        self.rewards = [-0.5, 2, 4]
        self.alpha = 0.4
        self.gamma = 0.9
        self.startEpsilon = 0.05
        self.pol = 2
        self.epsilon = self.startEpsilon
        
    def reset(self):
        # Fill states array with empty state/action pairs: [dealer hand[own hand[hit, stand]]]
        self.states = [[[0 for x in range(2)] for y in range(22)] for z in range(12)]
        #add double down for playerHand values of 9-11.
        for i in range (9,12):
            for j in range(12):
                self.states[j][i].append(0)
        #state array for the hands in which you are allowed to split. [dealer hand[own pair of cards][hit, stand, split]]
        self.splitStates = [[[0 for x in range(3)] for y in range(12)] for z in range(12)]
        #add double down, only for a pair of 5's
        for i in range(12):
            self.splitStates[i][5].append(0)
        #Fill states Value array with empty state/action pairs: [dealer hand[own hand]]
        self.vStates = [[0 for x in range(22)] for y in range(12)]
        #State Value array for the hands in which you are allowed to split. [dealer hand[own pair of cards]]
        self.vSplitStates = [[0 for x in range(12)] for y in range(12)]

        #rewards for loss, continue play, win
        self.memory = []
        self.winrates = []
        self.wins = 0
        self.naturalWins = 0
        self.losses = 0
        self.epsilon = self.startEpsilon

    def printResults(self):
        print("------------| Settings|------------")
        print("Random seed: " + str(self.randomSeed))
        print("Epsilon: " + str(self.startEpsilon))
        print("Learning rate (alpha): " + str(self.alpha))
        print("Discount factor (gamma): " + str(self.gamma))
        print("Reward for loss: " + str(self.rewards[0]))
        print("Reward for hit without bust: " + str(self.rewards[1]))
        print("Reward for win: " + str(self.rewards[2]))
        print("------------| Results |------------")
        print("Agent's wins: " + str(self.wins))
        print("Agent's natural wins: " + str(self.naturalWins))
        print("Agent's losses: " + str(self.losses))
        print("Agent's Money gain: " + str(self.money))
        if self.wins + self.losses == 0:
            print("Agent's winrate: Undefined, no games played.")
        else:
            print("Agent's winrate: " + str(self.wins / (self.wins + self.losses) * 100) + "%")

    #plot the learning curve
    def plotWinrates(self):
        plot1 = plt.figure()
        #smoothing
        box = np.ones(5)/5
        winratesSmooth = np.convolve(self.winrates, box, mode='valid')
        #adding the data to the plot
        line1 = plt.plot(winratesSmooth, "r-", label="raw winrates")
        plt.xlim([0,len(winratesSmooth) + 5])
        plt.ylim([32,44])
        plt.ylabel("Winrate over the last 50000 games played (%)")
        plt.xlabel("Amount of games played * 100")
        plt.title("Learning curve for blackjack using Q-learning")
        plt.grid(True)
        plt.savefig("learningcurve.png")
        plt.xlim(0,len(winratesSmooth) / 10 + 0.5)
        plt.savefig("learningcurvestart.png")
        plt.close()

    #function to save the current game to the agents memory to make a learning curve graph
    def addGameToMemory(self, game, epoch):
        self.memory.append(game)
        if epoch % 100 == 0:
            if len(self.memory) > 100:
                self.winrates.append((self.memory.count(1) / len(self.memory)) * 100)
        if len(self.memory) > 100000:
            self.memory.pop(0)

    #function to update the Q value of the choose state-action pair
    def updateStates(self, playerHand, action, dealerHand, newState, reward):
        # dealers hand 0-11 for each player hand from 0-21 (value of card hand) and action range from 0-2 (hit-stand-double down)
        new = 0
        if self.alg == 1: #Q-learning
            new = max(self.states[dealerHand][newState])
        if self.alg == 3: #QV-learning
            delta = reward + (self.gamma * self.vStates[dealerHand][newState]) - self.vStates[dealerHand][playerHand]
            self.vStates[dealerHand][playerHand] += (self.alpha * delta)
            new = self.vStates[dealerHand][newState]
        if self.alg == 4: #Monte Carlo
            new = self.vStates[dealerHand][playerHand]
            self.vStates[dealerHand][playerHand] += self.alpha * (reward - new)
        if self.alg == 1 or self.alg == 3 or self.alg == 4: #Q-learning and QV-learning and Monte Carlo
            #print("update")
            #print(dealerHand)
            #print(playerHand)
            #print(action)
            self.states[dealerHand][playerHand][action] += self.alpha * (reward + (self.gamma * new) - self.states[dealerHand][playerHand][action])
                    
    def updateSplitStates(self, playerHand, action, dealerHand, newState, reward):

        # Clamp newState to 0–11 because original table is 12 columns wide
        if newState > 11:
            newState = 11
        if newState < 0:
            newState = 0
    
        action = min(action, len(self.splitStates[dealerHand][playerHand]) - 1)
    
        if self.alg == 1:  # Q-learning
            new = max(self.states[dealerHand][newState])
    
        elif self.alg == 3:  # QV-learning
            delta = reward + (self.gamma * self.vSplitStates[dealerHand][newState]) - self.vSplitStates[dealerHand][playerHand]
            self.vSplitStates[dealerHand][playerHand] += self.alpha * delta
            new = self.vSplitStates[dealerHand][newState]
    
        elif self.alg == 4:  # Monte Carlo
            new = self.vSplitStates[dealerHand][playerHand]
            self.vSplitStates[dealerHand][playerHand] += self.alpha * (reward - new)
    
        self.splitStates[dealerHand][playerHand][action] += (
            self.alpha * (reward + self.gamma * new - self.splitStates[dealerHand][playerHand][action])
        )

    def getActionGreedy(self, playerHand, dealerHand):
        if playerHand.size == 2 and playerHand[0] == playerHand[1]:
            #allowed to split
            z = argmax(self.splitStates[dealerHand][playerHand[0]])
            if z == 2:
                return 3
            elif z == 3:
                return 2
            else:
                return z
        return argmax(self.states[dealerHand][np.sum(playerHand)])

    #returns the action with the highest Q value with a probability of 1 - epsilon, otherwise randomly choses an action
    def getActionEGreedy(self, playerHand, dealerHand):
        if random.random() < self.epsilon:
            #random action
            if playerHand.size == 2 and playerHand[0] == playerHand[1]:
                y = random.randint(0,len(self.splitStates[dealerHand][playerHand[0]]) - 1)
                if y == 2:
                    return 3
                elif y == 3:
                    return 2
                else:
                    return y
            return random.randint(0,len(self.states[dealerHand][np.sum(playerHand)]) - 1)
        else:
            #choose greedily
            return self.getActionGreedy(playerHand, dealerHand)

    #returns the action based on the softmax policy
    def getActionSoftmax(self, playerHand, dealerHand):
        #convert to probabilities
        agentPrefs = []
        denum = 0
        if playerHand.size == 2 and playerHand[0] == playerHand[1]:
            for i in range(len(self.splitStates[dealerHand][playerHand[0]])):
                denum += math.exp(self.splitStates[dealerHand][playerHand[0]][i])
            for i in range(len(self.splitStates[dealerHand][playerHand[0]])):
                num = math.exp(self.splitStates[dealerHand][playerHand[0]][i])
                agentPrefs.append(num/denum)
        else:
            for i in range(len(self.states[dealerHand][np.sum(playerHand)])):
                denum += math.exp(self.states[dealerHand][np.sum(playerHand)][i])
            for i in range(len(self.states[dealerHand][np.sum(playerHand)])):
                num = math.exp(self.states[dealerHand][np.sum(playerHand)][i])
                agentPrefs.append(num/denum)

        #choose action based on probabilties
        randomFloat = random.random()
        index = 0
        for i in range(len(agentPrefs)):
            index += agentPrefs[i]
            if (index >= randomFloat):
                if playerHand.size == 2 and playerHand[0] == playerHand[1]:
                    if i == 2:
                        return 3
                    elif i == 3:
                        return 2
                return i
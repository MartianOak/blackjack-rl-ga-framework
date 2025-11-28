import random
import agentGA
import jack
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

def winrate(agent):
    return agent.winrate

class genetics():

    def __init__(self):
        self.nrAgents = 30
        self.nrGens = 50
        self.par = 1
        self.inher = 1
        self.agentList = []
        self.epochs = 100000
        self.randomSeed = 1
        self.mutateChanceStart = 0.05
        self.mutateChance = self.mutateChanceStart
        self.bestWinrates = []
        self.allWinrates = []
        self.sumRank = 0

    def setPoliciesRanked(self, parent1, parent2, index1, index2):
        ranks = index1 + index2
        p1chance = index2 / ranks
        p2chance = index1 / ranks

        children = []
        for i in range(2):
            children.append(agentGA.AgentGA())

        #inherit regular states
        for child in range(2):
            for i in range(len(parent1.states)):
                for j in range(len(parent1.states[1])):
                    x = random.random()
                    if x < p1chance:
                        children[child].states[i][j] = parent1.states[i][j]
                    else:
                        children[child].states[i][j] = parent2.states[i][j]

        #inherit splitstates
        for child in range(2):
            for i in range(len(parent1.splitStates)):
                for j in range(len(parent1.splitStates[1])):
                    x = random.random()
                    if x < p1chance:
                        children[child].splitStates[i][j] = parent1.splitStates[i][j]
                    else:
                        children[child].splitStates[i][j] = parent2.splitStates[i][j]

        #return both children
        return children

    #inheritting policies from parents based on a single crossover point
    def setPoliciesCrossover(self, parent1, parent2):
        #create random crossover point
        c = [0,0,0]
        c[0] = random.randint(2,11)
        c[1] = random.randint(4,21)
        c[2] = random.randint(2,10)

        children = []
        for i in range(2):
            children.append(agentGA.AgentGA())

        #inherit regular states
        for child in range(2):
            for i in range(len(parent1.states)):
                for j in range(len(parent1.states[1])):
                    #before crossoverpoint
                    if j < c[1] or (j == c[1] and i < c[0]):
                        if child == 0:
                            children[child].states[i][j] = parent1.states[i][j]
                        elif child == 1:
                            children[child].states[i][j] = parent2.states[i][j]
                    #after crossoverpoint
                    else:
                        if child == 0:
                            children[child].states[i][j] = parent2.states[i][j]
                        elif child == 1:
                            children[child].states[i][j] = parent1.states[i][j]

        #inherit splitstates
        for child in range(2):
            for i in range(len(parent1.splitStates)):
                for j in range(len(parent1.splitStates[1])):
                    #before crossoverpoint
                    if j < c[2] or (j == c[2] and i < c[0]):
                        if child == 0:
                            children[child].splitStates[i][j] = parent1.splitStates[i][j]
                        elif child == 1:
                            children[child].splitStates[i][j] = parent2.splitStates[i][j]
                    #after crossoverpoint
                    else:
                        if child == 0:
                            children[child].splitStates[i][j] = parent2.splitStates[i][j]
                        elif child == 1:
                            children[child].splitStates[i][j] = parent1.splitStates[i][j]

        #return both children
        return children

    def getParentsRanked(self):
        tempAgentList = []
        for i in range(int(self.nrAgents / 2)):
            #select random parents based on ranks
            p1index = None
            p2index = None
            x = random.random()
            y = random.random()
            z = 0
            for i in range(self.nrAgents):
                z += ((i + 1) / self.sumRank)
                if x < z and p1index == None:
                    p1index = i
                if y < z and p2index == None:
                    p2index = i

            #create children based on selected inheritance
            #create 2 offspring per 2 parents
            if self.inher == 1:
                tempAgentList.extend(self.setPoliciesCrossover(self.agentList[p1index], self.agentList[p2index]))
            elif self.inher == 2:
                tempAgentList.extend(self.setPoliciesRanked(self.agentList[p1index], self.agentList[p2index], p1index, p2index))
        return tempAgentList

    def getParentsTourney(self):
        tempAgentList = []
        for i in range(int(self.nrAgents / 2)):
            #select 4 random parents and check which are better
            p1index = None
            p2index = None
            option1 = random.randint(0, self.nrAgents - 1)
            option2 = random.randint(0, self.nrAgents - 1)
            option3 = random.randint(0, self.nrAgents - 1)
            option4 = random.randint(0, self.nrAgents - 1)

            #print(len(self.agentList))
            #print(option1)
            #print(option2)
            if self.agentList[option1].winrate > self.agentList[option2].winrate:
                p1index = option1
            else:
                p1index = option2

            if self.agentList[option3].winrate > self.agentList[option4].winrate:
                p2index = option3
            else:
                p2index = option4

            #create children based on selected inheritance
            #create 2 offspring per 2 parents
            if self.inher == 1:
                tempAgentList.extend(self.setPoliciesCrossover(self.agentList[p1index], self.agentList[p2index]))
            elif self.inher == 2:
                tempAgentList.extend(self.setPoliciesRanked(self.agentList[p1index], self.agentList[p2index], p1index, p2index))
        return tempAgentList

    def fitnessFunction(self, gen):
        #sum of all ranks for parenting chance using rank selection
        for i in range(self.nrAgents):
            self.agentList[i].calcWinrate()
            self.allWinrates[gen].append(self.agentList[i].winrate)
        #sorting list of agents based on winrate
        self.agentList.sort(key=winrate)
        #print the best agent to terminal and create tables
        print("Gen " + str(gen) +" Best Agent: " + str("{:.2f}".format(self.agentList[self.nrAgents - 1].winrate)) + "%")
        self.bestWinrates.append(self.agentList[self.nrAgents - 1].winrate)
        self.agentList[self.nrAgents - 1].createSplitTable(gen, "GA", "")
        self.agentList[self.nrAgents - 1].createTable(gen, "GA", "")

        #select parents based on selected policy
        if self.par == 1:
            tempAgentList = self.getParentsRanked()
        elif self.par == 2:
            tempAgentList = self.getParentsTourney()

        #create random small mutations
        for agent in tempAgentList:
            agent.mutate(self.mutateChance)
        
        return tempAgentList

    def GA(self):
        #initialize the agents
        for i in range(self.nrAgents):
            self.agentList.append(agentGA.AgentGA())
            self.agentList[i].initializeRandom()
            self.agentList[i].epochs = self.epochs
            self.agentList[i].randomSeed = self.randomSeed
            self.sumRank += (i + 1)
        for i in range(self.nrGens):
            self.allWinrates.append([])

        #run for set amount of generations
        print("Using " + str(max(1, mp.cpu_count() - 1)) + " cores out of a total of " + str(mp.cpu_count()) + " present on this machine.")
        print("Running for " + str(self.nrGens) + " generations with " + str(self.nrAgents) + " agents each with " + str(self.epochs) + " epochs each.")
        for i in range(self.nrGens):
            print("Training Gen: " + str(i), end="\r")
            if i == self.nrGens - 1:
                break
            #----------------multiprocessing------------------
            #using the amount of cpu cores available on this machine - 2 (to ensure other processes can run simultaniously)
            with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
                self.agentList = p.map(jack.game, self.agentList)
            #-------------------------------------------------

            #create offspring based on fitness of parents
            self.agentList = self.fitnessFunction(i)

            #reset all agents wins and losses
            for j in range(self.nrAgents):
                self.agentList[j].reset()
            
            self.mutateChance -= (self.mutateChanceStart / self.nrGens)
        
        #find the best agent after final generation
        print("Testing the agents of the last generation with " + str(self.epochs * 10) + " epochs each.")
        for agent in self.agentList:
            agent.epochs *= 1

        with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
                self.agentList = p.map(jack.game, self.agentList)

        for i in range(self.nrAgents):
            self.agentList[i].calcWinrate()
            self.allWinrates[self.nrGens-1].append(self.agentList[i].winrate)
        self.agentList.sort(key=winrate, reverse=True)
        #return the best agent of the last generation as result
        self.bestWinrates.append(self.agentList[0].winrate)
        self.agentList[0].printResults()
        self.agentList[0].createSplitTable(i, "GA", "")
        self.agentList[0].createTable(i, "GA", "")
        self.createLearningCurve()
        return self.agentList[0]
    
    def createLearningCurve(self):
        x = np.arange(1,len(self.bestWinrates)+1,1)
        averageWinrate = []
        minWinrate = []
        for i in range(len(self.allWinrates)):
            minWinrate.append(min(self.allWinrates[i]))
            #print(self.allWinrates[i])
            averageWinrate.append(sum(self.allWinrates[i]) / len(self.allWinrates[i]))
            
        #print(minWinrate)
        #print(averageWinrate)
        parName = 'Undefined'
        inhername = 'Undefined'
        if self.par == 1:
            parName = 'Ranked'
        if self.par == 2:
            parName = 'Tourney'
        if self.inher == 1:
            inherName = 'Crossover'
        if self.inher == 2:
            inherName = 'Ranked'
        line1, = plt.plot(x, minWinrate, color="black", label="Worst")    
        line2, = plt.plot(x, averageWinrate, color="blue", label="Average")    
        line3, = plt.plot(x, self.bestWinrates, color="red", label="Best")
        plt.legend(handles=[line3, line2, line1], loc=4)
        plt.ylim([25,45])
        plt.xlim([1,len(self.bestWinrates)])
        plt.ylabel("Winrate (%)")
        plt.xlabel("Generation")
        plt.title("Learning curve of a GA playing blackjack (" + str(self.epochs) + " epochs per gen)")
        plt.grid(True)
        plt.savefig("learningcurveGA" + parName + inherName + ".png")
        plt.close()

    def reset(self):
        self.agentList = []
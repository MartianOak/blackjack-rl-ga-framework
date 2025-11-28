import random
import numpy as np
import agent
import agentRL
import blackjack
import agentGA
import GA
import PSO
import saveFile
import statistics
from time import time
import matplotlib.pyplot as plt
import multiprocessing as mp


def createLCplot(learningCurve, alg, pol):
    # calculate average winrate as well as the max and min over all agents
    lowestLen = -1
    for i in range(len(learningCurve)):
        if lowestLen == -1:
            lowestLen = len(learningCurve[i])
        if len(learningCurve[i]) < lowestLen:
            lowestLen = len(learningCurve[i])
    maxWR = []
    averageWR = []
    minWR = []
    total = maxs = 0
    mins = 101
    dev = []
    sd = []
    for i in range(lowestLen):
        for j in range(len(learningCurve)):
            if learningCurve[j][i] < mins: mins = learningCurve[j][i]
            if learningCurve[j][i] > maxs: maxs = learningCurve[j][i]
            total += learningCurve[j][i]
            dev.append(learningCurve[j][i])
        sd.append(statistics.stdev(dev))
        averageWR.append(total / len(learningCurve))
        maxWR.append(maxs)
        minWR.append(mins)
        maxs = total = 0
        mins = 101
        dev = []

    sdmin = []
    sdplus = []
    for i in range(lowestLen):
        sums = averageWR[i] - sd[i]
        sdmin.append(sums)
        sums = averageWR[i] + sd[i]
        sdplus.append(sums)
    #plot the results
    x = np.arange(100,(lowestLen * 100) + 100,100)
    line1, = plt.plot(x, maxWR, color="#f0b6a9", label="Min/Max")
    plt.plot(x, minWR, color="#f0b6a9")
    plt.fill_between(x, maxWR, minWR, color="#f0b6a9")
    line2, = plt.plot(x, sdplus, color="#ec8c77", label="Deviation")
    plt.plot(x, sdmin, color="#ec8c77")
    plt.fill_between(x, sdplus, sdmin, color="#ec8c77")
    line3, = plt.plot(x, averageWR, color="#ee3d18", label="Average learning curve")
    plt.legend(handles=[line3, line2, line1], loc=4)
    plt.ylim([30,44])
    plt.ylabel("Winrate (%)")
    plt.xlabel("Games played")
    algName = "Undefined"
    polName = "Undefined"
    if pol == 1:
        polName = "Greedy"
    if pol == 2:
        polName = "e-Greedy"
    if pol == 3:
        polName = "Softmax"
    if alg == 1:
        algName = "Q-learning"
    if alg == 3:
        algName = "QV-learning"
    plt.title("Average learning curve using " + str(len(learningCurve)) + " agents with " + algName + "(" + polName + ")")
    plt.grid(True)
    plt.savefig("learningCurveAverage" + algName + polName + ".png")
    plt.close()

def game(agent, gen = 0):
    #4 suits in 1 deck, 6 decks in 1 stack, then shuffle. keep playing games untill out of cards then reshuffle the full stack of cards.
    suit = [2,3,4,5,6,7,8,9,10,10,10,10,11]
    deck = suit + suit + suit + suit
    stack = deck + deck + deck + deck + deck + deck
    random.shuffle(stack)

    #the game
    for i in range (agent.epochs):
        if agent.alg != 2 and agent.alg != 5 and agent.alg != 6:  # no need for GA/PSO
            # update epsilon over time to converge to 0 (RL only)
            if agent.epochs > 0 and i < agent.epochs:
                agent.epsilon -= agent.startEpsilon / float(agent.epochs)
                if agent.epsilon < 0.0:
                    agent.epsilon = 0.0
            #print progress
            if i % 100 == 0:
                print("Simulating " + str(agent.epochs) + " games... " + str("{:.1f}".format(i / agent.epochs * 100)) + "% | epsilon: " + str("{:.9f}".format(agent.epsilon)), end="\r")

        #reshuffle deck if not enough cards left
        if len(stack) < 70:
            stack = deck + deck + deck + deck + deck + deck
            random.shuffle(stack)

        #player and dealer draw cards
        playerHand = np.array([stack.pop(0), stack.pop(0)])
        dealerHand = np.array([stack.pop(0), stack.pop(0)])

        #play the game with these hands
        blackjack.blackjack(stack, agent, i, playerHand, dealerHand)

    if agent.alg != 2 and agent.alg != 5 and agent.alg != 6: #dont print results of every agent in GA
        algName = "Undefined"
        polName = "Undefined"
        if agent.pol == 1:
            polName = "Greedy"
        if agent.pol == 2:
            polName = "e-Greedy"
        if agent.pol == 3:
            polName = "Softmax"
        if agent.alg == 1:
            algName = "Q-learning"
        if agent.alg == 3:
            algName = "QV-learning"
        agent.printResults()
        agent.createTable("", algName, polName)
        agent.createSplitTable("", algName, polName)
        #agent.plotWinrates()

    return agent

#function to parse input between an upper and lower bound
def inputNumber(lowerBound, upperBound):
    x = 0
    while x < lowerBound or x > upperBound:
        try:
             x = int(input())
        except:
            print("That is not an option, try again.")
    return x

def changeAlg(agent):
    #choose algorithm
    print("Choose the algorithm to be used:")
    print("1: Q-learning")
    print("2: Genetic algorithm")
    print("3: QV-learning")
    print("4: Particle Swarm Optimization")
    alg = inputNumber(1, 4)
    agent.alg = alg
    return alg

def changePol(agent):
    #choose exploration policies
    print("Choose the exploration policy (RL):")
    print("1: Greedy")
    print("2: E-Greedy")
    print("3: Softmax")
    agent.pol = inputNumber(1, 3)

def changeParent(genetics):
    print("Choose how parents are selected (GA):")
    print("1: Rank based")
    print("2: Tourney")
    genetics.par = inputNumber(1, 2)

def changeInheritance(genetics):
    print("Choose how children inherit genes from parents")
    print("1: Single point crossover")
    print("2: Chance by rank")
    genetics.inher = inputNumber(1, 2)

#function for giving input/output to change any variable
def changeVars(change, agent, genetics):
    if change == 1:
        print("Enter an integer for the random seed.")
        try:
            randomSeed = int(input())
            agent.randomSeed = randomSeed
            genetics.randomSeed = randomSeed
            random.seed(randomSeed)
            print("Random seed set to " + str(randomSeed))
        except:
            print("No proper input, returning to menu.")
    elif change == 2:
        print("Enter an integer > 0 for the epochs.")
        print("For genetic algorithm this value is per agent per generation.")
        try:
            epochs = int(input())
            if epochs > 0:
                agent.epochs = epochs
                genetics.epochs = epochs
                print("Epochs set to " + str(epochs))
            else:
                print("Value is not greater than 0")
        except:
            print("No proper input, returning to menu")
    elif change == 3:
        print("Enter 3 floats for the rewards. [Loss, hit without bust, win].")
        reward = input()
        rewards = reward.split(",")
        try:
            for i in range(3):
                rewards[i] = float(rewards[i])
            agent.rewards = rewards
            print("Rewards set to " + str(rewards))
        except:
            print("No proper input, returning to the menu.")
    elif change == 4:
        print("Enter a float between 0 and 1 for alpha.")
        try:
            alpha = float(input())
            if alpha >= 0 and alpha <= 1:
                agent.alpha = alpha
                print("Alpha set to " + str(alpha))
            else:
                print("Value not within bounds, returning to menu.")
        except:
            print("No proper input, returning to menu.")
    elif change == 5:
        print("Enter a float between 0 and 1 for gamma.")
        try:
            gamma = float(input())
            if gamma >= 0 and gamma <= 1:
                agent.gamma = gamma
                print("Gamma set to " + str(gamma))
            else:
                print("Value not within bounds, returning to menu.")
        except:
            print("No proper input, returning to menu.")
    elif change == 6:
        print("Enter a float between 0 and 1 for epsilon.")
        try:
            epsilon = float(input())
            if epsilon >= 0 and epsilon <= 1:
                agent.startEpsilon = epsilon
                agent.epsilon = epsilon
                print("Epsilon set to " + str(epsilon))
            else:
                print("Value not within bounds, returning to menu.")
        except:
            print("No proper input, returning to menu.")
    elif change == 7:
        print("Enter an integer > 0 for the amount of generations.")
        try:
            gens = int(input())
            if gens > 0:
                genetics.nrGens = gens
                print("Generations set to: " + str(gens))
            else:
                print("Value not within bounds, returning to menu.")
        except:
            print("No proper input, returning to menu.")
    elif change == 8:
        print("Enter an even integer > 0 for the amount of agents.")
        try:
            agents = int(input())
            if agents % 2 == 0 and agents > 0:
                genetics.nrAgents = agents
                print("Amount of agents set to " + str(agents))
            else:
                print("Value not within bounds, returning to menu.")
        except:
            print("No proper input, returning to menu.")
    elif change == 9:
        print("Enter a float for the mutation chance for each state between 0 and 1.")
        try:
            mutate = float(input())
            if mutate >= 0 and mutate <= 1:
                genetics.mutateChanceStart = mutate
                print("Mutations chance set to " + str(mutate))
            else:
                print("Value not within bounds, returning to menu.")
        except:
            print("No proper input, returning to menu.")

if __name__ == "__main__":
    agentGA = agentGA.AgentGA()
    agent = agentRL.AgentRL()
    genetics = GA.genetics()
    PSO = PSO.PSO()
    algorithm = saveFile.load(agent, genetics, PSO)
    random.seed(agent.randomSeed)
    learningCurve = []
    #main menu to choose an option
    while(True):
        print("Menu")
        print("1: Start")
        print("2: Change variables")
        print("3: Change algorithm")
        print("4: Change policy")
        print("5: Change Parent Selection")
        print("6: Change Inheritance")
        print("7: Reset agent")
        print("8: Save settings")
        print("9: Create learning curve for n agents")
        print("10: Exit")
        a = inputNumber(1,12)
        #run the algorithm
        if a == 1:
            ts = time()
            if algorithm != 2 and algorithm != 4:
                #play the game with a singular agent
                game(agent)
            elif algorithm == 2:
                #play the game with the genetics algorithm
                agentGA = genetics.GA()
                saveFile.saveGA(genetics)
            elif algorithm == 4:
                #play the game with PSO
                agentGA = PSO.pso()
                #savefile.savePSO(PSO)
            print("Processing time: " + str("{:.1f}".format(time() - ts)) + " seconds")
        #changing variables
        elif a == 2:
            print("Which variable to change?")
            print("---Universal variables------")
            print("1: random Seed. " + str(agent.randomSeed))
            print("2: epochs. " + str(agent.epochs))
            print("---RL Exclusive-------------")
            print("3: rewards. " + str(agent.rewards))
            print("4: alpha. " + str(agent.alpha))
            print("5: gamma. " + str(agent.gamma))
            print("6: epsilon. " + str(agent.startEpsilon))
            print("---Evolutionary Exclusives--")
            print("7: number of generations. " + str(genetics.nrGens))
            print("8: number of agents. " + str(genetics.nrAgents))
            print("9: mutation chance. " + str(genetics.mutateChanceStart))       
            print("10: return to menu.")
            change = inputNumber(1, 10)
            if change == 10:
                continue
            changeVars(change, agent, genetics)
        #change the algorithm
        elif a == 3:
            algorithm = changeAlg(agent)
        #change the exploration policy
        elif a == 4:
            changePol(agent)
        #change parent selection for GA
        elif a == 5:
            changeParent(genetics)
        #chance inheritance for children for GA
        elif a == 6:
            changeInheritance(genetics)
        #reset the agents
        elif a == 7:
            agent.reset()
            genetics.reset()
            print("Agent(s) has/have been reset.")
        #save settings
        elif a == 8:
            saveFile.save(agent, genetics, algorithm)
            print("Settings saved.")
        #create learning curve plot
        elif a == 9:
            print("How many agents? (2-1000)")
            agents = inputNumber(2, 1000)
            agentList = []
            for i in range(agents):
                agentList.append(agentRL.AgentRL())
                algorithm = saveFile.load(agentList[i],genetics, PSO)

            #use multiprocessing to create n agents at once
            with mp.Pool(mp.cpu_count() - 2) as p:
                agentList = p.map(game, agentList)

            for i in range(len(agentList)):
                learningCurve.append(agentList[i].winrates)

            # Fill states array with empty state/action pairs: [dealer hand[own hand[hit, stand]]]
            states = [[[0 for x in range(2)] for y in range(22)] for z in range(12)]
            #add double down for playerHand values of 9-11.
            for i in range (9,12):
                for j in range(12):
                    states[j][i].append(0)
            #state array for the hands in which you are allowed to split. [dealer hand[own pair of cards][hit, stand, split]]
            splitStates = [[[0 for x in range(3)] for y in range(12)] for z in range(12)]
            #add double down, only for a pair of 5's
            for i in range(12):
                splitStates[i][5].append(0)

            for i in range(agents):
                for j in range(len(agentList[i].states)):
                    for k in range(len(agentList[i].states[j])):
                        idx = np.argmax(agentList[i].states[j][k])
                        states[j][k][idx] += 1
                for j in range(len(agentList[i].splitStates)):
                    for k in range(len(agentList[i].splitStates[j])):
                        idx = np.argmax(agentList[i].splitStates[j][k])
                        splitStates[j][k][idx] += 1
            
            agentList[0].states = states
            agentList[0].splitStates = splitStates
            algName = "Undefined"
            polName = "Undefined"
            if agentList[0].pol == 1:
                polName = "Greedy"
            if agentList[0].pol == 2:
                polName = "e-Greedy"
            if agentList[0].pol == 3:
                polName = "Softmax"
            if agentList[0].alg == 1:
                algName = "Q-learning"
            if agentList[0].alg == 3:
                algName = "QV-learning"
            agentList[0].createTable("", algName, polName)
            agentList[0].createSplitTable("", algName, polName)
            createLCplot(learningCurve, algorithm, agentList[0].pol)
            saveFile.saveAgentResults(agentList)
        #exit program
        elif a == 10:
            saveFile.save(agent, genetics, algorithm)
            print("Settings saved.")
            print("Exiting...")
            break
        
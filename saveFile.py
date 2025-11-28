def save(agent, genetics, algorithm):
    #setting all info in the settings to currently used ones
    newLines = []
    with open("settings.txt", 'r') as s:
        lines = s.readlines()
        for line in lines:
            info = line.split('=')
            if info[0] == 'randomSeed':
                info[1] = str(agent.randomSeed) + "\n"
            elif info[0] == 'epochs':
                info[1] = str(agent.epochs) + "\n"
            elif info[0] == 'rewards':
                rewards = agent.rewards
                info[1] = ','.join(map(str,rewards))
                info[1] += "\n"
            elif info[0] == 'alpha':
                info[1] = str(agent.alpha) + "\n"
            elif info[0] == 'gamma':
                info[1] = str(agent.gamma) + "\n"
            elif info[0] == 'epsilon':
                info[1] = str(agent.startEpsilon) + "\n"
            elif info[0] == 'nrGens':
                info[1] = str(genetics.nrGens) + "\n"
            elif info[0] == 'nrAgents':
                info[1] = str(genetics.nrAgents) + "\n"
            elif info[0] == 'mutateChance':
                info[1] = str(genetics.mutateChanceStart) + "\n"
            elif info[0] == 'algorithm':
                info[1] = str(algorithm) + "\n"
            elif info[0] == 'policy':
                info[1] = str(agent.pol) + "\n"
            elif info[0] == 'parent':
                info[1] = str(genetics.par) + "\n"
            elif info[0] == 'inheritance':
                info[1] = str(genetics.inher) + "\n"
            line = '='.join(info)
            newLines.append(line)
        s.close()
    #saving
    with open('settings.txt', 'w') as s:
        s.writelines(newLines)
        s.close()

def load(agent, genetics, PSO):
    #loading settings from the savefile
    algorithm = 1
    with open('settings.txt') as s:
        lines = s.readlines()
        for line in lines:
            info = line.split('=')
            if info[0] == 'randomSeed':
                genetics.randomSeed = int(info[1])
                agent.randomSeed = int(info[1])
            elif info[0] == 'epochs':
                genetics.epochs = int(info[1])
                agent.epochs = int(info[1])
            elif info[0] == 'rewards':
                rewards = info[1].split(',')
                for i in range(len(rewards)):
                    rewards[i] = float(rewards[i])
                agent.rewards = rewards
            elif info[0] == 'alpha':
                agent.alpha = float(info[1])
            elif info[0] == 'gamma':
                agent.gamma = float(info[1])
            elif info[0] == 'epsilon':
                agent.startEpsilon = float(info[1])
                agent.epsilon = float(info[1])
            elif info[0] == 'nrGens':
                PSO.gen = int(info[1])
                genetics.nrGens = int(info[1])
            elif info[0] == 'nrAgents':
                PSO.n = int(info[1])
                genetics.nrAgents = int(info[1])
            elif info[0] == 'mutateChance':
                genetics.mutateChanceStart = float(info[1])
            elif info[0] == 'algorithm':
                algorithm = int(info[1])
                agent.alg = int(info[1])
            elif info[0] == 'policy':
                agent.pol = int(info[1])
            elif info[0] == 'parent':
                genetics.par = int(info[1])
            elif info[0] == 'inheritance':
                genetics.inher = int(info[1])
        s.close()
    return algorithm

def saveAgentResults(agentList):
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

    #saving all results of agent in textfile
    newLines = []
    for agent in agentList:
        newLines.append("Wins:" + str(agent.wins))
        newLines.append(" Losses:" + str(agent.losses))
        newLines.append(" Money Gain:" + str(agent.money))
        newLines.append(" Winrate:" + str("{:.2f}".format(agent.wins / (agent.wins + agent.losses) * 100)))
        newLines.append("\n")
    
    with open('agentsResults' + algName + polName + '.txt', 'w') as s:
        s.writelines(newLines)
        s.write("States\n")
        for i in range(len(agentList[0].states)):
            s.write(str(agentList[0].states[i]))
            s.write("\n")
        s.write("SplitStates\n")
        for i in range(len(agentList[0].splitStates)):
            s.write(str(agentList[0].splitStates[i]))
            s.write("\n")
        for i in range(len(agentList)):
            agentList[i].winrates = [round(num, 2) for num in agentList[i].winrates]
            s.write(str(agentList[i].winrates))
            s.write("\n")
        s.close()

def saveGA(genetics):
    parName = 'Undefined'
    inherName = 'Undefined'
    if genetics.par == 1:
        parName = 'Ranked'
    if genetics.par == 2:
        parName = 'Tourney'
    if genetics.inher == 1:
        inherName = 'Crossover'
    if genetics.inher == 2:
        inherName = 'Ranked'
    
    with open('GAresults' + parName + inherName + '.txt', 'w') as s:
        for gen in range(genetics.nrGens):
            genetics.allWinrates[gen].sort(reverse=True)
            s.write("Gen:" + str(gen) + "\n")
            for item in genetics.allWinrates[gen]:
                s.write(str("{:.2f}".format(item)) + ", ")
            s.write("\n")
        s.close()

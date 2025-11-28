import random
from numpy import argmax
import matplotlib.pyplot as plt
import os

class Agent():

    def __init__(self):
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

        self.money = 0
        self.wins = 0
        self.losses = 0
        self.naturalWins = 0
        self.splits = 0
        self.randomSeed = 1
        self.epochs = 100000
        self.alg = 1
    
    #creates the policy table for the split states
    def createSplitTable(self,gen, algName, polName):
        #initializing variables for the table
        val1 = [i for i in range(2,12)] 
        val2 = [i for i in range(10,1,-1)] 
        val3 = [["" for c in range(10)] for r in range(9)] 

        #set each cell to hit or stand
        for i in range(2,12):
            for j in range(2,11):
                action = argmax(self.splitStates[i][j])
                if action == 0:
                    val3[(j-10)*-1][i-2] = "H"
                elif action == 1:
                    val3[(j-10)*-1][i-2] = "S"
                elif action == 2:
                    val3[(j-10)*-1][i-2] = "P"
                elif action == 3:
                    val3[(j-10)*-1][i-2] = "D"

        #create the layout
        fig, ax = plt.subplots() 
        ax.set_axis_off() 
        table = ax.table( 
            cellText = val3,  
            rowLabels = val2,  
            colLabels = val1, 
            rowColours =["white"] * 9,  
            colColours =["white"] * 10, 
            cellLoc ='center',  
            loc ='upper left')      
        
        for key, cell in table._cells.items():
            if cell._text.get_text() == "H":
                cell.set_facecolor("palegreen")
            elif cell._text.get_text() == "S":
                cell.set_facecolor("pink")
            elif cell._text.get_text() == "D":
                cell.set_facecolor("blue")
            elif cell._text.get_text() == "P":
                cell.set_facecolor("purple")

        plt.title('Blackjack policy using ' + algName + ' with ' + polName + ' for splitting.') 
        
        os.makedirs("SplitTables", exist_ok=True)
        plt.savefig("SplitTables/tableSplit" + algName + polName + str(gen) + ".png")
        plt.close()

    #creates the policy table
    def createTable(self, gen, algName, polName):
        #initializing variables for the table
        val1 = [i for i in range(2,12)] 
        val2 = [i for i in range(20,3,-1)] 
        val3 = [["" for c in range(10)] for r in range(17)] 

        #set each cell to hit or stand
        for i in range(2,12):
            for j in range(4,21):
                action = argmax(self.states[i][j])
                if action == 0:
                    val3[(j-20)*-1][i-2] = "H"
                elif action == 1:
                    val3[(j-20)*-1][i-2] = "S"
                elif action == 2:
                    val3[(j-20)*-1][i-2] = "D"
        
        #create the layout
        fig, ax = plt.subplots() 
        ax.set_axis_off() 
        table = ax.table( 
            cellText = val3,  
            rowLabels = val2,  
            colLabels = val1, 
            rowColours =["white"] * 17,  
            colColours =["white"] * 10, 
            cellLoc ='center',  
            loc ='upper left')      
        
        for key, cell in table._cells.items():
            if cell._text.get_text() == "H":
                cell.set_facecolor("palegreen")
            elif cell._text.get_text() == "S":
                cell.set_facecolor("pink")
            elif cell._text.get_text() == "D":
                cell.set_facecolor("blue")

        plt.title('Blackjack policy using ' + algName + " with " + polName) 
        
        os.makedirs("Tables", exist_ok=True)
        plt.savefig("Tables/table" + algName + polName + str(gen) + ".png")
        plt.close()
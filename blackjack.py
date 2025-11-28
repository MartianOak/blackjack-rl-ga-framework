import numpy as np
import random
from numpy import argmax

#function to check whether the player or the dealer won
def checkWin(hand1, hand2):
    win = False
    if np.sum(hand2) > 21:
        win = True
    elif np.sum(hand1) > np.sum(hand2):
        win = True
    return win

def dealerTurn(stack, dealerHand):
    #dealers turn
    while(True):
        if np.sum(dealerHand) < 17:
            #dealer must hit when under 17
            dealerHand = np.append(dealerHand, stack.pop(0))
            #turn A's into a 1 if over 21
            if np.sum(dealerHand) > 21:
                if 11 in dealerHand:
                    dealerHand = np.where(dealerHand == 11, 1, dealerHand)
        else:
            #dealer must stand when 17+
            break
    return dealerHand

#main function to play the blackjack game
def blackjack(stack, agent, i, playerHand, dealerHand):
    alg = agent.alg
    win = None
    #print(stack)
    #check for natural
    if sum(playerHand) == 21:
        if sum(dealerHand) < 21:
            #if i > agent.epochs / 2:
            agent.naturalWins += 1
            win = True
        else:
            win = False
    elif sum(dealerHand) == 21 and dealerHand.size == 2:
        win = False

    #keep looping while hitting, if stand, then stop the loop
    if win == None:
        while(True):
            done = False
            #turn A's into a 1 if over 21
            if np.sum(playerHand) > 21:
                if 11 in playerHand:
                    playerHand[argmax(playerHand)] = 1

            #use the correct exploration policy to get the action
            if alg != 2 and alg != 5 and alg != 6:
                if agent.pol == 1:
                    x = agent.getActionGreedy(playerHand, dealerHand[0])
                elif agent.pol == 2:
                    x = agent.getActionEGreedy(playerHand, dealerHand[0])
                elif agent.pol == 3:
                    x = agent.getActionSoftmax(playerHand, dealerHand[0])
            elif alg == 2 or alg == 5 or alg == 6:
                x = agent.getActionGreedy(playerHand, dealerHand[0])

            if x == 0:
                #hit
                playerHand = np.append(playerHand, stack.pop(0))
                #turn A's into a 1 if over 21
                if np.sum(playerHand) > 21:
                    if 11 in playerHand:
                        playerHand[argmax(playerHand)] = 1
                #check for bust
                if np.sum(playerHand) > 21:
                    if alg == 1 or alg == 3:
                        if playerHand.size == 3 and playerHand[0] == playerHand[1]:
                            agent.updateSplitStates(playerHand[0], x, dealerHand[0], 0, agent.rewards[0])
                        else:
                            agent.updateStates(np.sum(playerHand[:-1]), x, dealerHand[0], 0, agent.rewards[0])
                    win = False
                    break
                else:
                    if alg == 1 or alg == 3:
                        if playerHand.size == 3 and playerHand[0] == playerHand[1]:
                            agent.updateSplitStates(playerHand[0], x, dealerHand[0], np.sum(playerHand), agent.rewards[1])
                        else:
                            agent.updateStates(np.sum(playerHand[:-1]), x, dealerHand[0], np.sum(playerHand), agent.rewards[1])
                    continue

            elif x == 2:
                #double down
                playerHand = np.append(playerHand, stack.pop(0))
                if np.sum(playerHand) > 21:
                    playerHand = np.where(playerHand == 11, 1, playerHand)
                dealerHand = dealerTurn(stack, dealerHand)

                if checkWin(playerHand, dealerHand):
                    if alg == 1 or alg == 3:
                        if playerHand.size == 3 and playerHand[0] == playerHand[1]:
                            agent.updateSplitStates(playerHand[0], 3, dealerHand[0], np.sum(playerHand), agent.rewards[2] * 2)
                        else:
                            agent.updateStates(np.sum(playerHand[:-1]), x, dealerHand[0], np.sum(playerHand), agent.rewards[2] * 2)
                    win = True
                    break
                else:
                    if alg == 1 or alg == 3:
                        if playerHand.size == 3 and playerHand[0] == playerHand[1]:
                            agent.updateSplitStates(playerHand[0], 3, dealerHand[0], np.sum(playerHand), agent.rewards[0] * 2)
                        else:
                            agent.updateStates(np.sum(playerHand[:-1]), x, dealerHand[0], np.sum(playerHand), agent.rewards[0] * 2)
                    win = False
                    break
                break

            elif x == 3:
                #split
                #print("split")
                #count how many times splitted
                if alg == 2 or alg == 5 or alg == 6:
                    agent.splits += 1
                elif i > agent.epochs / 2:
                    agent.splits += 1

                #reshuffle deck in not enough cards, since splits won't re-enter jack.game
                if len(stack) < 70:
                    suit = [2,3,4,5,6,7,8,9,10,10,10,10,11]
                    deck = suit + suit + suit + suit
                    stack = deck + deck + deck + deck + deck + deck
                    random.shuffle(stack)

                #play a game with the first card
                dealerHand = dealerTurn(stack, dealerHand)
                if blackjack(stack, agent, i, playerHand[:1], dealerHand):
                    if alg == 1 or alg == 3:
                        agent.updateSplitStates(playerHand[0], 2, dealerHand[0], np.sum(playerHand), agent.rewards[2])
                else:
                    if alg == 1 or alg == 3:
                        agent.updateSplitStates(playerHand[0], 2, dealerHand[0], np.sum(playerHand), agent.rewards[0])
                        
                #play a game with the second card
                if blackjack(stack, agent, i, playerHand[1:], dealerHand):
                    if alg == 1 or alg == 3:
                        agent.updateSplitStates(playerHand[1], 2, dealerHand[0], np.sum(playerHand), agent.rewards[2])
                else:
                    if alg == 1 or alg == 3:
                        agent.updateSplitStates(playerHand[0], 2, dealerHand[0], np.sum(playerHand), agent.rewards[0])
                break
            else:
                #no hit
                dealerHand = dealerTurn(stack, dealerHand)
                #check who wins
                if checkWin(playerHand, dealerHand):
                    if alg == 1 or alg == 3:
                        if playerHand.size == 2 and playerHand[0] == playerHand[1]:
                            agent.updateSplitStates(playerHand[0], x, dealerHand[0], np.sum(playerHand), agent.rewards[2])
                        else:
                            agent.updateStates(np.sum(playerHand), x, dealerHand[0], np.sum(playerHand), agent.rewards[2])
                    win = True
                    break
                else:
                    if alg == 1 or alg == 3:
                        if playerHand.size == 2 and playerHand[0] == playerHand[1]:
                            agent.updateSplitStates(playerHand[0], x, dealerHand[0], np.sum(playerHand), agent.rewards[0])
                        else:
                            agent.updateStates(np.sum(playerHand), x, dealerHand[0], np.sum(playerHand), agent.rewards[0])
                    win = False
                    break

    #count wins
    if win:
        if alg == 2 or alg == 5 or alg == 6:
            agent.wins += 1
        elif i > agent.epochs / 2:
            agent.wins += 1
        if alg != 2 and alg != 5 and alg != 6:
            agent.addGameToMemory(1, i)
    elif win == False:
        if alg == 2 or alg == 5 or alg == 6:
            agent.losses += 1
        elif i > agent.epochs / 2:
            agent.losses += 1
        if alg != 2 and alg != 5 and alg != 6:
            agent.addGameToMemory(0, i)
    #print(playerHand)
    #print(dealerHand)
    #print(str(np.sum(playerHand)) + " | " + str(np.sum(dealerHand)))
    #print("Win: " + str(win))
    #print("Wins: " + str(agent.wins))
    #print("Losses: " + str(agent.losses))
    #input()
    return win
                
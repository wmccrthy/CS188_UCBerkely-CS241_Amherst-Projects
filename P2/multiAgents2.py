# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from json.encoder import INFINITY
from xxlimited import foo
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0  #arbitrary starting score.
# A lot of these weights are arbitrary lol, just tried out different combinations until I got good performance
        ghostDists = []
        foodDists = []

        for ghostPos in successorGameState.getGhostPositions():
            if ghostPos == newPos:
                return -9999   # arbitrarily low? Just don't go to this spot if there's a ghost here
            ghostDists.append(manhattanDistance(ghostPos, newPos))
        if len(newFood.asList()) == 0:
            return 9999  # also arbitrary, it means that all the food will be eaten at this state.

        for dist in ghostDists:  # else check if there's a ghost nearby, penalty if there is.
            if dist <= 1.0:
                score -= 100
            elif dist <= 4:
                score -= 10

        for foodPos in newFood.asList():
            foodDists.append(manhattanDistance(foodPos, newPos))  # for a while I was doing (foodPos, ghostPos) lmao
            # fixing that sped things up a bunch.

        for dist in foodDists:
            if dist <= 1.5:
                score += 25
            elif dist <= 2:
                score += 10
            elif dist <= 3:
                score += 5
            # elif dist <= 5:   # right now it takes the pacman a while to find distant food things, but these make it worse lol
            #     score += 2
            # elif dist <= 10:
            #     score += 1

        score = score - 20 * len(newFood.asList())  # weight getting rid of all the food things highly.
        return successorGameState.getScore() + score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def maxValue(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        score = -999999999
        for action in gameState.getLegalActions(agent):
            (score2, action2) = self.minValue(gameState.generateSuccessor(0, action), 1, depth)
            if score2 > score:
                score, move = score2, action
        return (score, move)
    
    def minValue(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        score = 999999999
        for action in gameState.getLegalActions(agent):
            nextPlayerIndex = agent + 1
            if nextPlayerIndex < gameState.getNumAgents():
                # essentially, if player turn is MIN/if next index is a valid ghost index, should call min on next ghost, else call max on pacman
                # if still ghosts to be checked, call min function on next ghost 
                (score2, action2) = self.minValue(gameState.generateSuccessor(agent, action), nextPlayerIndex, depth)
                if score2 < score:
                    score, move = score2, action
            # call min for every ghost before going back to max (depth only increases after going back to max for every call of min)
    
            else:
                # function reaches this point at the final move in a ply (pacman and every ghost taken action)
                # meaning, it is pacman's turn, so nextPlayerIndex is 0 and call max function
                # bc one full ply will have been completed, function also needs to move on to next depth level
                nextPlayerIndex = 0
                nextDepth = depth + 1
                (score2, action2) = self.maxValue(gameState.generateSuccessor(agent, action), nextPlayerIndex, nextDepth)
                if score2 < score:
                    score, move = score2, action
        return (score, move)
    

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        score, move = self.maxValue(gameState, 0, 0)
        return move

        return None 
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        score = -999999999
        for action in gameState.getLegalActions(agent):
            (score2, action2) = self.minValue(gameState.generateSuccessor(agent, action), 1, depth, alpha, beta)
            if score2 > score:
                score, move = score2, action
            if score > beta:
                return (score, move)
            alpha = max(alpha, score)
        return (score, move)
    
    def minValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        score = 999999999
        for action in gameState.getLegalActions(agent):
            nextPlayerIndex = agent + 1
            if nextPlayerIndex < gameState.getNumAgents():
                # if still ghosts to be checked, call min function on next ghost
                # call min for every ghost before going back to max (depth only increases after going back to max despite # of calls of min)
                (score2, action2) = self.minValue(gameState.generateSuccessor(agent, action), nextPlayerIndex, depth, alpha, beta)
            # move = action2
                if score2 < score:
                    score, move = score2, action
                if score < alpha:
                    return (score, move)
                beta = min(beta, score)
            
            else:
                # function reaches this point at the final move in a ply (pacman and every ghost take action)
                # meaning, it is pacman's turn, so call max function
                # bc one full ply will have been completed, function needs to move on to next depth level
                nextPlayerIndex = 0
                nextDepth = depth + 1
                (score2, action2) = self.maxValue(gameState.generateSuccessor(agent, action), nextPlayerIndex, nextDepth, alpha, beta)
                if score2 < score:
                    score, move = score2, action
                if score < alpha:
                    return (score, move)
                beta = min(beta, score)

        return (score, move)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # psuedocode from textbook

        score, move = self.maxValue(gameState, 0, 0, -999999999, 999999999)

        return move


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    # modify the minValue function such that it weights ghost nodes (MIN) as the average of all legal successor actions
    # simple as that?
    # still needs to properly recurse 

    def maxValue(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        score = -999999999
        for action in gameState.getLegalActions(agent):
            (score2, action2) = self.expectiValue(gameState.generateSuccessor(0, action), 1, depth)
            if score2 > score:
                score, move = score2, action
        return (score, move)
    
    def expectiValue(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        score = 0.0
        for action in gameState.getLegalActions(agent):
            nextPlayerIndex = agent + 1
            if nextPlayerIndex < gameState.getNumAgents():
                (score2, action2) = self.expectiValue(gameState.generateSuccessor(agent, action), nextPlayerIndex, depth)
                # like with minValue, call expectiValue on every ghost before returning to max for pacman

                # score of each node increase by weight of each action in average 
                score += score2/len(gameState.getLegalActions(agent))
                
                
            else:
                # function reaches this point at the final move in a ply (pacman and every ghost take action)
                # meaning, it is pacman's turn, so call max function
                # bc one full ply will have been completed, function needs to move on to next depth level
                nextPlayerIndex = 0
                nextDepth = depth + 1
                (score2, action2) = self.maxValue(gameState.generateSuccessor(agent, action), nextPlayerIndex, nextDepth)
                # score of each node increase by weight of each action in average 
                score += score2/len(gameState.getLegalActions(agent))
                
        return (score, None)


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # max/expecti functions return (score, action) tuples; for getAction, just need to return the action 

        score, move = self.maxValue(gameState, 0, 0)
        return move
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # just copy pasted from q1, mostly.
    score = 0  # arbitrary starting score
    ghostDists = []
    foodDists = []

    if newScaredTimes[0] == 0:
        for ghostPos in currentGameState.getGhostPositions():
            if ghostPos == newPos:
                return -1000  # arbitrarily low? Just don't go to this spot if there's a ghost here
            # elif ghostPos == newPos and newScaredTimes == 0: #if it's scared, chase it down
            #     score += 500
            ghostDists.append(manhattanDistance(ghostPos, newPos))
        if len(newFood.asList()) == 0:
            return 1000  # also arbitrary, but it means that all the food will be eaten at this state.
        for dist in ghostDists:  # else check if there's a ghost nearby, penalty if there is.
            if dist <= 1.0:
                score -= 100
            elif dist <= 4:
                score -= 10
            elif dist <= 8:
                score -= 5

        for foodPos in newFood.asList():
            foodDists.append(manhattanDistance(foodPos, newPos)) 

        for dist in foodDists:
            if dist <= 1.5:
                score += 25
            elif dist <= 2:
                score += 10
            elif dist <= 3:
                score += 5
    else:
    # if ghost is scared, only thing that matters is how close we are to ghost
    # the closer pacman is, the better the score
        for ghostPos in currentGameState.getGhostPositions():
            if ghostPos == newPos:
                return currentGameState.getScore() + 100
            score += 1/(manhattanDistance(ghostPos, newPos))
            return currentGameState.getScore() + score

    score = score - 20 * len(newFood.asList())  # weight getting rid of all the food things highly.
    return currentGameState.getScore() + score


# Abbreviation
better = betterEvaluationFunction

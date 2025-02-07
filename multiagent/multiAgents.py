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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
           # Start with the game score
        score = successorGameState.getScore()
        
        # Food considerations
        foodList = newFood.asList()
        if foodList:
            # Encourage moving towards closest food
            closestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / (closestFoodDist + 1)  # Add reciprocal of distance
        
        # Ghost considerations
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            
            if newScaredTimes[i] > 0:
                # Ghost is scared - chase it for points
                if ghostDist > 0:
                    score += 1.0 / ghostDist
            else:
                # Ghost is dangerous - avoid it
                if ghostDist <= 1:
                    score -= 1000  # Heavy penalty for being too close
                elif ghostDist <= 2:
                    score -= 100   # Moderate penalty for being close
        
        # Slight penalty for stopping
        if action == Directions.STOP:
            score -= 10
            
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(state, depth, agentIndex):
            # Terminal state check
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            
            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                maxVal = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, depth, 1)
                    maxVal = max(maxVal, val)
                return maxVal
            
            # Ghost's turn (minimizing player)
            else:
                minVal = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                
                # If this is the last ghost, next turn is Pacman's with decreased depth
                if nextAgent >= state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth - 1
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = minimax(successor, nextDepth, nextAgent)
                    minVal = min(minVal, val)
                return minVal
        # Get the best action for Pacman
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestVal = float('-inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            val = minimax(successor, self.depth, 1)
            if val > bestVal:
                bestVal = val
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Terminal state check
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            
            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                maxVal = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphaBeta(successor, depth, 1, alpha, beta)
                    maxVal = max(maxVal, val)
                    alpha = max(alpha, maxVal)
                    if beta < alpha:
                        break  # Beta cutoff
                return maxVal
            
            # Ghost's turn (minimizing player)
            else:
                minVal = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                
                # If this is the last ghost, next turn is Pacman's with decreased depth
                if nextAgent >= state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth - 1
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    val = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                    minVal = min(minVal, val)
                    beta = min(beta, minVal)
                    if beta < alpha:
                        break  # Alpha cutoff
                return minVal
        
        # Get the best action for Pacman
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestVal = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            val = alphaBeta(successor, self.depth, 1, alpha, beta)
            if val > bestVal:
                bestVal = val
                bestAction = action
            alpha = max(alpha, bestVal)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Terminal states
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')
    
    # Base score
    score = currentGameState.getScore()
    
    # Pacman position
    pacmanPos = currentGameState.getPacmanPosition()
    
    # Food evaluation
    foodList = currentGameState.getFood().asList()
    if foodList:
        # Distance to closest food
        foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
        closestFoodDist = min(foodDistances)
        score += 10.0 / (closestFoodDist + 1)  # Encourage moving toward food
        
        # Penalty for remaining food
        score -= 4 * len(foodList)
    

# Abbreviation
better = betterEvaluationFunction

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

# Altun Hasanli
# ahasanli19
# 0073075

import random
import util

from game import Agent, Directions
from pacman import GameState
from util import manhattanDistance

plusInfinity = float("inf")
minusInfinity = float("-inf")


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
        chosenIndex = random.choice(bestIndices)

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
        newScore = successorGameState.getScore()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # If the game is won, return the maximum possible score
        if successorGameState.isWin():
            return plusInfinity

        # If the game is lost, return the minimum possible score
        if successorGameState.isLose():
            return minusInfinity

        # Calculate the minimum distance to a food
        foodList = newFood.asList()
        minFoodDistance = plusInfinity
        for food in foodList:
            minFoodDistance = min(minFoodDistance, manhattanDistance(newPos, food))

        # Calculate the minimum distance to a ghost considering scared times
        minGhostDistance = plusInfinity
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghost.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                continue
            minGhostDistance = min(minGhostDistance, distance)

        # If the ghost is too close, return the minimum possible score
        if minGhostDistance == 0:
            return minusInfinity

        # Go towards the nearest food and away from the nearest ghost
        return newScore + 1 / minFoodDistance - 1 / minGhostDistance


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def getMinimaxAgentControls(agentIndex: int) -> (float, Directions, callable):
    if agentIndex == 0:
        comparator = lambda score, bestScore: score > bestScore
        return minusInfinity, None, comparator
    else:
        comparator = lambda score, bestScore: score < bestScore
        return plusInfinity, None, comparator


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimaxAgent(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        bestScore, bestAction, comparator = getMinimaxAgentControls(agentIndex)

        nextAgent, nextDepth = (0, depth - 1) \
            if agentIndex == gameState.getNumAgents() - 1 else (agentIndex + 1, depth)

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.minimaxAgent(successor, nextAgent, nextDepth)
            if comparator(score, bestScore):
                bestScore, bestAction = score, action

        return bestScore, bestAction

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
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        _, action = self.minimaxAgent(gameState, 0, self.depth)
        return action


def getAlphaBetaAgentControls(agentIndex: int) -> (float, Directions, callable, callable, callable):
    if agentIndex == 0:
        comparator = lambda score, bestScore: score > bestScore
        alphaBetaComparator = lambda score, alpha, beta: score > beta
        updateAlphaBeta = lambda bestScore, alpha, beta: (max(alpha, bestScore), beta)
        return minusInfinity, None, comparator, alphaBetaComparator, updateAlphaBeta
    else:
        comparator = lambda score, bestScore: score < bestScore
        alphaBetaComparator = lambda score, alpha, beta: score < alpha
        updateAlphaBeta = lambda bestScore, alpha, beta: (alpha, min(beta, bestScore))
        return plusInfinity, None, comparator, alphaBetaComparator, updateAlphaBeta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabetaAgent(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        bestScore, bestAction, comparator, alphaBetaComparator, updateAlphaBeta = \
            getAlphaBetaAgentControls(agentIndex)

        nextAgent, nextDepth = (0, depth - 1) \
            if agentIndex == gameState.getNumAgents() - 1 else (agentIndex + 1, depth)

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.alphabetaAgent(successor, nextAgent, nextDepth, alpha, beta)

            if comparator(score, bestScore):
                bestScore, bestAction = score, action
            if alphaBetaComparator(score, alpha, beta):
                return score, action
            alpha, beta = updateAlphaBeta(bestScore, alpha, beta)
        return bestScore, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, action = self.alphabetaAgent(gameState, 0, self.depth, minusInfinity, plusInfinity)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxAgent(self, gameState: GameState, agentIndex: int, depth: int):
        maxScore, maxAction = minusInfinity, None
        nextAgent, nextDepth = (0, depth - 1) \
            if agentIndex == gameState.getNumAgents() - 1 else (agentIndex + 1, depth)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.expectimax(successor, nextAgent, nextDepth)
            if score > maxScore:
                maxScore, maxAction = score, action
        return maxScore, maxAction

    def expectAgent(self, gameState: GameState, agentIndex: int, depth: int):
        totalScore, totalAction = 0, None
        nextAgent, nextDepth = (0, depth - 1) \
            if agentIndex == gameState.getNumAgents() - 1 else (agentIndex + 1, depth)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.expectimax(successor, nextAgent, nextDepth)
            totalScore += score
            totalAction = action
        return totalScore / len(gameState.getLegalActions(agentIndex)), totalAction

    def expectimax(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxAgent(gameState, agentIndex, depth)

        return self.expectAgent(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, action = self.expectimax(gameState, 0, self.depth)
        return action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Combines the ideas of the reflex agent such as nearest food and ghost distances, scared ghost effect,
    and also incorporates the game score. The weights are adjusted to balance the importance of each factor.
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    score = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    if currentGameState.isWin():
        return plusInfinity

    if currentGameState.isLose():
        return minusInfinity

    minFoodDistance = 0
    if food:
        minFoodDistance = min(manhattanDistance(pos, food) for food in food)

    # Calculate the minimum distance to a ghost considering scared times
    minGhostDistance = plusInfinity
    ghostEffect = 0
    for ghost, scaredTime in zip(ghostStates, scaredTimes):
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pos, ghostPos)
        if scaredTime > 0:
            ghostEffect += (distance if distance > 10 else -10)
        elif distance < minGhostDistance:
            minGhostDistance = distance

    if minGhostDistance == 0:
        return minusInfinity

    ghostWeight = 7 / (minGhostDistance + 1)
    foodWeight = minFoodDistance / 3

    return score - ghostWeight - foodWeight


# Abbreviation
better = betterEvaluationFunction

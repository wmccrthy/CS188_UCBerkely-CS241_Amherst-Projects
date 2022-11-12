# search.py - Jack Peng, Wyatt McCarthy
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())  # these were the cause of failing q1 for a bit lol
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # need a starting node, a frontier stack with that node in it, and an empty explored set
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.Stack()  # remember to check util.py for how these methods work
    explored = set()
    frontier.push((startingNode, []))
    # the empty list being pushed is the path to get to the starting node (from the starting node?).
    # then loop, if frontier is empty, return failure
    # else, node <- pop frontier, if it's a solution, return it
    while not frontier.isEmpty():
        # print("test1")
        node, path = frontier.pop()
        # print(node, path) # path is a list of the actions it takes to get to this node?
        if problem.isGoalState(node):
            # print(path)
            return path
    # if it's not in explored, add it, and then add the node's children to frontier.
        if node not in explored:
            explored.add(node)
            # print(node)
            # children = problem.getSuccessors(node)
            for (child, newPath,x) in problem.getSuccessors(node):
                # since getSuccessors returns three things. x is prob distance or cost or smthg?
                if child not in explored:
                    totalPath = path + [newPath]
                    frontier.push((child, totalPath))
            # print(explored)
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.Queue()  # exactly the same as depth, just with a Queue here instead of a Stack.
    explored = set()
    frontier.push((startingNode, []))
    while not frontier.isEmpty():
        temp = frontier.pop()
        node, path = temp
        if problem.isGoalState(node):
            return path
    # if it's not in explored, add it, and then add the node's children to frontier.
        # print(type(node))
        if node not in explored:
            explored.add(node)
            for (child, newPath,x) in problem.getSuccessors(node):
                if child not in explored:
                    totalPath = path + [newPath]
                    frontier.push((child, totalPath))
    sys.exit("Failure, search didn't find a solution")


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.PriorityQueue()
    explored = set()
    frontier.push((startingNode, [], 0), 0)  # (tuple of (startingNode, path list, priority), priority)
    while not frontier.isEmpty():
        # node, path = frontier.pop()
        temp = frontier.pop()
        node, path, cost = temp
        if problem.isGoalState(node):
            # print(temp)
            return path
        # if it's not in explored, add it, and then add the node's children to frontier.
        if node not in explored:
            explored.add(node)
            for (child, newPath, newCost) in problem.getSuccessors(node):
                if child not in explored:
                    totalPath = path + [newPath]
                    totalCost = cost + newCost
                    frontier.push((child, totalPath, totalCost), totalCost)
                    # having issues accessing priority from the node that just got popped, so I just add it twice


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # same as UCS, but with the queue prioritized by f(n) = path cost + heuristic
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.PriorityQueue()
    explored = set()
    frontier.push((startingNode, [], 0), 0 + heuristic(startingNode, problem))
    # so it's ((node, path to node, pathcost to node), priority (which is pathcost + heuristic)
    while not frontier.isEmpty():
        # node, path = frontier.pop()
        temp = frontier.pop()
        node, path, pathCost= temp
        if problem.isGoalState(node):
            # print(temp)
            return path
        # if it's not in explored, add it, and then add the node's children to frontier.
        if node not in explored:
            explored.add(node)
            for (child, newPath, newPathCost) in problem.getSuccessors(node):
                if child not in explored:
                    totalPath = path + [newPath]
                    totalPathCost = pathCost + newPathCost
                    priority = totalPathCost + heuristic(child, problem)
                    frontier.push((child, totalPath, totalPathCost), priority)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# search.py
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
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST
    # n = Directions.NORTH
    # e = Directions.EAST
    
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # sol = []

    # # node format: (parent, action taken to get there, state, cost)
    # node = ((), '', problem.getStartState(), 0) 
    
    # sol_node = node

    # # actual searching procedure that follows psuedocode 

    # # first checks if starting state is goal
    # if problem.isGoalState(node[2]):
    #     sol_node = node

    # # initializes frontier with starting node as only element; initializes explored set
    # frontier = util.Stack()
    # frontier.push(node)
    # explored = set()
    # run = True

    # while run is True:
    #     curNode = frontier.pop()
    #     explored.add(curNode[2])
    #     for action in problem.getSuccessors(curNode[2]):
    #         # print(problem.getSuccessors(curNode[2]))
    #         child = (curNode, action[1], action[0], action[2])
    #         if child not in frontier.list:
    #             if action[0] not in explored:
    #                 if problem.isGoalState(action[0]):
    #                     sol_node = child
    #                     run = False
    #                 frontier.push(child)
    #     # if frontier.isEmpty:
    #     #     return
    # # once solution node is found, loop through the path from solution node to start node and append each action taken to returned list 
    # # print(sol_node)
    # while len(sol_node[1]) > 0:
    #      sol.append(sol_node[1])
    #      sol_node = sol_node[0] 
    # sol.reverse()
    # return sol

    # Jacks code:
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
        # print(node, path)# path is a list of the actions it takes to get to this node?
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


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST
    # n = Directions.NORTH
    # e = Directions.EAST
    
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # sol = []

    # # node format: (parent, action taken to get there, state, cost)
    # node = ((), '', problem.getStartState(), 0) 
    
    # sol_node = node

    # # actual searching procedure that follows psuedocode 

    # # first checks if starting state is goal
    # if problem.isGoalState(node[2]):
    #     sol_node = node

    # # initializes frontier with starting node as only element; initializes explored set
    # frontier = util.Queue()
    # frontier.push(node)
    # explored = set()
    # run = True

    # while run is True:
         
    #     curNode = frontier.pop()
    #     explored.add(curNode[2])
    #     for action in problem.getSuccessors(curNode[2]):
    #         # print(problem.getSuccessors(curNode[2]))
    #         child = (curNode, action[1], action[0], action[2])
    #         if child not in frontier.list:
    #             if action[0] not in explored:
    #                 if problem.isGoalState(action[0]):
    #                     sol_node = child
    #                     run = False
    #                 frontier.push(child)
    #     # if frontier.isEmpty:
    #     #     return
    # # once solution node is found, loop through the path from solution node to start node and append each action taken to returned list 
    # # print(sol_node)
    # while len(sol_node[1]) > 0:
    #      sol.append(sol_node[1])
    #      sol_node = sol_node[0] 
    # sol.reverse()
    # return sol

    # Jacks code:
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.Queue()  # copy pasted from depth, just with a Queue here instead of a Stack.
    explored = set()
    frontier.push((startingNode, []))
    while not frontier.isEmpty():
        node, path = frontier.pop()
        if problem.isGoalState(node):
            return path
    # if it's not in explored, add it, and then add the node's children to frontier.
        if node not in explored:
            explored.add(node)
            for (child, newPath,x) in problem.getSuccessors(node):
                if child not in explored:
                    totalPath = path + [newPath]
                    frontier.push((child, totalPath))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST
    # n = Directions.NORTH
    # e = Directions.EAST
    
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # sol = []

    # # node format: (parent, action taken to get there, state, cost)
    # node = ((), '', problem.getStartState(), 0) 
    
    # sol_node = node

    # # actual searching procedure that follows psuedocode 

    # # first checks if starting state is goal
    # if problem.isGoalState(node[2]):
    #     sol_node = node

    # # initializes frontier with starting node as only element; initializes explored set
    # frontier = util.PriorityQueue()
    # frontier.push(node, node[3])
    # explored = set()
    # run = True

    # while run is True:
    #     curNode = frontier.pop()
    #     if problem.isGoalState(curNode[2]):
    #         sol_node = curNode
    #         run = False
    #     explored.add(curNode[2])
    #     for action in problem.getSuccessors(curNode[2]):
    #         # print(problem.getSuccessors(curNode[2]))
    #         child = (curNode, action[1], action[0], action[2])
    #         if child not in frontier.heap:
    #             if action[0] not in explored:
    #                 frontier.push(child, child[3])
    #     # if frontier.isEmpty:
    #     #     return 
    # # once solution node is found, loop through the path from solution node to start node and append each action taken to returned list 
    # # print(sol_node)
    # while len(sol_node[1]) > 0:
    #      sol.append(sol_node[1])
    #      sol_node = sol_node[0] 
    # sol.reverse()
    # return sol

    # Jacks code:
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.PriorityQueue()
    explored = set()
    frontier.push((startingNode, [], 0), 0)
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


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST
    # n = Directions.NORTH
    # e = Directions.EAST
    
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # sol = []
    
    # # node format: (parent, action taken to get there, state, cost)
    # node = ((), '', problem.getStartState(), 0) 
    
    # sol_node = node

    # # actual searching procedure that follows psuedocode 

    # # first checks if starting state is goal
    # if problem.isGoalState(node[2]):
    #     sol_node = node

    # # initializes frontier with starting node as only element; initializes explored set
    # frontier = util.PriorityQueue()
    # frontier.push(node, node[3] + heuristic(node[2], problem))
    # explored = set()
    # run = True

    # while run is True:
    #     curNode = frontier.pop()
    #     if problem.isGoalState(curNode[2]):
    #         sol_node = curNode
    #         run = False
    #     explored.add(curNode[2])
    #     for action in problem.getSuccessors(curNode[2]):
    #         # print(problem.getSuccessors(curNode[2]))
    #         child = (curNode, action[1], action[0], action[2])
    #         if action[0] not in explored:
    #             frontier.push(child, child[3] + heuristic(child[2], problem))
    #     # if frontier.isEmpty:
    #     #     return 
    # # once solution node is found, loop through the path from solution node to start node and append each action taken to returned list 
    # # print(sol_node)
    # while len(sol_node[1]) > 0:
    #      sol.append(sol_node[1])
    #      sol_node = sol_node[0] 
    # sol.reverse()
    # return sol
    # Jacks code:

     # same as UCS, but with the queue prioritized by f(n) = path cost + heuristic
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    frontier = util.PriorityQueue()
    explored = set()
    frontier.push((startingNode, [], 0), 0 + heuristic(startingNode, problem))
    # so it's ((node, path to node, pathcost to node), priority (which is pathcost + heuristic)
    while not frontier.isEmpty():
        temp = frontier.pop()
        node, path, pathCost= temp
        if problem.isGoalState(node):
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

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

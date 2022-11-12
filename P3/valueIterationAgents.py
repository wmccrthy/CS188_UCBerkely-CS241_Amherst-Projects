# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from inspect import istraceback
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        
        for iter in range(self.iterations):
            newVals = util.Counter()
            # delta = 0
            # U = resU
            states = self.mdp.getStates()
            for state in states:
                if self.mdp.isTerminal(state):
                    # newVals[state] = 0  # this doesn't seem to matter, initially had it, nothing changed when I took it out
                    continue
                else:
                    qvalues = []
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        qvalues.append(self.computeQValueFromValues(state, action))
                    newVals[state] = max(qvalues)
                # temp = max(qvalues)
                # print(temp)
            self.values = newVals



        "*** YOUR CODE HERE ***"

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        #  Some useful mdp methods you will use:
        #       mdp.getStates()
        #       mdp.getPossibleActions(state)
        #       mdp.getTransitionStatesAndProbs(state, action)
        #       mdp.getReward(state, action, nextState)
        #       mdp.isTerminal(state)


        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        "*** YOUR CODE HERE ***"

        to_return = 0.0
        
        # sum of expected utility of all possible successor states given a state and action 
        # is the q value associated w that state and action 
        if self.mdp.isTerminal(state):
            return self.getValue(state)
        for successor in self.mdp.getTransitionStatesAndProbs(state, action):
                suc = successor[0]
                probability = successor[1]
                to_return += probability * ( self.mdp.getReward(state, action, suc) + self.discount * self.getValue(suc))
                

        # print(to_return)
        return to_return
                

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        to_return = None
        if self.mdp.isTerminal(state):
            return to_return
        action_maxval = -99999
        for action in self.mdp.getPossibleActions(state):
                 for successor in self.mdp.getTransitionStatesAndProbs(state, action):
                    #  suc = successor[0]
                    #  suc_val = self.discount * self.getValue(suc)
                     suc_val = self.computeQValueFromValues(state, action)
                     if suc_val > action_maxval:
                         action_maxval = suc_val
                         to_return = action
                    
        return to_return
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

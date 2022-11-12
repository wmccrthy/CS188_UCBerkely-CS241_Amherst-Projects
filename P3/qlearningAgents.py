# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update 
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = util.Counter()
        # use counter to keep track of q values 
        # store values in (state, action) tuples

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # this may work more closely to computeQvaluefromValue (value iteration function)
        # though, this class does not have a values instance variable, need to figure that out 

        return self.qValues[(state, action)]
        # return qvalues stored with key of passed in action and state 

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # first approach: try same methodology and logic as in value iteration;
        # didnt work, has to compute values from the q values (value iteration does opposite)
        # framework might be on the right path now, need to check specificities of called functions

        # goal: compute value assigned to state (equiv to max q value of state)
        
        # value to return 
        to_return = 0.0
    
        # if terminal state, return 0.0
        if len(self.getLegalActions(state)) == 0:
            return to_return

        # value to track best value found across legal actions 
        action_maxval = -99999
      
        # iteratre through all possible actions, find best q value
        for action in self.getLegalActions(state):
                #  for successor in self.mdp.getTransitionStatesAndProbs(state, action):
                #      suc = successor[0]
                #     #  suc_val = self.discount * self.getValue(suc)

                     sa_val = self.getQValue(state, action)
                     if sa_val > action_maxval:
                         action_maxval = sa_val
                     
                     
        
        to_return = action_maxval
                    
        # return highest q value in state
        return to_return

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # first approach: try same methodology and logic as in value iteration 


        # goal: return action associated with best q value 
        to_return = None
        
        # if terminal, return none 
        if len(self.getLegalActions(state)) == 0:
            return to_return

        # value to track highest qvalue action (which should be returned) 
        action_maxval = -99999
        for action in self.getLegalActions(state):
                #  for successor in self.mdp.getTransitionStatesAndProbs(state, action):
                #      suc = successor[0]
                #     #  suc_val = self.discount * self.getValue(suc)
                
                # for all legal actions, check if max q value so far, ultimately store action associated w highest q val
                  sa_val = self.getQValue(state, action)
                  if sa_val > action_maxval:
                       action_maxval = sa_val
                       to_return = action
                  elif sa_val == action_maxval: 
                    # break ties randomly 
                       if util.flipCoin(random.random()):
                          action_maxval = sa_val
                          to_return = action                                     

        # return best action 
        return to_return

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
      


        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # if no valid actions, return none
        if len(legalActions) == 0:
          return action

        # else w probability epsilon, to ensure exploration, return random action
        elif util.flipCoin(self.epsilon):
          action =  random.choice(legalActions)

        # return best action otherwise (to ensure maximize reward)
        else:
          action =  self.computeActionFromQValues(state)



        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # this is where I assign qValues for self.qValues[(state, action)]


        if state != None:
          # update q val of relevant state, action key 
          self.qValues[(state, action)] += self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action))
        
        # update to next state and action
        state = nextState

        action = self.computeActionFromQValues(state)
        # what to do with reward? 

        return action

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.2,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        # weights maps features to weights 
        # 

    def getWeights(self):
        return self.weights 

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        # for feat in self.featExtractor.getFeatures(state, action):
        
        # return  self.featExtractor.getFeatures(state, action) * self.weights[self.featExtractor.getFeatures(state, action)[(state, action)]]
        return self.getWeights() * self.featExtractor.getFeatures(state, action)

        

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # self.weights[featurefunction(state,action)] = 

        # raw code from normal Qlearning agent; use to plug in weight computation where relevant 
        # regarding features, loop through each feature for a state 

        # if state != None:
        #   # for feat in self.featExtractor.getFeatures(state, action):
        #   #   self.weights[feat] += self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)) * feat
        #   self.weights[self.featExtractor.getFeatures(state, action)[(state, action)]] += self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)) * self.featExtractor.getFeatures(state, action)[(state, action)]
        # state = nextState
        # action = self.computeActionFromQValues(state)
        # # what to do with reward? 

        # return action
        feats = self.featExtractor.getFeatures(state, action)  # getting feats
        nextVal = reward + self.discount * self.computeValueFromQValues(nextState)
        curVal = self.getQValue(state, action)
        diff = nextVal - curVal

        # and then map features to weight values.
        for feature in feats:
            self.weights[feature] += self.alpha * diff * feats[feature]



      
        
        # util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

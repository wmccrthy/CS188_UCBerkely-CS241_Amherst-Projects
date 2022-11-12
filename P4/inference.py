# inference.py
# ------------
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


import itertools
import random
import busters
import game
import util

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        # take sum, divide 1/sum, multiply all values in dist by quotient of 1 and sum 
        if self.total() != 0:
            norm_factor = 1/self.total()
            for item in list(self.items()):
                self[item[0]] *= norm_factor

        # raiseNotDefined()

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        
        while True:
            # randomly choose position from distribution,
            # create random float from 0-1
            # if random float <= probability of randomly chosen position, sample that position
            # this ensures positions are sampled based on their weight (probability)
                item = random.choice(list(self.items()))
                samp_float = random.random()
                if samp_float <= item[1]/self.total():
                    return item[0]

        
      
        raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        trueDistance =  manhattanDistance(pacmanPosition, ghostPosition)
        # return this if not special 
        # handling of special jail case; 
        if noisyDistance == None:
            if ghostPosition == jailPosition:
                # noisyDistance reading is None, return probability 1 (as ghost is in jail)
                # if ghost in jail 
                return 1
            else:
                return 0
        else:
            if ghostPosition == jailPosition:
                return 0
            p_noisy_given_true = busters.getObservationProbability(noisyDistance, trueDistance)
            return p_noisy_given_true

        # raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # observation = noisy distance reading that should be passed into getObservationProb()

        legal_positions = self.allPositions
        pacman_pos = gameState.getPacmanPosition()
        beliefs = self.beliefs

        # take current belief, multipy by observation prob, 
        for pos in legal_positions:
            # belief abt position multiplied by observationProb of observation
            beliefs[pos] *= self.getObservationProb(observation, pacman_pos, pos, self.getJailPosition())
        # raiseNotDefined()
        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # utility variables 
        legal_positions = self.allPositions
        pacman_pos = gameState.getPacmanPosition()
        oldBeliefs = util.Counter()
        # beliefs = self.beliefs
        # total_prob = util.Counter()

        for pos in legal_positions:
            oldBeliefs[pos] = self.beliefs[pos]
            # initialize Counter(dictionary essentially) of oldBeliefs to use in updating self.beliefs

        # for each possible new position 
        for newPos in legal_positions:
            newBelief = 0
            # set prob = 0 
            for oldPos in legal_positions:
                # for each possible old position (sums over all possible old positions)
                # prob of new position += product of the prob that ghost is in newposition, given they were in old position, and the prob they were in that old positon to begin with
                newBelief += self.getPositionDistribution(gameState, oldPos)[newPos] * oldBeliefs[oldPos]
            self.beliefs[newPos] = newBelief
        
        # self.observeUpdate(gameState.getNoisyGhostDistances(), gameState)

        # for oldPos in legal_positions:
        #     # for each potential old pos of ghost; get prob distribution of each pos (in next state)
        #     # 
        #     newPosDist = self.getPositionDistribution(gameState, oldPos)
        #     # new probability distribution map given ghost was in oldPos

        #     for pos in legal_positions:
        #         total_prob[pos] += newPosDist[pos]
        #         # over all possible positions of ghost in t + 1, gather total prob ghost is in position pos, for all possible positions in t 

        
        # for pos in legal_positions:
        #     beliefs[pos] *= total_prob[pos]


        # raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        num_particles = self.numParticles
        positions = self.legalPositions

        # evenly distribute particles across positions 
        # list should be filled with positions (mult for each position, as each particle is a position)

        self.particles = []

        # for all legal positions, for x in range(0, num_particles/len(positions)) self.particles.append(pos) 

        # for pos in positions:
        #     for x in range(int(num_particles/len(positions))):
        #         self.particles.append(pos)


        while len(self.particles) < num_particles:
            for pos in positions:
                if len(self.particles) < num_particles:
                    self.particles.append(pos)
                else: 
                    return

        # while len(self.particles) < num_particles:
        #     pos = 0
        #     self.particles.append(positions[0])
        #     pos += 1


        "*** YOUR CODE HERE ***"
        # raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # utility variables 
        pacman_pos = gameState.getPacmanPosition()
        jail_pos = self.getJailPosition()
        legal_positions = self.legalPositions
        current_dist = self.getBeliefDistribution()

        # construct weight distribution over self.particles based on observation
        weight_dist = DiscreteDistribution()
        # thinking: initialize new weight distribution; for each pos in legal positions, 
        # update weight_dist[pos] = current_dist[pos] * self.getObservationProb(observation, pacman_pos, pos, jail_pos)

        # for pos in legal_positions:
        #     weight_dist[pos] = current_dist[pos] * self.getObservationProb(observation, pacman_pos, pos, jail_pos)

        for particle in self.particles:
            weight_dist[particle] = current_dist[particle] * self.getObservationProb(observation, pacman_pos, particle, jail_pos)

        weight_dist.normalize()

        # check for special case
        if weight_dist.total() == 0:
            self.initializeUniformly(gameState)
        else:
        # resample from weighted distribution to refill list of particles 
            self.particles = []

            while len(self.particles) < self.numParticles:
                self.particles.append(weight_dist.sample())
        

    
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"

        legal_positions = self.legalPositions
        current_dist = self.getBeliefDistribution()
        test = util.Counter()

        # for oldPos in legal_positions:
        #     test[oldPos] = self.getPositionDistribution(gameState, oldPos)

        # for oldPos in self.particles:
        #     test[oldPos] = self.getPositionDistribution(gameState, oldPos)


        # idea: construct new distribution over self.particles based on taking a time step
        
        # for each pos in legal positions, update time_dist accordingly (refer to other elapse time method)

        # after constructing this updated distribution, set self.particles = [], iterate while len(self.particles) < self.numParticles 
        # and append to self.particles by sampling from time_dist 
        
        # time_dist = DiscreteDistribution()
        # for pos in legal_positions: 
        #     newProb = 0.0
        #     for oldPos in legal_positions:
        #         newProb += test[oldPos][pos] * current_dist[oldPos]
        #     time_dist[pos] = newProb
    
        # time_dist = DiscreteDistribution()
        # for particle in self.particles: 
        #     newProb = 0.0
        #     for oldPos in legal_positions:
        #         newProb += test[oldPos][particle] * current_dist[oldPos]
        #     time_dist[particle] = newProb
        
        
        temp_particles = self.particles
        self.particles = []
        for particle in temp_particles:
            temp_dist = self.getPositionDistribution(gameState, particle)
            self.particles.append(temp_dist.sample())

        

        # while len(self.particles) < self.numParticles:
        #     self.particles.append(time_dist.sample())

        
        # raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"

        # conv list of particles into belief distribution

        beliefs = DiscreteDistribution()
        positions = self.legalPositions

        # for pos in positions:
        #     beliefs[pos] = self.particles.count(pos)/self.numParticles
        
        for particle in self.particles:
            beliefs[particle] = self.particles.count(particle)/self.numParticles
        
        return beliefs

        # idea: iterate through all possible positions, dist[pos] = the number of particles with that pos/number of particles total
        raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        

        # totally wrong below 
        # for pos in positions:
        #     for pos2 in positions:
        #         self.particles.append((pos, pos2))
        
        self.particles = []
        "*** YOUR CODE HERE ***"
        cartProd = itertools.product(self.legalPositions, self.legalPositions)

        positions = []
        for item in cartProd:
            positions.append(item)
        random.shuffle(positions)	#not sure if this is what he meant by shuffling the list of cartesian products. Doesn’t give an error, at least?

        while len(self.particles) < self.numParticles:
            for pos in positions:
                if len(self.particles) < self.numParticles:
                    self.particles.append(pos)
                else:
                    return

        "*** YOUR CODE HERE ***"
        raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacman_pos = gameState.getPacmanPosition()
        # use self.getJailPosition(i) for individual ghosts 
        legal_positions = self.legalPositions
        current_dist = self.getBeliefDistribution()

        weight_dist = DiscreteDistribution()

        # thinking: need to create weighted distribution w updated probabilities for each particle,
        # based on the observations for all ghosts 

        # for particle in self.particles:
        #     prob = 0
        #     for i in range(self.numGhosts):
        #         if observation[i] != 0:
        #             prob += self.getObservationProb(observation[i], pacman_pos, particle[i], self.getJailPosition(i)) * current_dist[particle]
        #         else:
        #             prob = 0
        #     weight_dist[particle] = prob

        # to try next

        for particle in self.particles:
            totalObsProb = 1
            for i in range(self.numGhosts):
                totalObsProb *= self.getObservationProb(observation[i], pacman_pos, particle[i], self.getJailPosition(i))
            # weighted belief of particle = observationprobability summed over all ghosts * current belief abt particle 
            weight_dist[particle] = totalObsProb * current_dist[particle]

        # then, resample from this weighted dist, and check for special case 
        weight_dist.normalize()
        if weight_dist.total() == 0:
            self.initializeUniformly(gameState)
        else:
            self.particles = []
            while len(self.particles) <self.numParticles:
                self.particles.append(weight_dist.sample())
        






        # pacman_pos = gameState.getPacmanPosition()
        # # we need to get the noisy manhattan distances to all ghosts, so we need noisy dists first
        # # print(observation)
        # # noiDists = gameState.getNoisyGhostDistances()
        # # print(noiDists)
        # # print("Test 1")

        # # probs = util.Counter()
        # current_dist = self.getBeliefDistribution()
        # jail_pos = util.Counter()
        # weight_dist = DiscreteDistribution()


        
        # for particle in self.particles:
        #     prob = current_dist[particle]
        #     for i in range(self.numGhosts):
        #         # print(particle[i])
        #         jail_pos[i] = self.getJailPosition(i)
        #         if observation[i] != None:
        #             # print(particle[i])
        #             # print(particle[i])
        #             # print(particle)
        #             prob *= observation[i] * self.getObservationProb(observation[i], pacman_pos, particle[i], jail_pos[i])
        #         else:
        #             prob = 0
        #             # particle = self.getJailPosition(i)
        #             # print(particle)
		# # trying to figure out what to do here, seems to be the issue
        #     weight_dist[particle] = prob 

        # weight_dist.normalize()

        # # check for special case
        # if weight_dist.total() == 0:
        #     self.initializeUniformly(gameState)
        # else:            # resample from weighted distribution to refill list of particles
        #     self.particles = []

        # while len(self.particles) < self.numParticles:
        #     self.particles.append(weight_dist.sample())

    
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions
            for i in range(self.numGhosts):
                # for all ghosts

                # temporary distribution (weight of time) = position distribution given list of old ghost positions,
                # index of ghost, and ghost agent 
                temp_dist = self.getPositionDistribution(gameState, newParticle, i, self.ghostAgents[i])

                # new belief for ghost i, given position distribution for old position, old belief, is sampled from
                # that distribution
                newParticle[i] = temp_dist.sample()
            newParticle = tuple(newParticle)
            newParticles.append(newParticle)

        self.particles = newParticles
        
            # now loop through and update each entry in newParticle...

        "*** YOUR CODE HERE ***"
            # raiseNotDefined()

        """*** END YOUR CODE HERE ***"""
            # newParticles.append(tuple(newParticle))

        

        # temp_particles = self.particles
        # self.particles = []
        # for particle in temp_particles:
        #     temp_dist = self.getPositionDistribution(gameState, particle)
        #     self.particles.append(temp_dist.sample())


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist

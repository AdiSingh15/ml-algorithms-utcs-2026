from Helpers.game import *
from Helpers.learningAgents import ReinforcementAgent
from Helpers.featureExtractors import *
import random
import Helpers.util
import math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent implementing:
      - getQValue
      - computeValueFromQValues
      - computeActionFromQValues
      - getAction
      - update
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.eval = False
        # qVals[state][action] -> float Q-value
        # Initialized on first access in getQValue
        self.qVals = {}

    # ------------------------------------------------------------------
    # Core Q-table access
    # ------------------------------------------------------------------

    def getQValue(self, state, action):
        """
        Return Q(state, action).
        If the state has never been seen, initialize it with 0 for all
        legal actions and return 0.
        """
        action = action.capitalize()
        if state not in self.qVals:
            # Initialize every legal action at 0 for this state
            self.qVals[state] = {a.capitalize(): 0.0
                                  for a in self.getLegalActions(state)}
        if action not in self.qVals[state]:
            self.qVals[state][action] = 0.0
        return self.qVals[state][action]

    # ------------------------------------------------------------------
    # Policy computation
    # ------------------------------------------------------------------

    def computeValueFromQValues(self, state):
        """
        Return max_a Q(state, a).
        Returns 0 at terminal states (no legal actions).
        """
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return 0.0
        best_q = float('-inf')
        for action in legal_actions:
            q = self.getQValue(state, action)
            if q > best_q:
                best_q = q
        return best_q

    def computeActionFromQValues(self, state):
        """
        Return the action with the highest Q-value.
        Returns None at terminal states.
        Breaks ties randomly among equally best actions.
        """
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None

        best_q = float('-inf')
        best_actions = []
        for action in legal_actions:
            q = self.getQValue(state, action)
            if q > best_q:
                best_q = q
                best_actions = [action]
            elif q == best_q:
                best_actions.append(action)

        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------

    def getAction(self, state):
        """
        With probability self.epsilon take a random legal action;
        otherwise take the greedy best action.
        Returns None at terminal states.
        """
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None

        # Do NOT overwrite self.epsilon -- use it as-is
        if random.random() < self.epsilon:
            return random.choice(legal_actions)   # explore
        return self.computeActionFromQValues(state)  # exploit

    # ------------------------------------------------------------------
    # Q-value update (Bellman equation)
    # ------------------------------------------------------------------

    def update(self, state, action, nextState, reward):
        """
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        action = action.capitalize()
        curr_q     = self.getQValue(state, action)
        next_max_q = self.computeValueFromQValues(nextState)
        td_target  = reward + self.discount * next_max_q
        self.qVals[state][action] = curr_q + self.alpha * (td_target - curr_q)

    # ------------------------------------------------------------------
    # Required interface methods
    # ------------------------------------------------------------------

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
      

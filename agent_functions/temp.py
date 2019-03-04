import action as Action
import numpy as np
import random
import copy

def print_grid(P, precision=4, title=None):
    """Prints a grid of probabilities."""
    if title is not None:
        print('\n' + title)
        print('-'*len(title))
    for y in range(len(P)-1, -1, -1):
        print()
        msg = ''
        str_len = 2 + precision
        for x in range(len(P)):
            if P[x][y] == 0:
                val = str_len * '-'
            elif P[x][y] == 1:
                val = '1.'
                while len(val) < str_len:
                    val += '0'
            else:
                val = '{:.{prec}}'.format(P[x][y], prec=precision-1)
                while len(val) < str_len:
                    val += ' '
            msg += ' {} '.format(val)
        print(msg)
    print()


def _adjacent_locs(loc, world_size=4):
    """Returns all adjacent locations."""
    adj = []
    x, y = loc
    # square below
    if y > 0:
        adj.append([x, y-1])
    # square above
    if y < world_size-1:
        adj.append([x, y+1])
    # square to the left
    if x > 0:
        adj.append([x-1, y])
    # square to the right
    if x < world_size-1:
        adj.append([x+1, y])
    return adj


def _recompute_probs_after_removal(P):
    return P * (np.sum(P) - 1)/np.sum(P)


def _compute_posteriors(pred_loc, class_priors, percept_loc, predictor_priors,
                        predictor_removed_priors, percept, sensor_model,
                        world_size):
    """Computes P(P_xy|B_percept_loc), using Bayes Rule"""
    pcpt_x, pcpt_y  = percept_loc
    class_prior     = class_priors[pred_loc[0]][pred_loc[1]]
    predictor_prior = predictor_priors[pcpt_x, pcpt_y]
    likelihood      = sensor_model(percept_loc, pred_loc, predictor_removed_priors, percept, world_size)
    if predictor_prior == 0:
        posterior = 0
    else:
        posterior = (likelihood * class_prior) / predictor_prior  # Bayes Rule
    return posterior

def _compute_predictor_priors(class_priors):
    """Computes the probability of a breeze in each location."""
    predictor_priors = np.zeros(class_priors.shape)  # Breeze or Stench probs
    world_size = predictor_priors.shape[0]
    for x in range(world_size):
        for y in range(world_size):
            adj_locs = _adjacent_locs([x, y], world_size)
            #
            # P(B) = P(>= 1 Pit in adj loc)
            #      = 1 - P(0 Pits in adj loc)
            #      = 1 - PI (1 - P(Pit in adj_loc)) for all adj_locs
            #
            predictor_priors[x][y] = 1 - np.prod([(1 - class_priors[ax][ay]) for ax,ay in adj_locs])
    return predictor_priors

def _adjacent_sensor_model(percept_loc, pred_loc, predictor_removed_priors, percept, world_size):
    """Sensor model for Pit -> Breeze, Wumpus -> Stench.
    Computes likelihood of a Breeze given a Pit (for specific location).
    If the prediction location is adjacent to the percept location, then
    the probability is 1. Otherwise, return the prior probability of there
    being a breeze in the percept location, with the prior updated to
    account for the removal of the Pit in question.
    """
    adj_locs = _adjacent_locs(percept_loc, world_size)
    if percept.get_breeze():
        if pred_loc in adj_locs:
            return 1
        else:
            px, py = percept_loc
            return predictor_removed_priors[px][py]
    else:
        if pred_loc in adj_locs:
            return 0
        else:
            px, py = percept_loc
            return 1 - predictor_removed_priors[px][py]

class AgentFunction:

    WORLD_SIZE = 4
    NUM_PITS = 2
    NUM_WUMPI = 1
    NUM_GOLD = 1

    def __init__(self):
        self.agent_name = 'Agent Smith'
        self.agent_loc = [0, 0]
        self.agent_dir = 'E'
        self.P = np.full((self.WORLD_SIZE, self.WORLD_SIZE), (self.NUM_PITS/15))  # prob of Pit in each square
        self.W = np.full((self.WORLD_SIZE, self.WORLD_SIZE), (self.NUM_WUMPI/15))  # prob of Wumpus in each square
        self.G = np.full((self.WORLD_SIZE, self.WORLD_SIZE), (self.NUM_GOLD/16))  # prob of Wumpus in each square
        self.P[0][0] = 0
        self.W[0][0] = 0

    def get_agent_name(self):
        return self.agent_name

    def process(self, percept):
        # Perceive
        self._update_state_pre_action(percept)
        # Think - choose action
        action = self._choose_action()
        # Think - update state due to action
        self._update_state_post_action(action)
        # Act
        return action

    def _update_state_pre_action(self, percept):
        """Updates belief state based on incoming percept."""
        self._update_P(percept)
        self._update_W(percept)
        self._update_G(percept)

    def _update_P(self, percept):
        """Updates pit belief state based on incoming percept."""
        # For each xy, compute P(P_xy|B_percept_loc)
        percept_loc = self.agent_loc
        prior_P = copy.copy(self.P)
        B = _compute_predictor_priors(prior_P)
        # Pit probabilities with 1 pit removed (see future computation in
        # _compute_B_given_P())
        prior_P_removed = _recompute_probs_after_removal(prior_P)
        B_pit_removed = _compute_predictor_priors(prior_P_removed)
        for x in range(self.WORLD_SIZE):
            for y in range(self.WORLD_SIZE):
                pred_loc = [x, y]
                self.P[x][y] = _compute_posteriors(pred_loc, prior_P,
                                                   percept_loc, B,
                                                   B_pit_removed, percept,
                                                   _adjacent_sensor_model,
                                                   self.WORLD_SIZE)
        print_grid(prior_P, title='Prior Pit probs')
        print_grid(B, title='Breeze probs')
        print_grid(self.P, title='Posterior Pit probs')

    def _update_W(self, percept):
        """Update wumpus probs for each loc based on new percept."""
        x, y = self.agent_loc
        self.W[x][y] = 0
        # TODO
        if percept.get_scream():
            self.W = np.zeros((self.WORLD_SIZE,self.WORLD_SIZE))

    def _update_G(self, percept):
        """Update gold probs for each loc based on new percept."""
        x, y = self.agent_loc
        if percept.get_glitter():
            self.G[x][y] = 1
        else:
            self.G[x][y] = 0
        x, y = self.agent_loc
        # TODO

    def _possible_actions(self):
        """Determines possible actions."""
        return [
            Action.GO_FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
        ]

    def _new_agent_state(self, action):
        """Predicts new agent state based on action."""
        new_agent_loc = copy.copy(self.agent_loc)
        new_agent_dir = copy.copy(self.agent_dir)
        if action == Action.GO_FORWARD:
            if self.agent_dir == 'N':
                if self.agent_loc[1] < self.WORLD_SIZE-1:
                    new_agent_loc[1] += 1
            elif self.agent_dir == 'E':
                if self.agent_loc[0] < self.WORLD_SIZE-1:
                    new_agent_loc[0] += 1
            elif self.agent_dir == 'S':
                if self.agent_loc[1] > 0:
                    new_agent_loc[1] -= 1
            elif self.agent_dir == 'W':
                if self.agent_loc[0] > 0:
                    new_agent_loc[0] -= 1
        elif action == Action.TURN_LEFT:
            if self.agent_dir == 'N':
                new_agent_dir = 'W'
            elif self.agent_dir == 'E':
                new_agent_dir = 'N'
            elif self.agent_dir == 'S':
                new_agent_dir = 'E'
            elif self.agent_dir == 'W':
                new_agent_dir = 'S'
        elif action == Action.TURN_RIGHT:
            if self.agent_dir == 'N':
                new_agent_dir = 'E'
            elif self.agent_dir == 'E':
                new_agent_dir = 'S'
            elif self.agent_dir == 'S':
                new_agent_dir = 'W'
            elif self.agent_dir == 'W':
                new_agent_dir = 'N'
        return new_agent_loc, new_agent_dir

    def _choose_action(self):
        """Chooses action based current belief state, which was updated by percept."""
        x, y = self.agent_loc

        # Grab gold if we know the gold is in our current loc
        if self.G[x][y] == 1:
            return Action.GRAB

        # Determine possible actions
        possible_actions = self._possible_actions()
        possible_new_locs, possible_new_dirs = self._possible_new_locs_and_dirs(possible_actions)

        # if no possible actions
        if len(possible_actions) == 0:
            return Action.NO_OP

        # Remove actions that don't result in any state change
        for a, l, d in zip(possible_actions, possible_new_locs, possible_new_dirs):
            if self.agent_loc == l and self.agent_dir == d:
                possible_actions.remove(a)
                possible_new_locs.remove(l)
                possible_new_dirs.remove(d)

        # Compute death probs for each loc due to Wumpus and Pits
        possible_action_death_probs = []
        possible_action_gold_probs = []
        for a, l in zip(possible_actions[0:], possible_new_locs[0:]) :
            x, y = l
            pit_prob = self.P[x][y]
            wumpus_prob = self.W[x][y]
            gold_prob = self.G[x][y]
            death_prob = pit_prob + wumpus_prob
            possible_action_death_probs.append(death_prob)
            possible_action_gold_probs.append(gold_prob)
            # print('action: {}'.format(a))
            # print('pit_prob: {}'.format(pit_prob))
            # print('wumpus_prob: {}'.format(wumpus_prob))
            # print('death_prob: {}'.format(death_prob))
            # print('gold_prob: {}'.format(gold_prob))

        # Transform action death probs and gold probs into action probs
        possible_action_death_probs = np.array(possible_action_death_probs)
        if len(possible_action_death_probs) == 1:
            action_probs = np.ones(1)
        else:
            # Take complement of death probs and normalize
            death_action_probs = np.ones((len(possible_action_death_probs))) - possible_action_death_probs
            possible_action_gold_probs = np.array(possible_action_gold_probs)
            # Combine (take weighted mean) death action probs and gold action probs
            # Experimental Formula to combine death probabilities and gold probabilities into action probabilities
            # Note that death_action_probs are overly stated, e.g. a death prob of .05 will result in a death
            # action prob of .95
            action_probs = (.25 * death_action_probs) + (1. * possible_action_gold_probs)
            # normalize
            action_probs = [float(i)/sum(action_probs) for i in action_probs]

        print('possible_action_death_probs: {}'.format(possible_action_death_probs))
        print('possible_action_gold_probs: {}'.format(possible_action_gold_probs))
        print('action probabilities: {}'.format(action_probs))

        # TODO 02/15/2019 - Don't use nondeterminism
        # Randomly choose action based on action probabilities
        return np.random.choice(possible_actions, p=action_probs)

    def _possible_new_locs_and_dirs(self, possible_actions):
        """Returns respective agent locations and directons for each action taken."""
        possible_new_locs = []
        possible_new_dirs = []
        for a in possible_actions:
            # find resulting agent location and direction
            new_loc, new_dir = self._new_agent_state(a)
            possible_new_locs.append(new_loc)
            possible_new_dirs.append(new_dir)
        return possible_new_locs, possible_new_dirs


    def _update_state_post_action(self, action):
        self.agent_loc, self.agent_dir = self._new_agent_state(action)

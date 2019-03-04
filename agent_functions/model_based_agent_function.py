import action as Action
import numpy as np
import random
import copy

def print_grid(P, precision=3, title=None):
    """Prints a grid of probabilities."""
    if title is not None:
        print('\n' + title)
        print('-'*len(title))
    for y in range(len(P)-1, -1, -1):
        print()
        msg = ''
        str_len = 2 + precision
        for x in range(len(P)):
            val = P[x][y]
            val = round(val, precision)
            if val == 0:
                str_val = str_len * '-'
            elif val == 1:
                str_val = '1.'
                while len(str_val) < str_len:
                    str_val += '0'
            else:
                # str_val = '{:.{prec}}'.format(P[x][y], prec=precision-1)
                str_val = str(val)
                while len(str_val) < str_len:
                    str_val += ' '
            msg += ' {} '.format(str_val)
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


def _forward_loc(agent_loc, agent_dir, world_size):
    """Returns location in front of agent. If facing a wall, return None."""
    x, y = agent_loc
    if agent_dir == 'N':
        if y < world_size-1:
            return (x, y+1)
        else:
            return None
    elif agent_dir == 'E':
        if x < world_size-1:
            return (x+1, y)
        else:
            return None
    elif agent_dir == 'S':
        if y > 0:
            return (x, y-1)
        else:
            return None
    elif agent_dir == 'W':
        if x > 0:
            return (x-1, y)
        else:
            return None


def _recompute_probs_after_removal(P):
    _sum = np.sum(P)
    if _sum == 0:
        return np.zeros(P.shape)
    else:
        return P * (_sum - 1)/_sum


def _compute_predictor_priors(ClassPriors, world_size):
    """Computes the probability of a breeze in each location."""
    PredictorPriors = np.zeros((world_size, world_size))  # Breeze probs
    # num_unknown_locs = np.where((P > 0) & (P < 1))  # a location is unknown if it's prob isn't 1 or 0
    for x in range(world_size):
        for y in range(world_size):
            adj_locs = _adjacent_locs([x, y], world_size)
            #
            # P(B) = P(>= 1 Pit in adj loc)
            #      = 1 - P(0 Pits in adj loc)
            #      = 1 - PI (1 - P(Pit in adj_loc)) for all adj_locs
            #
            PredictorPriors[x][y] = 1 - np.prod([(1 - ClassPriors[ax][ay]) for ax,ay in adj_locs])
    return PredictorPriors


def _compute_likelihood(percept_loc, pred_loc, PredictorPriorsRemoved, world_size, percept, is_percept_on):
    """Computes likelihood of a Breeze given a Pit (for specific location).
    If the prediction location is adjacent to the percept location, then
    the probability is 1. Otherwise, return the prior probability of there
    being a breeze in the percept location, with the prior updated to
    account for the removal of the Pit in question.
    """
    adj_locs = _adjacent_locs(percept_loc, world_size)
    if is_percept_on(percept):
        if pred_loc in adj_locs:
            return 1
        else:
            px, py = percept_loc
            return PredictorPriorsRemoved[px][py]
    else:
        if pred_loc in adj_locs:
            return 0
        else:
            px, py = percept_loc
            return 1 - PredictorPriorsRemoved[px][py]

def _compute_posterior(pred_loc, percept_loc, ClassPriors, PredictorPriors, PredictorPriorsRemoved, world_size, percept, is_percept_on):
    """Computes P(P_xy|B_percept_loc), using Bayes Rule"""
    pcpt_x, pcpt_y = percept_loc

    if is_percept_on(percept):
        predictor_prior = PredictorPriors[pcpt_x][pcpt_y]
    else:
        predictor_prior = 1 - PredictorPriors[pcpt_x][pcpt_y]

    likelihood      = _compute_likelihood(percept_loc, pred_loc, PredictorPriorsRemoved, world_size, percept, is_percept_on)
    class_prior     = ClassPriors[pred_loc[0]][pred_loc[1]]
    if predictor_prior == 0:
        posterior = 0
    else:
        posterior = (likelihood * class_prior) / predictor_prior  # Bayes Rule
    return posterior


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
        self.Pcpt = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))  # whether a percept has been processed in each loc
        self.P[0][0] = 0
        self.W[0][0] = 0
        self._update_D() #  sets self.D, i.e. prob of Death in each square
        self.last_action = None
        self.has_arrow = True

    def get_agent_name(self):
        return self.agent_name

    def process(self, percept):
        ax, ay = self.agent_loc
        # PERCEIVE
        # Only update state if the percept is new
        if self.Pcpt[ax][ay] == 0:
            self._update_P(percept)
            self._update_W(percept)
            self._update_G(percept)
            self._update_D()
        if self.last_action == Action.SHOOT:
            # Only update Wumpus Probabilities
            self._update_W(percept)
            self._update_D()
        # THINK
        action = self._choose_action()
        # ACT
        self._update_state_post_action(action)
        return action

    def _update_P(self, percept):
        """Updates pit belief state based on incoming percept."""
        percept_loc = self.agent_loc
        ax, ay = self.agent_loc
        self.P[ax][ay] = 0
        # For each xy, compute P(P_xy|B_percept_loc)
        prior_P = copy.copy(self.P)
        B = _compute_predictor_priors(prior_P, self.WORLD_SIZE)  # Breeze probabilities
        prior_P_removed = _recompute_probs_after_removal(prior_P)  # Pit probabilities with 1 pit removed (see future computation in _compute_likelihood())
        B_pit_removed = _compute_predictor_priors(prior_P_removed, self.WORLD_SIZE)
        is_percept_on = lambda p: p.get_breeze()
        for x in range(self.WORLD_SIZE):
            for y in range(self.WORLD_SIZE):
                pred_loc = [x, y]
                self.P[x][y] = _compute_posterior(pred_loc, percept_loc, prior_P, B, B_pit_removed, self.WORLD_SIZE, percept, is_percept_on)
        # print_grid(prior_P, title='Prior Pit probs')
        # print_grid(prior_P_removed, title='Prior Pit Removed probs')
        # print_grid(B, title='Breeze probs')
        # print_grid(self.P, title='Posterior Pit probs')

    def _update_W(self, percept):
        """Updates wumpus belief state based on incoming percept."""
        percept_loc = self.agent_loc
        ax, ay = self.agent_loc
        self.W[ax][ay] = 0
        if self.last_action == Action.SHOOT:
            # If killed Wumpus, clear all Wumpus probabilities
            if percept.get_scream():
                self.W = np.zeros(self.W.shape)
                # print_grid(self.W, title='Posterior Wumpus probs')
                return
            # If missed, Wumpus is not in forward cell
            else:
                forward_loc = _forward_loc(self.agent_loc, self.agent_dir, self.WORLD_SIZE)
                if forward_loc is not None:
                    fx, fy = forward_loc
                    self.W[fx][fy] = 0

        prior_W = copy.copy(self.W)
        St = _compute_predictor_priors(prior_W, self.WORLD_SIZE)  # Stench probabilities
        prior_W_removed = _recompute_probs_after_removal(prior_W)  # Wumpus probabilities with 1 pit removed (see future computation in _compute_likelihood())
        St_W_removed = _compute_predictor_priors(prior_W_removed, self.WORLD_SIZE)
        is_percept_on = lambda p: p.get_stench()
        for x in range(self.WORLD_SIZE):
            for y in range(self.WORLD_SIZE):
                pred_loc = [x, y]
                self.W[x][y] = _compute_posterior(pred_loc, percept_loc, prior_W, St, St_W_removed, self.WORLD_SIZE, percept, is_percept_on)
        # print_grid(prior_W, title='Prior Wumpus probs')
        # print_grid(St, title='Stench probs')
        # print_grid(self.W, title='Posterior Wumpus probs')

    def _update_G(self, percept):
        """Update gold probs for each loc based on new percept."""
        percept_loc = self.agent_loc
        ax, ay = self.agent_loc
        self.G[ax][ay] = 0
        if percept.get_glitter():
            self.G[ax][ay] = 1
        else:
            self.G[ax][ay] = 0
        unknown_indices = np.where((self.G != 0) & (self.G != 1))
        # NOTE - the above indices are numpy indices, not cartesian x,y indices
        # E.g. [[x1, y1], [x2, y2]] would look like (array([x1, x2]), array([y1, y2])) in numpy
        unknown_loc_count = len(unknown_indices[0])
        if unknown_loc_count > 0:
            new_unknown_prob = 1. * self.NUM_GOLD / unknown_loc_count
            self.G[unknown_indices] = new_unknown_prob
        # print_grid(self.G, title='Posterior Gold probs')

    def _update_D(self):
        """Updates death probs by combining Pit and Wumpus probs.
        P(death) = 1 - !death
        P(death) = 1 - (!W)(!P)
        """
        self.D = 1 - ((1 - self.P)*(1 - self.W))
        print_grid(self.D, title='Posterior Death probs')

    def _choose_action(self):
        """Chooses action based current belief state, which was updated by percept."""
        # Grab gold if we know the gold is in our current loc
        ax, ay = self.agent_loc
        if self.G[ax][ay] == 1:
            print('Glitter => Grabbing')
            return Action.GRAB

        # Try to shoot Wumpus if pretty sure that it's in front of agent
        forward_loc = _forward_loc(self.agent_loc, self.agent_dir, self.WORLD_SIZE)
        if forward_loc is not None:
            fx, fy = forward_loc
            if self.has_arrow and self.W[fx][fy] > .33:
                print('Shooting')
                return Action.SHOOT

        # Determine possible actions
        possible_actions = self._possible_actions()
        possible_new_locs, possible_new_dirs = self._possible_new_locs_and_dirs(possible_actions)

        # if no possible actions
        if len(possible_actions) == 0:
            print('No possible actions. Returning NO_OP')
            return Action.NO_OP

        DEATH_PROB_THRESH = .33
        NO_OPP_PROB = .01

        # Prune useless actions
        approved_actions = []
        approved_new_locs = []
        approved_new_dirs = []
        for a, l, d in zip(possible_actions, possible_new_locs, possible_new_dirs):
            # Remove actions that don't result in any state change
            if self.agent_loc == l and self.agent_dir == d:
                print('removing {} from possible actions since agent\' state doesn\'t change'.format(Action.print_action(a)))
                continue
            x, y = l
            if a in [Action.TURN_LEFT, Action.TURN_RIGHT]:
                # Remove actions that result in agent turning to face a wall
                if ((x == self.WORLD_SIZE-1 and d == 'E') or
                    # TODO debug below line not working
                    (y == self.WORLD_SIZE-1 and d == 'N') or
                    (x == 0 and d == 'W') or
                    (y == 0 and d == 'S')):
                    print('removing {} from possible actions since agent would face a wall.'.format(Action.print_action(a)))
                    continue
                # Remove actions that result in agent facing a high death prob
                # forward_loc = _forward_loc(l, d, self.WORLD_SIZE)
                # if forward_loc is not None:
                #     fx, fy = forward_loc
                #     # TODO debug below lines not working
                #     if self.D[fx][fy] > DEATH_PROB_THRESH:
                #         print('removing {} from possible actions since agent would face high death prob.'.format(Action.print_action(a)))
                #         continue
            approved_actions.append(a)
            approved_new_locs.append(l)
            approved_new_dirs.append(d)

        if len(approved_actions) == 0:
            raise ValueError('approved actions cannot be zero')

        # Compute death and gold probs for each loc
        action_death_probs = []
        action_gold_probs = []
        for a, l in zip(approved_actions[0:], approved_new_locs[0:]) :
            x, y = l
            gold_prob = self.G[x][y]
            death_prob = self.D[x][y]
            action_death_probs.append(death_prob)
            action_gold_probs.append(gold_prob)

        # TODO 02/22/2019 - prefer a loc where no Percept has been received yet (unexplored loc)

        if approved_actions[0] == Action.GO_FORWARD and action_death_probs[0] < DEATH_PROB_THRESH:
            print('Moving forward since path is clear.')
            return approved_actions[0]
        elif approved_actions[0] == Action.GO_FORWARD and len(approved_actions) == 1 :
            print('No safe move, returning NO OP')
            return Action.NO_OP
        # If option to turn left or right, turn in the direction with lower death prob
        elif len(approved_actions) == 2 and Action.TURN_LEFT in approved_actions and Action.TURN_RIGHT in approved_actions:
            min_idx = np.argmin(action_death_probs)
            print('min_idx: {}'.format(min_idx))
            print('approved_actions[min_idx]: {}'.format(approved_actions[min_idx]))
            return approved_actions[min_idx]
        elif len(approved_actions) == 1:
            print('Returning only action in approved actions.')
            return approved_actions[0]
        else:
            # Randomly choose between stationary strategies (turn, no opp).
            print('approved_actions: {}'.format(Action.print_actions(approved_actions)))
            turn_prob = (1-NO_OPP_PROB)/(len(approved_actions)-1)
            stationary_action_probs = np.full(np.shape(approved_actions), turn_prob)
            stationary_action_probs[-1] = NO_OPP_PROB
            stationary_actions = approved_actions[1:] + [Action.NO_OP]
            return np.random.choice(stationary_actions, p=stationary_action_probs)

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
        x, y = self.agent_loc
        if action == Action.GO_FORWARD:
            if self.agent_dir == 'N':
                if y < self.WORLD_SIZE-1:
                    new_agent_loc[1] += 1
            elif self.agent_dir == 'E':
                if x < self.WORLD_SIZE-1:
                    new_agent_loc[0] += 1
            elif self.agent_dir == 'S':
                if y > 0:
                    new_agent_loc[1] -= 1
            elif self.agent_dir == 'W':
                if x > 0:
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
        ax0, ay0 = self.agent_loc  # original agent loc
        self.Pcpt[ax0][ay0] = 1
        self.agent_loc, self.agent_dir = self._new_agent_state(action)
        self.last_action = action
        if action == Action.SHOOT:
            self.has_arrow = False


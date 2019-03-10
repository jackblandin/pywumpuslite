import action as Action
import numpy as np
import random
import copy


class EnvConfig():
    def __init__(self, world_size, num_pits, num_wumpi, num_gold):
        self.world_size = world_size
        self.num_pits = num_pits
        self.num_wumpi = num_wumpi
        self.num_gold = num_gold


class BeliefState():

    def __init__(self, agent_loc, agent_dir, P, W, G, PcptMap, has_arrow,
                 last_action, env_config):
        self.agent_loc = agent_loc
        self.agent_dir = agent_dir
        self.P = P
        self.W = W
        self.G = G
        self.D = self._compute_D()
        self.PcptMap = PcptMap
        self.has_arrow = has_arrow
        self.last_action = last_action
        self.WORLD_SIZE = env_config.world_size
        self.NUM_PITS = env_config.num_pits
        self.NUM_WUMPI = env_config.num_wumpi
        self.NUM_GOLD = env_config.num_gold

    def update(self, percept):
        """Belief update. This is the state estimator function (SE(b, a, o).
        Note that 'a' is passed via self.last_action.
        """
        # Only update state if the percept is new
        ax, ay = self.agent_loc
        if self.PcptMap[ax][ay] == 0:
            self._update_P(percept)
            self._update_W(percept)
            self._update_G(percept)
            self.D = self._compute_D()
        if self.last_action == Action.SHOOT:
            # Only update Wumpus Probabilities
            self._update_W(percept)
            self.D = self._compute_D()

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

    def _compute_D(self):
        """Updates death probs by combining Pit and Wumpus probs.
        P(death) = 1 - !death
        P(death) = 1 - (!W)(!P)
        """
        D = 1 - ((1 - self.P)*(1 - self.W))
        print_grid(D, title='Posterior Death probs')
        return D


class Transition():
    def __init__(self, initial_state, action, new_state):
        """A transition is essentially just a <initial_state, action, new_state> tuple."""
        self.initial_state = initial_state
        self.action = action
        self.new_state = new_state


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

def _transition_func(state, action, env_config):
    """Predicts new agent belief state based on action."""
    new_agent_loc = copy.copy(state.agent_loc)
    new_agent_dir = copy.copy(state.agent_dir)
    x, y = state.agent_loc
    world_size = env_config.world_size
    if action == Action.GO_FORWARD:
        if state.agent_dir == 'N':
            if y < world_size-1:
                new_agent_loc[1] += 1
        elif state.agent_dir == 'E':
            if x < world_size-1:
                new_agent_loc[0] += 1
        elif state.agent_dir == 'S':
            if y > 0:
                new_agent_loc[1] -= 1
        elif state.agent_dir == 'W':
            if x > 0:
                new_agent_loc[0] -= 1
    elif action == Action.TURN_LEFT:
        if state.agent_dir == 'N':
            new_agent_dir = 'W'
        elif state.agent_dir == 'E':
            new_agent_dir = 'N'
        elif state.agent_dir == 'S':
            new_agent_dir = 'E'
        elif state.agent_dir == 'W':
            new_agent_dir = 'S'
    elif action == Action.TURN_RIGHT:
        if state.agent_dir == 'N':
            new_agent_dir = 'E'
        elif state.agent_dir == 'E':
            new_agent_dir = 'S'
        elif state.agent_dir == 'S':
            new_agent_dir = 'W'
        elif state.agent_dir == 'W':
            new_agent_dir = 'N'

    new_PcptMap = np.copy(state.PcptMap)
    new_PcptMap[x][y] = 1
    new_P = np.copy(state.P)
    new_W = np.copy(state.W)
    new_G = np.copy(state.G)
    has_arrow = (state.has_arrow) and (action != Action.SHOOT)
    new_state = BeliefState(new_agent_loc, new_agent_dir, new_P, new_W,
                            new_G, new_PcptMap, has_arrow, action,
                            env_config)
    return new_state


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
        self.agent_name = 'Sterling Archer'
        P = np.full((self.WORLD_SIZE, self.WORLD_SIZE), (self.NUM_PITS/15))  # prob of Pit in each square
        W = np.full((self.WORLD_SIZE, self.WORLD_SIZE), (self.NUM_WUMPI/15))  # prob of Wumpus in each square
        G = np.full((self.WORLD_SIZE, self.WORLD_SIZE), (self.NUM_GOLD/16))  # prob of Wumpus in each square
        P[0][0] = 0
        W[0][0] = 0
        PcptMap = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))  # whether a percept has been processed in each loc
        self.env_config = EnvConfig(self.WORLD_SIZE,
                                    self.NUM_PITS,
                                    self.NUM_WUMPI,
                                    self.NUM_GOLD)
        self.belief_state = BeliefState([0,0], 'E', P, W, G, PcptMap, True,
                                        None, self.env_config)

    def get_agent_name(self):
        return self.agent_name

    def process(self, percept):
        # PERCEIVE
        self.belief_state.update(percept)
        # THINK
        action = self._choose_action()
        # ACT
        self.belief_state = _transition_func(self.belief_state, action,
                                             self.env_config)
        return action

    def _choose_action(self):
        """Chooses action based current belief state, which was updated by percept."""
        # Grab gold if we know the gold is in our current loc
        ax, ay = self.belief_state.agent_loc
        if self.belief_state.G[ax][ay] == 1:
            print('Glitter => Grabbing')
            return Action.GRAB

        # Try to shoot Wumpus if pretty sure that it's in front of agent
        forward_loc = _forward_loc(self.belief_state.agent_loc, self.belief_state.agent_dir, self.WORLD_SIZE)
        if forward_loc is not None:
            fx, fy = forward_loc
            if self.belief_state.has_arrow and self.belief_state.W[fx][fy] > .33:
                print('Shooting')
                return Action.SHOOT

        # Determine possible actions
        possible_actions = self._possible_actions()

        # if no possible actions
        # TODO 03/10/2019 - Right now, agent never returns NO_OP, since it will
        # just oscillate b/w turn left/right when it's stuck. If this issue is
        # not fixed through search, then we should put in a hack to return a
        # NO_OP when it's stuck oscillating.
        if len(possible_actions) == 0:
            print('No possible actions. Returning NO_OP')
            return Action.NO_OP

        DEATH_PROB_THRESH = .33

        # Prune useless actions
        # A transition is essentially just a <initial_state, action, new_state> tuple
        possible_transitions = [Transition(self.belief_state, a, _transition_func(self.belief_state, a, self.env_config)) for a in possible_actions]
        approved_transitions = []
        for t in possible_transitions:
            # Remove actions that don't result in any state change
            a = t.action
            l = t.new_state.agent_loc
            d = t.new_state.agent_dir
            print('a: {}, l: {}, d: {}'.format(a,l,d))
            if self.belief_state.agent_loc == l and self.belief_state.agent_dir == d:
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
            approved_transitions.append(t)

        if len(approved_transitions) == 0:
            raise ValueError('approved transitions cannot be zero')

        approved_actions = [t.action for t in approved_transitions]

        # Remove GO_FORWARD actions with high death prob
        if Action.GO_FORWARD in approved_actions:
            go_forw_idx = approved_actions.index(Action.GO_FORWARD)
            t = approved_transitions[go_forw_idx]
            x, y = t.new_state.agent_loc
            if (t.new_state.D[x][y] >= DEATH_PROB_THRESH):
                del approved_actions[go_forw_idx]
                del approved_transitions[go_forw_idx]
                print('Removing GO_FORWARD since the death prob is too high.')

        # Preference order:
        #   1. GO_FORWARD
        #   2. TURN LEFT/RIGHT
        if Action.GO_FORWARD in approved_actions:
            print('Moving forward since path is clear.')
            return Action.GO_FORWARD
        elif len(approved_actions) == 2 and Action.TURN_LEFT in approved_actions and Action.TURN_RIGHT in approved_actions:
            # If faced with both left and right options, choose action that
            # results in facing a lower death probablity

            # TURN LEFT -> GO_FORWARD (two-step lookahead)
            idx_left = approved_actions.index(Action.TURN_LEFT)
            state_left = approved_transitions[idx_left].new_state
            left_forw_loc = _forward_loc(state_left.agent_loc, state_left.agent_dir, self.env_config.world_size)
            lx, ly = left_forw_loc

            # TURN LEFT -> GO_FORWARD (two-step lookahead)
            idx_right = approved_actions.index(Action.TURN_RIGHT)
            state_right = approved_transitions[idx_right].new_state
            right_forw_loc = _forward_loc(state_right.agent_loc, state_right.agent_dir, self.env_config.world_size)
            rx, ry = right_forw_loc

            print("TL, TR only options, choosing to turn in direction of facing new state with lower death prob of forward dir.")
            print("state_left.D[lx][ly] = {}".format(state_left.D[lx][ly]))
            print("state_right.D[rx][ry] = {}".format(state_right.D[rx][ry]))
            if state_left.D[lx][ly] < state_right.D[rx][ry]:
                return Action.TURN_LEFT
            elif state_left.D[lx][ly] < state_right.D[rx][ry]:
                return Action.TURN_RIGHT
            else:
                # TL/TR have equal prob => Choose LEFT (on avg leads to unexplored loc)
                return Action.TURN_LEFT

            # Randomly choose b/w left, and right
            # return np.random.choice(approved_actions, p=[.5, .5])
        else:
            if len(approved_actions) > 1:
                print('approved_actions: {}'.format(approved_actions))
                raise Exception('Should never reach here with more than one approved action')
            else:
                print('Returning the only approved action')
                return approved_actions[0]

    def _possible_actions(self):
        """Determines possible actions."""
        return [
            Action.GO_FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
        ]

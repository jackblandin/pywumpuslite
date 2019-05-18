import action as Action
import numpy as np
import random
import copy

##########
# Search #
##########

def _path_to_loc(cur_loc, dest_loc, explored_locs):
    """
    Using Depth First Search, computes the path from the current loc to the destination loc.
    Only explored nodes will be used to get there.
    The object returned will be the path of the Node, which is a sequence of locs.
    """

    class Node:
        def __init__(self, parent_node, loc):
            self.parent_node = parent_node
            self.loc = loc
        def path(self):
            """Builds the path of nodes obtained to get here."""
            path = [self.loc]
            if self.parent_node is None:
                return path
            else:
                return self.parent_node.path() + path

    cur_loc_node = Node(None, cur_loc)

    e = []  # explored (note that this is different than explored_locs param)
    f = []  # frontier

    # Initial goal check
    if dest_loc == cur_loc:
        return cur_loc_node.path()

    # Add current loc to explored
    e.append(cur_loc)

    # Build initial frontier with locs adj. to cur. loc
    adj_to_cur = _adjacent_locs(cur_loc)
    for l in (explored_locs + [dest_loc]):
        if l in adj_to_cur:
            f.insert(0, Node(cur_loc_node, l))

    def _expand(parent):
        adj_locs = _adjacent_locs(parent.loc)
        adj_and_explored_locs = []
        for l in adj_locs:
            if (l in explored_locs or l == dest_loc) and l not in e:
                child = Node(parent, l)
                adj_and_explored_locs.append(child)
        return adj_and_explored_locs

    while len(f) > 0:
        n = f.pop()
        e.append(n.loc)
        if n.loc == dest_loc:
            return n.path()
        else:
            f = _expand(n) + f

    print('cur_loc: {}'.format(cur_loc))
    print('dest_loc: {}'.format(dest_loc))
    print('explored_locs: {}'.format(explored_locs))

    raise Exception('No action sequence found')


def _action_seq_to_adj_loc(cur_loc, cur_dir, dest_loc, actions=None):
    """Called recursively.
    Given current loc, current direction, and the destination loc, which must be adjacent to current loc,
    computes the action sequence needed to get there as well as the final direction.
    """
    if actions is None:
        actions = []
    # Check for stopping criteria
    if _forward_loc(cur_loc, cur_dir) == dest_loc:
        actions.append(Action.GO_FORWARD)
        return actions, cur_dir
    else:
        # Turn Left (could also turn Right)
        actions.append(Action.TURN_LEFT)
        new_dir = _turn_left_dir(cur_dir)
        # Recurse
        return _action_seq_to_adj_loc(cur_loc, new_dir, dest_loc, actions)


def _action_seq_to_non_adj_loc(cur_loc, cur_dir, dest_loc, explored_locs):
    """
    Given current loc, current direction, and the destination loc, computes the action sequence needed to get there.
    Uses depth first search by calling _path_to_loc.
    """
    path = _path_to_loc(cur_loc, dest_loc, explored_locs)
    d = cur_dir
    actions = []
    for idx in range(len(path)-1):
        actions, d = _action_seq_to_adj_loc(path[idx], d, path[idx+1], actions)
    return actions


def _search_heuristic(belief_state, frontier_loc):
        def manhattan(loc1, loc2):
            return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
        al = belief_state.agent_loc
        fx, fy = frontier_loc
        d = belief_state.D[fx][fy]
        m = manhattan(al, frontier_loc)
        # Only compare manhattan distance if death probabilities are equal
        return 100*d + m


#################

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
        return D


class Transition():
    def __init__(self, initial_state, action, new_state):
        """A transition is essentially just a <initial_state, action, new_state> tuple."""
        self.initial_state = initial_state
        self.action = action
        self.new_state = new_state


def print_grid(P, precision=2, title=None):
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


def _create_grid_from_locs(locs, world_size=4):
    """
    Creates a world_size x world_size grid with 1s for provided locs and zeros
    for missing locs.
    """
    grid = np.zeros((world_size, world_size))
    for x,y in locs:
        grid[x][y] = 1
    return grid


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


def _forward_loc(agent_loc, agent_dir, world_size=4):
    """Returns location in front of agent. If facing a wall, return None."""
    x, y = agent_loc
    if agent_dir == 'N':
        if y < world_size-1:
            return [x, y+1]
        else:
            return None
    elif agent_dir == 'E':
        if x < world_size-1:
            return [x+1, y]
        else:
            return None
    elif agent_dir == 'S':
        if y > 0:
            return [x, y-1]
        else:
            return None
    elif agent_dir == 'W':
        if x > 0:
            return [x-1, y]
        else:
            return None

def _turn_left_dir(agent_dir):
    """Returns new direction after turning left."""
    if agent_dir == 'N':
        new_agent_dir = 'W'
    elif agent_dir == 'E':
        new_agent_dir = 'N'
    elif agent_dir == 'S':
        new_agent_dir = 'E'
    elif agent_dir == 'W':
        new_agent_dir = 'S'
    return new_agent_dir


def _turn_right_dir(agent_dir):
    """Returns new direction after turning right."""
    if agent_dir == 'N':
        new_agent_dir = 'E'
    elif agent_dir == 'E':
        new_agent_dir = 'S'
    elif agent_dir == 'S':
        new_agent_dir = 'W'
    elif agent_dir == 'W':
        new_agent_dir = 'N'
    return new_agent_dir

def _transition_func(state, action, env_config):
    """Predicts new agent belief state based on action."""
    new_agent_loc = copy.copy(state.agent_loc)
    new_agent_dir = copy.copy(state.agent_dir)
    x, y = state.agent_loc
    world_size = env_config.world_size
    if action == Action.GO_FORWARD:
        forward_loc = _forward_loc(state.agent_loc, state.agent_dir)
        if forward_loc is not None:
            new_agent_loc = forward_loc
    elif action == Action.TURN_LEFT:
        new_agent_dir = _turn_left_dir(state.agent_dir)
    elif action == Action.TURN_RIGHT:
        new_agent_dir = _turn_right_dir(state.agent_dir)
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
        self.DEATH_PROB_THRESH = .8
        self.WUMPUS_PROB_SHOOT_TRESH = .33
        self.belief_state = BeliefState([0,0], 'E', P, W, G, PcptMap, True,
                                        None, self.env_config)
        self.planned_actions = []
        self.frontier_locs = []
        self.explored_locs = []

    def get_agent_name(self):
        return self.agent_name

    def process(self, percept):
        # PERCEIVE
        self.belief_state.update(percept)
        # THINK
        action = self._next_action(percept)
        # ACT
        self.belief_state = _transition_func(self.belief_state, action,
                                             self.env_config)
        return action

    def _next_action(self, percept):
        """
        Computes the next action to take, based on the existing plan. If no
        plan exists, create one.
        """
        # Add loc to explored
        if self.belief_state.agent_loc not in self.explored_locs:
            self.explored_locs.append(self.belief_state.agent_loc)
        # If arriving in a frontier loc, remove loc from frontier
        if self.belief_state.agent_loc in self.frontier_locs:
            self.frontier_locs.remove(self.belief_state.agent_loc)
        # If on path to a specific frontier loc, keep going, otherwise, develop
        # a new plan.
        if len(self.planned_actions) == 0:
            self.planned_actions = self._compute_plan(percept)
        print('planned_actions: {}'.format(self.planned_actions))
        action = self.planned_actions.pop(0)
        # Print out the frontier and explored grids
        frontier_grid = _create_grid_from_locs(self.frontier_locs)
        explored_grid = _create_grid_from_locs(self.explored_locs)
        print_grid(frontier_grid, title="Frontier")
        print_grid(explored_grid, title="Explored")
        print_grid(self.belief_state.D, title='Posterior Death probs')
        return action

    def _compute_plan(self, percept):
        """Determines the next sequence of actions to take.
        1. If glitter => [GRAB]
        2. If high Wumpus probability => [SHOOT]
        3. Otherwise, determine which frontier loc has highest expected value,
           and return a sequence of actions to get to this loc.
        """
        assert len(self.planned_actions) == 0

        # Grab gold if we know the gold is in our current loc
        ax, ay = self.belief_state.agent_loc
        if self.belief_state.G[ax][ay] == 1:
            return [Action.GRAB]

        # Try to shoot Wumpus if pretty sure that it's in front of agent
        forward_loc = _forward_loc(self.belief_state.agent_loc, self.belief_state.agent_dir, self.WORLD_SIZE)
        if forward_loc is not None:
            fx, fy = forward_loc
            if self.belief_state.has_arrow and self.belief_state.W[fx][fy] > self.WUMPUS_PROB_SHOOT_TRESH:
                return [Action.SHOOT]

        # TODO 03/10/2019 Add info gathering to Utility function
        # TODO 03/10/2019 Sort explored list ahead of time to save on time

        # Compute new frontier locs
        adj_locs = _adjacent_locs(self.belief_state.agent_loc)
        for x, y in adj_locs:
            # Don't include current loc in frontier locs
            if self.belief_state.agent_loc[0] == x and self.belief_state.agent_loc[1] == y:
                continue
            # Don't include if already in explored
            if [x, y] in self.explored_locs:
                continue
            # Don't include if already in frontier locs
            if [x, y] in self.frontier_locs:
                continue
            # Don't include high death prob locs
            if self.belief_state.D[x][y] >= self.DEATH_PROB_THRESH:
                continue

            # Sort frontier using
            # 1. Death probablity
            # 2. Manhattan distance (city block)
            inserted = False
            for idx, f in enumerate(self.frontier_locs):
                hf = _search_heuristic(self.belief_state, f)
                h = _search_heuristic(self.belief_state, [x, y])
                if h < hf:
                    self.frontier_locs.insert(idx, [x, y])
                    inserted = True
                    break
            if not inserted:
                self.frontier_locs.append([x, y])

        # If no safe frontier locs, return NO OP
        if len(self.frontier_locs) == 0:
            return 100 * [Action.NO_OP]

        f = self.frontier_locs[0]
        print('selected_frontier_loc: {}'.format(f))
        actions = _action_seq_to_non_adj_loc(self.belief_state.agent_loc,
                                             self.belief_state.agent_dir,
                                             f,
                                             self.explored_locs)
        return actions

    def _possible_actions(self):
        """Determines possible actions."""
        return [
            Action.GO_FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
        ]


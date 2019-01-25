from random import randint
from agent_function import AgentFunction


class Agent:

    def __init__(self, world, percept, non_deterministic_mode):
        self.world = world
        self.percept = percept
        self.non_deterministic_mode = non_deterministic_mode
        self.is_dead = False
        self.has_gold = False
        self.agent_function = AgentFunction()
        self.location = world.get_agent_location()
        self.direction = world.get_agent_direction()
        self.agent_icon = None
        self._set_agent_icon(self.direction)
        self.num_arrows = 1
        self.world_size = world.world_size

    def get_has_gold(self):
        return self.has_gold

    def get_is_dead(self):
        return self.is_dead

    def shoot_arrow(self):
        """
        If agent has arrows, decrements arrow and returns True.
        If agent does not, returns False.
        """
        if self.num_arrows >= 1:
            self.num_arrows -= 1
            return True
        else:
            return False

    def get_name(self):
        return self.agent_function.get_agent_name()

    def get_agent_icon(self):
        return self.agent_icon

    def get_direction(self):
        return self.direction

    def get_location(self):
        return self.location

    def choose_action(self):
        return self.agent_function.process(self.percept)

    def go_forward(self):
        if self.non_deterministic_mode:
            move_dir = self._non_deterministic_move()
            # Move NORTH
            if self.direction == 'N':
                if move_dir == 'F':
                    if self.location[0]+1 < self.world_size:
                        self.location[0] += 1
                    else:
                        self.world.bump = True
                elif move_dir == 'L':
                    if self.location[1]-1 >= 0:
                        self.location[1] -= 1
                    else:
                        self.world.bump = True
                elif move_dir == 'R':
                    if self.location[1]+1 < self.world_size:
                        self.location[1] += 1
                    else:
                        self.world.bump = True
            # Move EAST
            if self.direction == 'E':
                if move_dir == 'F':
                    if self.location[1]+1 < self.world_size:
                        self.location[1] += 1
                    else:
                        self.world.bump = True
                elif move_dir == 'L':
                    if self.location[0]+1 < self.world_size:
                        self.location[0] += 1
                    else:
                        self.world.bump = True
                elif move_dir == 'R':
                    if self.location[0]-1 >= 0:
                        self.location[0] -= 1
                    else:
                        self.world.bump = True
            # Move SOUTH
            elif self.direction == 'S':
                if move_dir == 'F':
                    if self.location[0]-1 >= 0:
                        self.location[0] -= 1
                    else:
                        self.world.bump = True
                elif move_dir == 'L':
                    if self.location[1]+1 < self.world_size:
                        self.location[1] += 1
                    else:
                        self.world.bump = True
                elif move_dir == 'R':
                    if self.location[1]-1 >= 0:
                        self.location[1] -= 1
                    else:
                        self.world.bump = True
            # Move WEST
            elif self.direction == 'W':
                if move_dir == 'F':
                    if self.location[1]-1 >= 0:
                        self.location[1] -= 1
                    else:
                        self.world.bump = True
                if move_dir == 'L':
                    if self.location[0]-1 >= 0:
                        self.location[0] -= 1
                    else:
                        self.world.bump = True
                elif move_dir == 'R':
                    if self.location[0]+1 < self.world_size:
                        self.location[0] += 1
                    else:
                        self.world.bump = True
        # If deterministic mode
        else:
            if self.direction == 'N':
                if self.location[0]+1 < self.world_size:
                    self.location[0] += 1
                else:
                    self.world.bump = True
            elif self.direction == 'E':
                if self.location[1]+1 < self.world_size:
                    self.location[1] += 1
                else:
                    self.world.bump = True
            elif self.direction == 'S':
                if self.location[0]-1 >= 0:
                    self.location[0] -= 1
                else:
                    self.world.bump = True
            elif self.direction == 'W':
                if self.location[1]-1 >= 0:
                    self.location[1] -= 1
                else:
                    self.world.bump = True

    def turn_right(self):
        if self.direction == 'N':
            self._set_agent_icon('E')
        elif self.direction == 'E':
            self._set_agent_icon('S')
        elif self.direction == 'S':
            self._set_agent_icon('W')
        elif self.direction == 'W':
            self._set_agent_icon('N')

    def turn_left(self):
        if self.direction == 'N':
            self._set_agent_icon('W')
        elif self.direction == 'E':
            self._set_agent_icon('N')
        elif self.direction == 'S':
            self._set_agent_icon('E')
        elif self.direction == 'W':
            self._set_agent_icon('S')

    def shoot(self):
        if self.num_arrows == 1:
            self.num_arrows -= 1
            return True
        else:
            return False

    def _non_deterministic_move(self):
        random_move_idx = randint(0, 9)
        random_move_idx_map = {
            0: 'F', 1: 'F', 2: 'F', 3: 'F', 4: 'F', 5: 'F', 6: 'F', 7: 'F',
            8: 'L', 9: 'R',
        }
        random_move = random_move_idx_map[random_move_idx]
        return random_move

    def _set_agent_icon(self, direction):
        self.direction = direction
        if direction == 'N':
            self.agent_icon = 'A'
        elif direction == 'E':
            self.agent_icon = '>'
        elif direction == 'S':
            self.agent_icon = 'V'
        elif self.direction == 'W':
            self.agent_icon = '<'

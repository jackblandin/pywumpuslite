import numpy as np
from utils import out


class Environment:

    def __init__(self, world_size, input_world, out_writer):
        self.world_size = world_size
        self.world = np.copy(input_world)
        self.out_writer = out_writer
        self.percepts = np.full((world_size, world_size, 4), ' ')
        self.agent = None
        self.prev_agent_pos = self.get_agent_location()
        self.bump = None
        self.scream = None
        # create divider bar for display output
        self.bar = '-' * ((world_size * 6) - 1)

        self._set_percept_map()

    def get_agent_direction(self):
        for i in range(0, self.world_size):
            for j in range(0, self.world_size):
                agent_icon = self.world[i][j][3]
                if agent_icon == 'A':
                    return 'N'
                if agent_icon == '>':
                    return 'E'
                if agent_icon == 'V':
                    return 'S'
                if agent_icon == '<':
                    return 'W'

    def get_agent_location(self):
        agent_pos = np.empty(2, dtype=int)
        # TODO 01/25/2019 - can we speed this up with numpy?
        for i in range(0, self.world_size):
            for j in range(0, self.world_size):
                if self.world[i][j][3] != ' ':
                    agent_pos[0] = i
                    agent_pos[1] = j
        return agent_pos

    def place_agent(self, agent):
        # Set previous agent locaiton to blank space
        self.world[self.prev_agent_pos[0]][self.prev_agent_pos[1]][3] = ' '

        # Set current agent location to agent's icon
        self.agent = agent
        self.world[agent.get_location()[0]][agent.get_location()[1]][3] = (
            agent.get_agent_icon())

        # Set previous agent location indexes to current location
        self.prev_agent_pos[0] = agent.get_location()[0]
        self.prev_agent_pos[1] = agent.get_location()[1]

    def get_bump(self):
        return self.bump

    def get_breeze(self):
        x, y = self.agent.get_location()
        return self.percepts[x][y][0] == 'B'

    def get_stench(self):
        x, y = self.agent.get_location()
        return self.percepts[x][y][1] == 'S'

    def get_glitter(self):
        x, y = self.agent.get_location()
        return self.percepts[x][y][2] == 'G'

    def get_scream(self):
        return self.scream

    def grab_gold(self):
        x, y = self.agent.get_location()
        if self.percepts[x][y][2] != 'G':
            return False
        else:
            self.percepts[x][y][2] = ' '
            self.world[x][y][2] = ' '
            return True

    def check_death(self):
        x, y = self.agent.get_location()
        if self.world[x][y][0] == 'P':
            return True
        elif self.world[x][y][1] == 'W':
            return True
        return False

    def shoot_arrow(self):
        """If Wumpus in line-of-sight, removes Wumpus, resets Stench percepts
        Returns
        -------
        Bool
            whether or not the arrow hit the Wumpus
        """
        agent_dir = self.agent.get_direction()
        loc = self.agent.get_location()

        def reset_stench_percept(x, y):
            if x-1 >= 0:
                self.percepts[x-1][y][1] = ' '
            if x+1 < self.world_size:
                self.percepts[x+1][y][1] = ' '
            if y-1 >= 0:
                self.percepts[x][y-1][1] = ' '
            if y+1 < self.world_size:
                self.percepts[x][y+1][1] = ' '
        if agent_dir == 'N':
            for i in range(loc[0], self.world_size):
                if self.world[i][loc[1]][1] == 'W':
                    # remove Wumpus from square
                    self.world[i][loc[1]][1] = '*'
                    x = i
                    y = loc[1]
                    reset_stench_percept(x, y)
                    return True
        elif agent_dir == 'E':
            for i in range(loc[1], self.world_size):
                if self.world[loc[0]][i][1] == 'W':
                    # remove Wumpus from square
                    self.world[loc[0]][i][1] = '*'
                    x = loc[0]
                    y = i
                    reset_stench_percept(x, y)
                    return True
        elif agent_dir == 'S':
            for i in range(loc[0], -1, -1):
                if self.world[i][loc[1]][1] == 'W':
                    # remove Wumpus from square
                    self.world[i][loc[1]][1] = '*'
                    x = i
                    y = loc[1]
                    reset_stench_percept(x, y)
                    return True
        elif agent_dir == 'W':
            for i in range(loc[1], -1, -1):
                if self.world[loc[0]][i][1] == 'W':
                    # remove Wumpus from square
                    self.world[loc[0]][i][1] = '*'
                    x = loc[0]
                    y = i
                    reset_stench_percept(x, y)
                    return True
        return False

    def print_percepts(self):
        print(' ' + self.bar)
        for i in range(self.world_size-1, -1, -1):
            for j in range(0, 2):
                msg = ''
                for k in range(0, self.world_size):
                    if j == 0:
                        msg += '| {} {} '.format(self.percepts[i][k][0],
                                                self.percepts[i][k][1])
                    else:
                        msg += '| {} {} '.format(self.percepts[i][k][2],
                                                self.percepts[i][k][3])
                    if k == self.world_size-1:
                        msg += ('|')
                print(msg)
            print(' ' + self.bar)
        print()

    def print_env(self):
        """
           -----------------------
          | P W | P W | P W | P W |
          | G A | G A | G A | G A |
           -----------------------
          | P W | P W | P W | P W |
          | G A | G A | G A | G A |
           -----------------------
          | P W | P W | P W | P W |
          | G A | G A | G A | G A |
           ----------------------- 23
          | P W | P W | P W | P W | A A |
          | G A | G A | G A | G A | A A |
           ----------------------------- 29

         P,W,G,A
        """
        msg = '\n ' + self.bar
        out(self.out_writer, msg)
        # for each row
        for i in range(self.world_size-1, -1, -1):
            # for each sub-row
            for j in range(0, 2):
                # for each
                msg = ''
                for k in range(0, self.world_size):
                    if j == 0:
                        msg += '| {} {} '.format(self.world[i][k][0],
                                                 self.world[i][k][1])
                    else:
                        msg += '| {} {} '.format(self.world[i][k][2],
                                                 self.world[i][k][3])
                    if (k == self.world_size-1):
                        msg += '|'
                out(self.out_writer, msg + ' ')
            out(self.out_writer, ' ' + self.bar)
        out(self.out_writer, '\n', False)

    def _set_percept_map(self):
        """Sets percepts of each square based on all possible object locations"
        World: Pit, Wumpus, Gold, Agent
        Percepts: Breeze, Stench, Glitter, Scream
        """
        for i in range(0, self.world_size):
            for j in range(0, self.world_size):
                for k in range(0, 4):
                    if self.world[i][j][k] == 'P':
                        if j-1 >= 0:
                            self.percepts[i][j-1][k] = 'B'
                        if i+1 < self.world_size:
                            self.percepts[i+1][j][k] = 'B'
                        if j+1 < self.world_size:
                            self.percepts[i][j+1][k] = 'B'
                        if i-1 >= 0:
                            self.percepts[i-1][j][k] = 'B'
                    elif self.world[i][j][k] == 'W':
                        if j-1 >= 0:
                            self.percepts[i][j-1][k] = 'S'
                        if i+1 < self.world_size:
                            self.percepts[i+1][j][k] = 'S'
                        if j+1 < self.world_size:
                            self.percepts[i][j+1][k] = 'S'
                        if i-1 >= 0:
                            self.percepts[i-1][j][k] = 'S'
                    elif self.world[i][j][k] == 'G':
                        self.percepts[i][j][k] = 'G'
        # self.print_percepts()

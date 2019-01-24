import sys
import numpy as np
from random import randint

def _generate_random_wumpus_word(size, random_agent_loc):
    new_world = np.full((size, size, 4), ' ')
    occupied = np.full((size, size), False)

    pits = 2
    wumpii = 1

    # default agent location and orientation
    agent_xloc = 0
    agent_yloc = 0
    agent_orient = '>'

    # randomly generate agent location and orientation
    if random_agent_loc:
        agent_xloc = randint(0, size-1)
        agent_yloc = randint(0, size-1)

        random_dir = randint(0, 3)
        agent_orient_map = { 0: 'A', 1: '>', 2: 'V', 3: '<' }
        agent_orient = agent_orient_map[random_dir]

    # place agent in the new world
    new_world[agent_xloc][agent_yloc] = agent_orient

    # pit and wumpus generation
    for idx, (n, lbl) in enumerate(zip([pits, wumpii], ['P', 'W'])):
        for i in range(n):
            x = randint(0, size-1)
            y = randint(0, size-1)

            while (x == agent_xloc and y == agent_yloc) or (occupied[x][y] == True):
                x = randint(0, size-1)
                y = randint(0, size-1)

            occupied[x][y] = True

            new_world[x][y][idx] = lbl

    x = randint(0, size-1)
    y = randint(0, size-1)

    occupied[x][y] = True

    new_world[x][y][2] = 'G'

    return new_world


if __name__ == '__main__':
    world_size = 4
    num_trials = 50
    max_steps = 50

    non_deterministic_mode = True
    random_agent_loc = False
    user_defined_seed = False

    out_filename = 'wumpus_out.txt'
    f = open(out_filename, 'w')

    seed = randint(-1000000, 1000000)

    for i, arg in enumerate(sys.argv[1:]):
        if len(sys.argv) > i + 2:
            next_arg = sys.argv[i+2] # skip two since starting from 1

        if arg == '-d' and int(next_arg) > 1:
            world_size = int(next_arg)

        elif arg == '-s':
            max_steps = int(next_arg)

        elif arg == '-t':
            num_trials = int(next_arg)

        elif arg == '-a':
            random_agent_loc = bool(int(next_arg))

        elif arg == '-r':
            seed = int(next_arg)
            user_defined_seed = True

        elif arg == '-f':
            out_filename = next_arg

        elif arg == '-n':
            non_deterministic_mode = bool(int(next_arg))

    outs = [
        'Dimensions: {} x {}'.format(world_size, world_size),
        'Maximum number of steps: {}'.format(max_steps),
        'Number of trials: {}'.format(num_trials),
        'Random agent location: {}'.format(random_agent_loc),
        'Random number seed: {}'.format(seed),
        'Output filename: {}'.format(out_filename),
        'Non-Deterministic Behavior: {}'.format(non_deterministic_mode),
    ]

    for out in outs:
        print(out)
        f.write(out + '\n')

    # wumpus world is a <size x size x 4> matrix ( xloc, yloc, <Pit, Wumpus, Gold, agent-orientation> )
    wumpus_world = _generate_random_wumpus_word(world_size, random_agent_loc)
    wumpus_env = Environemnt(world_size, wumpus_world, f)

    trial_scores = np.empty(num_trials)

    for i in range(trial_scores):
        trial = Simulation(wumpus_env, max_steps, f, non_deterministic_mode)
        trial_scores[i] = trial.get_score()

        out = '-' * 40
        print(out); f.write(out + '\n')

        wumpus_world = _generate_random_wumpus_word(world_size, random_agent_loc)

        wumpus_env = Environemnt(world_size, wumpus_world, f)

    for i in range(trial_scores):
        out = 'Trial {} score: {}'.format(i, trial_scores[i])
        print(out); f.write(out + '\n')

    total_score = np.sum(trial_scores)
    out = '\nTotal Score: {}'.format(total_score)
    print(out); f.write(out + '\n')

    avg_score = np.mean(trial_scores)
    out = '\nAverage Score: {}'.format(avg_score)
    print(out); f.write(out + '\n')

    f.close()
    print('Finished')

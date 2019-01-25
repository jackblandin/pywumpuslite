import sys
import numpy as np
from random import randint
from utils import out
from environment import Environment
from simulation import Simulation


WORLD_SIZE = 4
NUM_TRIALS = 100
MAX_STEPS = 50
RANDOM_AGENT_LOC = False
NON_DETERMINISTIC_MODE = True
RANDOM_WORLD = True
SEED = randint(-1000000, 1000000)
OUT_FILENAME = 'wumpus_output.txt'


def run_application(world_size=WORLD_SIZE, max_steps=MAX_STEPS,
                    num_trials=NUM_TRIALS, random_agent_loc=RANDOM_AGENT_LOC,
                    seed=SEED, non_deterministic_mode=NON_DETERMINISTIC_MODE,
                    random_world=RANDOM_WORLD, out_filename=OUT_FILENAME,
                    out_writer=None):

    if out_writer is None:
        out_writer = open(out_filename, 'w')

    outs = [
        'Dimensions: {} x {}'.format(world_size, world_size),
        'Maximum number of steps: {}'.format(max_steps),
        'Number of trials: {}'.format(num_trials),
        'Random agent location: {}'.format(random_agent_loc),
        'Random number seed: {}'.format(seed),
        'Output filename: {}'.format(out_filename),
        'Non-Deterministic Behavior: {}'.format(non_deterministic_mode),
    ]

    for msg in outs:
        out(out_writer, msg)

    # wumpus world is a < size x size x 4 > matrix
    # < xloc, yloc, <Pit, Wumpus, Gold, agent-orientation> >
    if random_world:
        wumpus_world = _generate_random_wumpus_world(out_writer, world_size,
                                                 random_agent_loc)
    else:
        wumpus_world = _generate_static_wumpus_world(world_size)

    wumpus_env = Environment(world_size, wumpus_world, out_writer)

    trial_scores = np.empty(num_trials)

    # use as return value for debugging
    last_wumpus_env = None

    for i in range(len(trial_scores)):
        trial = Simulation(wumpus_env, max_steps, out_writer,
                           non_deterministic_mode)
        trial.run()
        trial_scores[i] = trial.get_score()

        out(out_writer, 'Trial number: {}'.format(i+1))
        out(out_writer, '-' * 40)

        last_wumpus_env = wumpus_env

        if random_world:
            wumpus_world = _generate_random_wumpus_world(out_writer,
                                                         world_size,
                                                         random_agent_loc)
        else:
            wumpus_world = _generate_static_wumpus_world(world_size)

        wumpus_env = Environment(world_size, wumpus_world, out_writer)

    for i in range(len(trial_scores)):
        msg = 'Trial {} score: {}'.format(i, trial_scores[i])
        out(out_writer, msg)

    total_score = np.sum(trial_scores)
    msg = '\nTotal Score: {}'.format(total_score)
    out(out_writer, msg)

    avg_score = np.mean(trial_scores)
    msg = '\nAverage Score: {}'.format(avg_score)
    out(out_writer, msg)

    out_writer.close()

    print('Finished')

    return last_wumpus_env


def cli_run_application():
    world_size = WORLD_SIZE
    num_trials = NUM_TRIALS
    max_steps = MAX_STEPS
    non_deterministic_mode = NON_DETERMINISTIC_MODE
    random_agent_loc = RANDOM_AGENT_LOC
    random_world = RANDOM_WORLD
    seed = SEED
    out_filename = OUT_FILENAME

    for i, arg in enumerate(sys.argv[1:]):
        if len(sys.argv) > i + 2:
            next_arg = sys.argv[i+2]  # skip two since starting from 1

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

        elif arg == '-f':
            out_filename = next_arg

        elif arg == '-n':
            non_deterministic_mode = bool(int(next_arg))

        elif arg == '-w':
            random_world = bool(int(next_arg))

    return run_application(world_size=world_size,
                           max_steps=max_steps,
                           num_trials=num_trials,
                           random_agent_loc=random_agent_loc,
                           seed=seed,
                           non_deterministic_mode=non_deterministic_mode,
                           random_world=random_world,
                           out_filename=out_filename)


def _generate_random_wumpus_world(out_writer, size, random_agent_loc):

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
        agent_orient_map = {0: 'A', 1: '>', 2: 'V', 3: '<'}
        agent_orient = agent_orient_map[random_dir]

    # place agent in the new world
    new_world[agent_xloc][agent_yloc][3] = agent_orient

    # pit generation
    pit_occupied = np.full((size, size), False)  # no overlapping pits
    for i in range(pits):
        x = randint(0, size-1)
        y = randint(0, size-1)
        while (x == agent_xloc and y == agent_yloc) or pit_occupied[x][y]:
            x = randint(0, size-1)
            y = randint(0, size-1)
        new_world[x][y][0] = 'P'
        pit_occupied[x][y] = True
        occupied[x][y] = True

    # wumpus generation
    wumpus_occupied = np.full((size, size), False)  # no overlapping wumpii
    for i in range(wumpii):
        x = randint(0, size-1)
        y = randint(0, size-1)
        while (x == agent_xloc and y == agent_yloc) or wumpus_occupied[x][y]:
            x = randint(0, size-1)
            y = randint(0, size-1)
        new_world[x][y][1] = 'W'
        wumpus_occupied[x][y] = True
        occupied[x][y] = True

    # Gold generation
    x = randint(0, size-1)
    y = randint(0, size-1)
    new_world[x][y][2] = 'G'
    occupied[x][y] = True

    return new_world

def _generate_static_wumpus_world(size):
    new_world = np.full((size, size, 4), ' ')
    new_world[0][0][3] = '>'
    new_world[3][0][0] = 'P'
    new_world[2][2][0] = 'P'
    new_world[0][3][1] = 'W'
    new_world[3][3][2] = 'G'
    return new_world


if __name__ == '__main__':
    cli_run_application()

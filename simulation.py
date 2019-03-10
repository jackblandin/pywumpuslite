import action as Action
from agent import Agent
from transfer_percept import TransferPercept
from utils import out


class Simulation:

    ACTION_COST = -1
    DEATH_COST = -1000
    SHOOT_COST = -10
    GOLD_REWARD = 1000

    def __init__(self, env, max_steps, out_writer,
                 non_deterministic_mode):
        self.env = env
        self.max_steps = max_steps
        self.out_writer = out_writer
        self.non_deterministic_mode = non_deterministic_mode
        self.current_score = None
        self.simulation_running = False
        self.last_action = None
        self.did_agent_find_gold = False
        self.was_agent_killed = False
        self.was_agent_stuck = False

    def run(self):
        self.current_score = 0
        self.simulation_running = True
        self.last_action = 0

        step_counter = 0

        transfer_percept = TransferPercept(self.env)
        agent = Agent(self.env, transfer_percept, self.non_deterministic_mode)

        env = self.env
        env.place_agent(agent)
        env.print_env()
        self._print_current_percept_sequence(transfer_percept)

        msg = 'Current score: {}'.format(self.current_score)
        out(self.out_writer, msg)

        while self.simulation_running and step_counter < self.max_steps:
            msg = 'Last action: {}'.format(
                Action.print_action(self.last_action))
            out(self.out_writer, msg)

            msg = 'Time step: {}'.format(step_counter)
            out(self.out_writer, msg)

            self._handle_action(agent, agent.choose_action(), env)
            env.place_agent(agent)

            env.print_env()
            self._print_current_percept_sequence(transfer_percept)

            msg = 'Current score: {}'.format(self.current_score)
            out(self.out_writer, msg)

            step_counter += 1

            last_action_before_end = self.last_action

            if (step_counter == self.max_steps or not self.simulation_running):
                msg = 'Last action: {}'.format(
                    Action.print_action(self.last_action))
                out(self.out_writer, msg)

                msg = 'Time step: {}'.format(step_counter)
                out(self.out_writer, msg)

                self.last_action = Action.END_TRIAL

            if agent.get_has_gold():
                self.did_agent_find_gold = True
                msg = '\n{} found the GOLD!'.format(agent.get_name())
                out(self.out_writer, msg)

            if agent.get_is_dead():
                self.was_agent_killed = True
                msg = '\n{} is DEAD!'.format(agent.get_name())
                out(self.out_writer, msg)

            if last_action_before_end == Action.NO_OP:
                self.was_agent_stuck = True

        self._print_end_world(env)

    def get_score(self):
        return self.current_score

    def get_did_agent_find_gold(self):
        return self.did_agent_find_gold

    def get_was_agent_killed(self):
        return self.was_agent_killed

    def get_was_agent_stuck(self):
        return self.was_agent_stuck

    def _print_end_world(self, env):
        env.print_env()

        msg = 'Final score: {}'.format(self.current_score)
        out(self.out_writer, msg)

        msg = 'Last action: {}'.format(self.last_action)
        out(self.out_writer, msg)

    def _print_current_percept_sequence(self, transfer_percept):
        out(self.out_writer, str(transfer_percept))

    def _handle_action(self, agent, action, env):

        def handle_go_forward():
            if env.get_bump():
                env.bump = False
            agent.go_forward()
            env.place_agent(agent)
            if env.check_death():
                self.current_score += self.DEATH_COST
                self.simulation_running = False
                agent.is_dead = True
            else:
                self.current_score += self.ACTION_COST
            if env.get_scream():
                env.scream = False

        def handle_turn_right():
            self.current_score += self.ACTION_COST
            agent.turn_right()
            env.place_agent(agent)
            if env.get_bump():
                env.bump = False
            if env.get_scream():
                env.scream = False

        def handle_turn_left():
            self.current_score += self.ACTION_COST
            agent.turn_left()
            env.place_agent(agent)
            if env.get_bump():
                env.bump = False
            if env.get_scream():
                env.scream = False

        def handle_grab():
            if env.grab_gold():
                self.current_score += self.GOLD_REWARD
                self.simulation_running = False
                agent.has_gold = True
            else:
                self.current_score += self.ACTION_COST
            env.place_agent(agent)
            if env.get_bump():
                env.bump = False
            if env.get_scream():
                env.scream = False

        def handle_shoot():
            if agent.shoot_arrow():
                if env.shoot_arrow():
                    env.scream = True
                self.current_score += self.SHOOT_COST
            else:
                if env.get_scream():
                    env.scream = False
                self.current_score += self.ACTION_COST
            env.place_agent(agent)
            if env.get_bump():
                env.bump = False

        def handle_no_opp():
            env.place_agent(agent)
            if env.get_bump():
                env.bump = False
            if env.get_scream():
                env.scream = False

        self.last_action = action

        if action == Action.GO_FORWARD:
            return handle_go_forward()
        if action == Action.TURN_RIGHT:
            return handle_turn_right()
        if action == Action.TURN_LEFT:
            return handle_turn_left()
        if action == Action.GRAB:
            return handle_grab()
        if action == Action.SHOOT:
            return handle_shoot()
        if action == Action.NO_OP:
            return handle_no_opp()

        raise ValueError('Invalid Action: {}'.format(action))

import action as Action

class AgentFunction:

    def __init__(self):
        self.agent_name = 'Agent Smith'

    def process(self, percept):
        bump = percept.get_bump()
        glitter = percept.get_glitter()
        breeze = percept.get_breeze()
        stench = percept.get_stench()
        scream = percept.get_scream()

        if glitter:
            return Action.GRAB

        if scream and not breeze:
            return Action.GO_FORWARD

        if not stench and not breeze:
            return Action.GO_FORWARD

        if stench:
            return Action.SHOOT

        return Action.NO_OP

    def get_agent_name(self):
        return self.agent_name

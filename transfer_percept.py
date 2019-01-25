class TransferPercept:

    def __init__(self, env):
        self.env = env

    def get_bump(self):
        return self.env.get_bump()

    def get_glitter(self):
        return self.env.get_glitter()

    def get_breeze(self):
        return self.env.get_breeze()

    def get_stench(self):
        return self.env.get_stench()

    def get_scream(self):
        return self.env.get_scream()

    def __repr__(self):
        bump_msg, glitter_msg, breeze_msg, stench_msg, scream_msg = (
            'none', 'none', 'none', 'none', 'none')

        if self.get_bump():
            bump_msg = 'Bump'

        if self.get_glitter():
            glitter_msg = 'Glitter'

        if self.get_breeze():
            breeze_msg = 'Breeze'

        if self.get_stench():
            stench_msg = 'Stench'

        if self.get_scream():
            scream_msg = 'Scream'

        msg = 'Percept: < {}, {}, {}, {}, {} >\n'.format(bump_msg, glitter_msg,
                                                     breeze_msg, stench_msg,
                                                     scream_msg)
        return msg


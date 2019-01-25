START_TRIAL = 0
GO_FORWARD = 1
TURN_RIGHT = 2
TURN_LEFT = 3
GRAB = 4
SHOOT = 5
NO_OP = 6
END_TRIAL = 7

def print_action(action):
    if action == START_TRIAL:
        return 'START_TRIAL'
    if action == GO_FORWARD:
        return 'GO_FORWARD'
    if action == TURN_RIGHT:
        return 'TURN_RIGHT'
    if action == TURN_LEFT:
        return 'TURN_LEFT'
    if action == GRAB:
        return 'GRAB'
    if action == SHOOT:
        return 'SHOOT'
    if action == NO_OP:
        return 'NO_OP'
    if action == END_TRIAL:
        return 'END_TRIAL'
    raise ValueError('Invalid action: {}'.format(action))

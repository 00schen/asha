import pybullet as p
import numpy as np

class KeyboardPolicy:
    def __init__(self):
        self.action = np.zeros(6)

    def get_action(self, obs):
        keys = p.getKeyboardEvents()
        inputs = [
            p.B3G_RIGHT_ARROW,
            p.B3G_LEFT_ARROW,
            ord('r'),
            ord('f'),
            p.B3G_DOWN_ARROW,
            p.B3G_UP_ARROW,
        ]
        noop = True
        for key in inputs:
            if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
                self.action = np.zeros(6)
                self.action[inputs.index(key)] = 1
                noop = False
        return self.action, {}

    def reset(self):
        self.action = np.zeros(6)

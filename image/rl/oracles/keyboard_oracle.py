import pybullet as p
from .base_oracles import UserInputOracle


class KeyboardOracle(UserInputOracle):
    def _query(self):
        keys = p.getKeyboardEvents()
        inputs = {
            p.B3G_LEFT_ARROW: 'left',
            p.B3G_RIGHT_ARROW: 'right',
            ord('r'): 'forward',
            ord('f'): 'backward',
            p.B3G_UP_ARROW: 'up',
            p.B3G_DOWN_ARROW: 'down'
        }
        self.action = 'noop'
        for key in inputs:
            if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
                self.action = inputs[key]

        return {"action": self.action}

import numpy as np


class Oracle:
    def __init__(self):
        self.size = 6
        self.status = OracleStatus()

    def _query(self, obs=None, info=None):
        pass

    def reset(self):
        pass


class OracleStatus:
    def __init__(self):
        self.action = None
        self.curr_intervention = False
        self.new_intervention = False


class UserModelOracle(Oracle):
    def __init__(self, rng, threshold=.5, epsilon=0, blank=1):
        super().__init__()
        self.rng = rng
        self.epsilon = epsilon
        self.blank = blank
        self.threshold = threshold

    def get_action(self, obs, info):
        criterion, target_pos = self._query(obs, info)
        action = np.zeros(self.size)
        if self.rng.random() < self.blank * criterion:
            traj = target_pos - info['tool_pos']
            axis = np.argmax(np.abs(traj))
            index = 2 * axis + (traj[axis] > 0)
            if self.rng.random() < self.epsilon:
                index = self.rng.integers(self.size)
            action[index] = 1
        return action, {}


class UserInputOracle(Oracle):
    def __init__(self):
        super().__init__()

    def get_action(self, obs, info=None):
        user_info = self._query()
        action = {
            'left': np.array([0, 1, 0, 0, 0, 0]),
            'right': np.array([1, 0, 0, 0, 0, 0]),
            'forward': np.array([0, 0, 1, 0, 0, 0]),
            'backward': np.array([0, 0, 0, 1, 0, 0]),
            'up': np.array([0, 0, 0, 0, 0, 1]),
            'down': np.array([0, 0, 0, 0, 1, 0]),
            'noop': np.array([0, 0, 0, 0, 0, 0])
        }[self.action]
        return action, user_info

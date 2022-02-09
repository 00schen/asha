import numpy as np


class DemonstrationPolicy:
    def __init__(self, policy, env, p):
        self.polcies = [policy, RandomPolicy(env.rng, epsilon=1 / 10)]
        self.p = p
        self.rng = env.rng

    def get_action(self, obs):
        p = [self.p, 1 - self.p]
        actions = [policy.get_action(obs) for policy in self.polcies]
        act_tuple = self.rng.choice(actions, p=p)
        return act_tuple

    def reset(self):
        for policy in self.polcies:
            policy.reset()


class RandomPolicy:
    def __init__(self, rng, epsilon=.25):
        self.epsilon = epsilon
        self.rng = rng

    def get_action(self, obs):
        rng = self.rng
        self.action_index = self.action_index if self.rng.random() > self.epsilon else self.rng.choice(6)
        action = np.zeros(6)
        action[self.action_index] = 1
        return action, {}

    def reset(self):
        self.action_index = 0

from rlkit.data_management.replay_buffer import ReplayBuffer
import numpy as np


class BalancedReplayBuffer(ReplayBuffer):
    def __init__(self, main_buffer, prior_buffer):
        self.main_buffer = main_buffer
        self.prior_buffer = prior_buffer

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self.main_buffer.add_sample(observation, action, reward, terminal, next_observation, **kwargs)

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self, **kwargs):
        return self.main_buffer.num_steps_can_sample() + self.prior_buffer.num_steps_can_sample()

    def add_path(self, path):
        self.main_buffer.add_path(path)
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        main_batch = self.main_buffer.random_batch(batch_size // 2)
        prior_batch = self.prior_buffer.random_batch(batch_size // 2)
        combined_batch = {key: np.concatenate((main_batch[key], prior_batch[key]), axis=0) for key in main_batch.keys()}
        return combined_batch

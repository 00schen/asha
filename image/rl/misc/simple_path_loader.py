import numpy as np
from rlkit.util.io import load_local_or_remote_file


class SimplePathLoader:
    def __init__(self, demo_path, demo_path_proportion, replay_buffer):
        self.demo_path = demo_path
        self.demo_path_proportion = demo_path_proportion
        self.replay_buffer = replay_buffer

    def load_demos(self):
        if type(self.demo_path) is not list:
            self.demo_path = [self.demo_path]
        for demo_path, proportion in zip(self.demo_path, self.demo_path_proportion):
            data = load_local_or_remote_file(demo_path)
            print("using", len(data), "paths for training")
            for i, path in enumerate(data[:proportion]):
                self.load_path(path, self.replay_buffer)

    def load_path(self, path, replay_buffer, obs_dict=None):
        tran_iter = zip(list(path['observations'][1:]) + [path['next_observations'][-1]],
                        list(path['actions']),
                        list(path['rewards']),
                        list(path['terminals']),
                        list(path['env_infos']),
                        list(path['agent_infos'])
                        )

        env = replay_buffer.env
        processed_trans = []
        obs = path['observations'][0]
        obs = env.adapt_reset(obs, path['env_infos'][0])

        for next_obs, action, r, done, info, ainfo in tran_iter:
            info.update(ainfo)
            action = ainfo.get(env.action_type, action)
            try:
                action = info[env.action_type]
            except:
                breakpoint()
            next_obs, r, done, info = env.adapt_step(next_obs, r, done, info)

            processed_trans.append((obs, next_obs, action, r, done, info))
            obs = next_obs

        new_path = dict(zip(
            ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'env_infos'],
            list(zip(*processed_trans))
        ))
        path.update(new_path)

        path['next_observations'] = np.array(path['next_observations'])
        path['actions'] = np.array(path['actions'])
        path['rewards'] = np.array(path['rewards'])[:, np.newaxis]
        path['terminals'] = np.array(path['terminals'])[:, np.newaxis]
        replay_buffer.add_path(path)

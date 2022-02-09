from rlkit.samplers.data_collector import MdpPathCollector
import pybullet as p
from rlkit.samplers.rollout_functions import rollout
from rl.misc.env_wrapper import real_gaze
import time


def _wait_for_key(env, agent, o, key=p.B3G_SPACE, update_obs_class=real_gaze):
    while True:
        keys = p.getKeyboardEvents()
        if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
            break

    # for some reason needed for obs to be updated
    time.sleep(0.1)

    for adapt in env.adapts:
        if isinstance(adapt, update_obs_class):
            adapt.update_obs(o)


class FullPathCollector(MdpPathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            real_user=False
    ):
        super().__init__(env,
                         policy,
                         max_num_epoch_paths_saved,
                         render, render_kwargs,
                         rollout_fn,
                         save_env_in_snapshot)
        self.reset_callback = _wait_for_key if real_user else None
        self.reset_callback = None

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths=False,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length,
                render=self._render,
                render_kwargs=self._render_kwargs,
                reset_callback=self.reset_callback
            )
            path_len = len(path['actions'])
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_snapshot(self):
        return dict()

import abc

import gtimer as gt
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
import pybullet as p
from rlkit.core import logger
import numpy as np
import time
from rl.misc.env_wrapper import sim_target


class BatchRLAlgorithm(TorchBatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            calibration_data_collector,
            calibration_indices,
            trajs_per_index,
            calibration_buffer: ReplayBuffer,
            pretrain_steps=100,
            max_failures=10,
            calibrate_split=True,
            real_user=True,
            relabel_failures=True,
            seedid=0,
            curriculum=False,
            goal_baseline=False,
            goal_noise_std=0,
            latent_calibrate=False,
            calibrate_input=None,
            **kwargs,
    ):
        super().__init__(
            num_expl_steps_per_train_loop=1, *args, **kwargs
        )
        self.calibration_data_collector = calibration_data_collector
        if calibration_indices is None:
            calibration_indices = self.expl_env.base_env.target_indices
        self.calibration_indices = calibration_indices
        self.trajs_per_index = trajs_per_index
        self.calibration_buffer = calibration_buffer
        self.pretrain_steps = pretrain_steps
        self.max_failures = max_failures
        self.calibrate_split = calibrate_split
        self.real_user = real_user
        self.relabel_failures = relabel_failures
        self.seedid = seedid
        self.curriculum = curriculum
        self.goal_baseline = goal_baseline
        self.goal_noise_std = goal_noise_std
        self.latent_calibrate = latent_calibrate
        self.calibrate_input = calibrate_input
        self.blocks = []

        self.metrics = {'success_episodes': [],
                        'episode_lengths': [],
                        'success_blocks': [],
                        'block_lengths': []
                        }

        if self.real_user:
            self.metrics['correct_rewards'] = []
            self.metrics['correct_blocks'] = []
            self.metrics['user_feedback'] = []

        if self.expl_env.env_name == 'Valve':
            self.metrics['final_angle_error'] = []
            self.metrics['init_angle_error'] = []

    def _sample_and_train(self, steps, buffers):
        self.training_mode(True)
        for _ in range(steps):
            for _ in range(len(self.trainer.vaes)):
                batches = [buffer.random_batch(self.batch_size // len(buffers)) for buffer in buffers]
                train_data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0].keys()}
                self.trainer.train(train_data)
        gt.stamp('training', unique=False)
        self.training_mode(False)

    def _calibrate_goal_baseline(self):
        if self.real_user:
            pass
        else:
            calibration_points = np.array([(-1, -1), (-1, 0), (-1, 1),
                                           (0, -1), (0, 0), (0, 1),
                                           (1, -1), (1, 0), (1, -1)])
            calibration_points = np.tile(calibration_points, (self.trajs_per_index, 1))

            data = calibration_points + np.random.normal(0, self.goal_noise_std, calibration_points.shape)
            self.expl_data_collector._policy.x_svr_estimator.fit(data, y=calibration_points[:, 0])
            self.expl_data_collector._policy.y_svr_estimator.fit(data, y=calibration_points[:, 1])

    def _calibrate(self):
        old_feature = None
        sim_target_adapt = None
        if self.calibrate_input is not None:
            for adapt in self.expl_env.adapts:
                if isinstance(adapt, sim_target):
                    sim_target_adapt = adapt
                    old_feature = adapt.feature
                    adapt.feature = self.calibrate_input
                    break

        self.expl_env.seed(self.seedid)
        self.expl_env.base_env.calibrate_mode(True, self.calibrate_split)
        calibration_data = []

        # the buffer actually used for calibration
        calibration_buffer = self.calibration_buffer if self.calibration_buffer is not None else self.replay_buffer

        for _ in range(self.trajs_per_index):
            for index in self.calibration_indices:
                self.expl_env.new_goal(index)
                calibration_paths = self.calibration_data_collector.collect_new_paths(
                    self.max_path_length,
                    1,
                    discard_incomplete_paths=False,
                )
                self.calibration_data_collector.end_epoch(-1)

                calibration_buffer.add_paths(calibration_paths)
                calibration_data.extend(calibration_paths)

        logger.save_extra_data(calibration_data, 'calibration_data.pkl', mode='pickle')

        gt.stamp('pretrain exploring', unique=False)

        old_objective = None
        if self.latent_calibrate:
            old_objective = self.trainer.objective
            self.trainer.objective = 'latent'
        self._sample_and_train(self.pretrain_steps, [calibration_buffer])

        if self.latent_calibrate:
            self.trainer.objective = old_objective
            self.trainer.second_half_latent = True

        if old_feature is not None and sim_target_adapt is not None:
            sim_target_adapt.feature = old_feature

    def _train(self):
        if self.goal_baseline:
            self._calibrate_goal_baseline()

        else:
            self._calibrate()

        self.expl_env.seed(self.seedid + 100)
        self.eval_env.seed(self.seedid + 200)

        self.expl_env.base_env.calibrate_mode(False, False)
        self.eval_env.base_env.calibrate_mode(False, False)

        failed_paths = []
        successful_paths = []
        self.expl_env.new_goal()

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.eval_paths:
                self.eval_data_collector.collect_new_paths(
                    self.eval_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                gt.stamp('evaluation sampling')

            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            assert len(new_expl_paths) == 1
            path = new_expl_paths[0]

            real_success = path['env_infos'][-1]['task_success']
            timeout = len(path['observations']) == self.max_path_length and not real_success

            gt.stamp('exploration sampling', unique=False)
            if self.real_user:
                success = None
                # automate reward if timeout and not valve env
                if timeout and not self.expl_env.env_name == 'Valve':
                    time.sleep(1)
                    success = real_success
                    self.metrics['correct_rewards'].append(None)

                elif self.expl_env.env_name == 'Valve':
                    success = path['env_infos'][-1]['feedback']

                # valve success feedback is True if user terminated as a success, -1 otherwise
                if not isinstance(success, bool):
                    while True:
                        keys = p.getKeyboardEvents()

                        if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED:
                            success = True

                            # relabel with wrong goals that were reached if not actual success.
                            # currently specific only to light switch and bottle, handled by valve differently
                            if not real_success:
                                if 'current_string' in path['env_infos'][-1]:
                                    wrong_reached_index = np.where(path['env_infos'][-1]['current_string'] == 0)[0][0]
                                    wrong_reached_goal = path['env_infos'][0]['switch_pos'][wrong_reached_index]

                                # assumes only 2 targets in bottle, and version of bottle with only 1 goal
                                elif 'unique_targets' in path['env_infos'][-1]:
                                    wrong_reached_index = 1 - path['env_infos'][-1]['unique_index']
                                    wrong_reached_goal = path['env_infos'][0]['unique_targets'][wrong_reached_index]

                                elif self.expl_env.env_name == 'Valve':
                                    wrong_reached_goal = None

                                else:
                                    raise NotImplementedError()
                                    
                            break
                        elif p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_WAS_TRIGGERED:
                            success = False
                            break

                    self.metrics['correct_rewards'].append(success == real_success)
                    self.metrics['user_feedback'].append(success)

            else:
                success = real_success

            self.metrics['success_episodes'].append(real_success)
            self.metrics['episode_lengths'].append(len(path['observations']))

            if self.expl_env.env_name == 'Valve':
                self.metrics['final_angle_error'].append(np.abs(path['env_infos'][-1]['angle_error']))
                self.metrics['init_angle_error'].append(np.abs(path['env_infos'][0]['angle_error']))

            if success:
                successful_paths.append(path)
            else:
                failed_paths.append(path)

            # no actual relabeling right now, since goals for all paths should be same
            # only add to paths to buffer if successful
            if success:
                paths_to_add = successful_paths + failed_paths if self.relabel_failures else successful_paths

                # have to relabel goals in valve with angle actually reached
                if self.expl_env.env_name == 'Valve':
                    new_target_angle = successful_paths[-1]['env_infos'][-1]['valve_angle']
                    new_goal = np.array([np.sin(new_target_angle), np.cos(new_target_angle)])
                    for path in paths_to_add:
                        for i in range(len(path['observations'])):
                            path['observations'][i]['goal'] = new_goal.copy()
                            path['next_observations'][i]['goal'] = new_goal.copy()

                self.replay_buffer.add_paths(paths_to_add)
                gt.stamp('data storing', unique=False)

                buffers = [self.replay_buffer]
                if self.calibration_buffer is not None:
                    buffers.append(self.calibration_buffer)
                self._sample_and_train(self.num_trains_per_train_loop, buffers)

            block_timeout = len(failed_paths) >= self.max_failures
            if success or block_timeout or epoch == self.num_epochs - 1:
                self.metrics['success_blocks'].append(real_success)
                self.metrics['block_lengths'].append(len(failed_paths) + len(successful_paths))
                if self.real_user:
                    self.metrics['correct_blocks'].append(block_timeout or real_success)
                self.blocks.append(failed_paths + successful_paths)

                failed_paths = []
                successful_paths = []
                self.expl_env.new_goal()  # switch positions do not change if not block end

                if self.curriculum and hasattr(self.expl_env.base_env, 'update_curriculum'):
                    self.expl_env.base_env.update_curriculum(success)

            self._end_epoch(epoch)

            logger.save_extra_data(self.metrics, 'metrics.pkl', mode='pickle')
            logger.save_extra_data(self.blocks, 'data.pkl', mode='pickle')

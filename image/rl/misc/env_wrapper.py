from functools import reduce
import os
from pathlib import Path
import h5py
from collections import deque

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm

import pybullet as p
import assistive_gym as ag
from gym import spaces, Env

import cv2
import torch
from gaze_capture.face_processor import FaceProcessor
from gaze_capture.ITrackerModel import ITrackerModel
import threading
from rl.oracles import *

main_dir = str(Path(__file__).resolve().parents[2])


def default_overhead(config):
    factory_map = {
        'session': session_factory,
    }
    factories = [factory_map[factory] for factory in config['factories']]
    factories = [action_factory] + factories
    wrapper = reduce(lambda value, func: func(value), factories, LibraryWrapper)

    class Overhead(wrapper):
        def __init__(self, config):
            super().__init__(config)
            self.rng = default_rng(config['seedid'])
            adapt_map = {
                'oracle': oracle,
                'static_gaze': static_gaze,
                'real_gaze': real_gaze,
                'joint': joint,
                'sim_keyboard': sim_keyboard,
                'keyboard': keyboard,
                'goal': goal,
                'reward': reward,
                'sim_target': sim_target,
                'dict_to_array': dict_to_array,
            }
            self.adapts = [adapt_map[adapt] for adapt in config['adapts']]
            self.adapts = [adapt(self, config) for adapt in self.adapts]
            self.adapt_step = lambda obs, r, done, info: reduce(lambda sub_tran, adapt: adapt._step(*sub_tran),
                                                                self.adapts, (obs, r, done, info))
            self.adapt_reset = lambda obs, info=None: reduce(lambda obs, adapt: adapt._reset(obs, info), self.adapts,
                                                             (obs))

        def step(self, action):
            tran = super().step(action)
            tran = self.adapt_step(*tran)
            return tran

        def reset(self):
            obs = super().reset()
            obs = self.adapt_reset(obs)
            return obs

    return Overhead(config)


class LibraryWrapper(Env):
    def __init__(self, config):
        self.env_name = config['env_name']
        self.base_env = {
            "OneSwitch": ag.OneSwitchJacoEnv,
            "Bottle": ag.BottleJacoEnv,
            "Valve": ag.ValveJacoEnv,
            "BlockPush": ag.BlockPushJacoEnv,
        }[config['env_name']]
        self.base_env = self.base_env(**config['env_kwargs'])
        self.observation_space = self.base_env.observation_space
        self.encoder_observation_space = None
        if hasattr(self.base_env, 'encoder_observation_space'):
            self.encoder_observation_space = self.base_env.encoder_observation_space
        self.action_space = self.base_env.action_space
        self.feature_sizes = self.base_env.feature_sizes
        self.terminate_on_failure = config['terminate_on_failure']

    def step(self, action):
        obs, r, done, info = self.base_env.step(action)
        if self.terminate_on_failure and hasattr(self.base_env, 'wrong_goal_reached'):
            done = done or self.base_env.wrong_goal_reached()
        return obs, r, done, info

    def reset(self):
        return self.base_env.reset()

    def render(self, mode=None, **kwargs):
        return self.base_env.render(mode)

    def seed(self, value):
        self.base_env.seed(value)

    def close(self):
        self.base_env.close()

    def get_base_env(self):
        return self.base_env


def action_factory(base):
    class Action(base):
        def __init__(self, config):
            super().__init__(config)
            self.action_type = config['action_type']
            self.action_space = {
                "trajectory": spaces.Box(-.1, .1, (3,)),
                "joint": spaces.Box(-.25, .25, (7,)),
                "disc_traj": spaces.Box(0, 1, (6,)),
            }[config['action_type']]
            self.translate = {
                'trajectory': self.trajectory,
                'joint': self.joint,
                'disc_traj': self.disc_traj,
            }[config['action_type']]
            self.smooth_alpha = config['smooth_alpha']

        def joint(self, action, info={}):
            clip_by_norm = lambda traj, limit: traj / max(1e-4, norm(traj)) * np.clip(norm(traj), None, limit)
            action = clip_by_norm(action, .25)
            info['joint'] = action
            return action, info

        def target(self, coor, info={}):
            base_env = self.base_env
            info['target'] = coor
            joint_states = p.getJointStates(base_env.robot, jointIndices=base_env.robot_left_arm_joint_indices,
                                            physicsClientId=base_env.id)
            joint_positions = np.array([x[0] for x in joint_states])

            link_pos = p.getLinkState(base_env.robot, 13, computeForwardKinematics=True, physicsClientId=base_env.id)[0]
            new_pos = np.array(coor) + np.array(link_pos) - base_env.tool_pos

            new_joint_positions = np.array(
                p.calculateInverseKinematics(base_env.robot, 13, new_pos, physicsClientId=base_env.id))
            new_joint_positions = new_joint_positions[:7]
            action = new_joint_positions - joint_positions
            return self.joint(action, info)

        def trajectory(self, traj, info={}):
            clip_by_norm = lambda traj, min_l=None, max_l=None: traj / max(1e-4, norm(traj)) * np.clip(norm(traj),
                                                                                                       min_l, max_l)
            traj = clip_by_norm(traj, .07, .1)
            info['trajectory'] = traj
            return self.target(self.base_env.tool_pos + traj, info)

        def disc_traj(self, onehot, info={}):
            info['disc_traj'] = onehot
            index = np.argmax(onehot)
            traj = [
                np.array((-1, 0, 0)),
                np.array((1, 0, 0)),
                np.array((0, -1, 0)),
                np.array((0, 1, 0)),
                np.array((0, 0, -1)),
                np.array((0, 0, 1)),
            ][index]
            return self.trajectory(traj, info)

        def step(self, action):
            action, ainfo = self.translate(action)
            obs, r, done, info = super().step(action)
            info = {**info, **ainfo}
            return obs, r, done, info

        def reset(self):
            self.action = np.zeros(7)
            return super().reset()

    return Action


def session_factory(base):
    class Session(base):
        def __init__(self, config):
            config['env_kwargs']['session_goal'] = True
            super().__init__(config)
            self.goal_reached = False

        def new_goal(self, index=None):
            self.base_env.set_target_index(index)
            self.base_env.reset_noise()
            self.goal_reached = False

        def step(self, action):
            o, r, d, info = super().step(action)
            if info['task_success']:
                self.goal_reached = True
            return o, r, d, info

        def reset(self):
            return super().reset()

    return Session


class array_to_dict:
    def __init__(self, master_env, config):
        pass

    def _step(self, obs, r, done, info):
        if not isinstance(obs, dict):
            obs = {'raw_obs': obs}
        return obs, r, done, info

    def _reset(self, obs, info=None):
        if not isinstance(obs, dict):
            obs = {'raw_obs': obs}
        return obs


class goal:
    """
    Chooses what features from info to add to obs
    """

    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.goal_feat_func = dict(
            Kitchen=lambda info: [info['target1_pos'], info['orders'], info['tasks']],
            Bottle=None,
            OneSwitch=None,
            Valve=None,
            BlockPush=lambda info: [info['ground_truth']]
        )[self.env_name]
        self.hindsight_feat = dict(
            Kitchen={'tool_pos': 3, 'orders': 2, 'tasks': 6},
            Bottle={'tool_pos': 3},
            OneSwitch={'tool_pos': 3},
            Valve={'valve_angle': 2},
            BlockPush={'ground_truth': 3}
        )[self.env_name]
        master_env.goal_size = self.goal_size = sum(self.hindsight_feat.values())

    def _step(self, obs, r, done, info):
        if self.goal_feat_func is not None:
            obs['goal'] = np.concatenate([np.ravel(state_component) for state_component in self.goal_feat_func(info)])

        hindsight_feat = np.concatenate(
            [np.ravel(info[state_component]) for state_component in self.hindsight_feat.keys()])

        obs['hindsight_goal'] = hindsight_feat
        return obs, r, done, info

    def _reset(self, obs, info=None):
        if self.goal_feat_func is not None:
            obs['goal'] = np.zeros(self.goal_size)

        obs['hindsight_goal'] = np.zeros(self.goal_size)
        return obs


class static_gaze:
    def __init__(self, master_env, config):
        self.gaze_dim = config['gaze_dim']
        del master_env.feature_sizes['goal']
        master_env.feature_sizes['gaze_features'] = self.gaze_dim
        self.env_name = master_env.env_name
        self.master_env = master_env
        with h5py.File(os.path.join(str(Path(__file__).resolve().parents[2]), 'gaze_capture', 'gaze_data',
                                    config['gaze_path']), 'r') as gaze_data:
            self.gaze_dataset = {k: v[()] for k, v in gaze_data.items()}
        self.per_step = True

    def sample_gaze(self, index):
        unique_target_index = index
        data = self.gaze_dataset[str(unique_target_index)]
        return self.master_env.rng.choice(data)

    def _step(self, obs, r, done, info):
        if self.per_step:
            if self.env_name == 'OneSwitch':
                self.static_gaze = self.sample_gaze(self.master_env.base_env.target_indices.index(info['unique_index']))
            elif self.env_name == 'Bottle':
                self.static_gaze = self.sample_gaze(info['unique_index'])
        obs['gaze_features'] = self.static_gaze
        return obs, r, done, info

    def _reset(self, obs, info=None):
        if self.env_name == 'OneSwitch':
            index = self.master_env.base_env.target_indices.index(self.master_env.base_env.unique_index)
        elif self.env_name == 'Bottle':
            index = self.master_env.base_env.unique_index
        obs['gaze_features'] = self.static_gaze = self.sample_gaze(index)
        return obs


class real_gaze:
    def __init__(self, master_env, config):
        self.gaze_dim = config['gaze_dim']
        del master_env.feature_sizes['goal']
        master_env.feature_sizes['gaze_features'] = self.gaze_dim
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.webcam = cv2.VideoCapture(0)
        self.face_processor = FaceProcessor(
            os.path.join(main_dir, 'gaze_capture', 'model_files', 'shape_predictor_68_face_landmarks.dat'))

        self.i_tracker = ITrackerModel()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.i_tracker.cuda()
            state = torch.load(os.path.join(main_dir, 'gaze_capture', 'checkpoint.pth.tar'))['state_dict']
        else:
            self.device = "cpu"
            state = torch.load(os.path.join(main_dir, 'gaze_capture', 'checkpoint.pth.tar'),
                               map_location=torch.device(ptu.device))['state_dict']
        self.i_tracker.load_state_dict(state, strict=False)

        self.gaze = np.zeros(self.gaze_dim)
        self.gaze_lock = threading.Lock()
        self.gaze_thread = None

    def record_gaze(self):
        _, frame = self.webcam.read()
        features = self.face_processor.get_gaze_features(frame)

        if features is None:
            print("GAZE NOT CAPTURED")
            gaze = np.zeros(self.gaze_dim)
        else:
            i_tracker_input = [torch.from_numpy(feature)[None].float().to(self.device) for feature in features]
            i_tracker_features = self.i_tracker(*i_tracker_input).detach().cpu().numpy()
            gaze = i_tracker_features[0]

        self.gaze_lock.acquire()
        self.gaze = gaze
        self.gaze_lock.release()

    def restart_gaze_thread(self):
        if self.gaze_thread is None or not self.gaze_thread.is_alive():
            self.gaze_thread = threading.Thread(target=self.record_gaze, name='gaze_thread')
            self.gaze_thread.start()

    def update_obs(self, obs):
        self.gaze_lock.acquire()
        obs['gaze_features'] = self.gaze
        self.gaze_lock.release()

    def _step(self, obs, r, done, info):
        self.restart_gaze_thread()
        self.update_obs(obs)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.restart_gaze_thread()
        self.update_obs(obs)
        return obs


class sim_target:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.feature = config.get('feature')
        del master_env.feature_sizes['goal']
        self.target_size = master_env.feature_sizes['target'] = 2 if self.env_name == 'Valve' else 3

        # should change to automate for all features eventually
        if self.feature == 'direction':
            self.target_size = master_env.feature_sizes['target'] = 3
        elif self.feature == 'target_position':
            self.target_size = master_env.feature_sizes['target'] = 2

        self.goal_noise_std = config['goal_noise_std']

    def _step(self, obs, r, done, info):
        self.add_target(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.add_target(obs, info)
        return obs

    def add_target(self, obs, info):
        if self.feature is None or self.feature is 'goal':
            target = obs['goal']
        elif info is None:
            target = np.zeros(self.target_size)
        else:
            target = info[self.feature]
        noise = np.random.normal(scale=self.goal_noise_std, size=target.shape) if self.goal_noise_std else 0
        obs['target'] = target + noise

from rl.policies.keyboard_policy import KeyboardPolicy
class keyboard:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.feature = config.get('feature')
        del master_env.feature_sizes['goal']
        self.size = master_env.feature_sizes['target'] = config.get('keyboard_size', 6)
        self.mode = config.get('mode')
        self.noise_p = config.get('keyboard_p')
        self.blank_p = config.get('blank_p')
        self.smoothing = config.get('smoothing')
        self.lag = config.get('lag')

        self.policy = KeyboardPolicy(master_env, demo=False)

    def _step(self, obs, r, done, info):
        self.add_target(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.policy.reset()
        self.action = np.zeros(self.size)
        self.lag_queue = deque(np.zeros((self.lag, self.size))) if self.lag else deque()
        self.add_target(obs, info)
        return obs

    def add_target(self, obs, info):
        action, _ = self.policy.get_action(obs)
        obs['user_input'] = action
        self.action = self.smoothing * self.action + action
        action = (1-self.smoothing)*self.action
        self.lag_queue.append(action)
        lag_action = self.lag_queue.popleft()
        action = lag_action
        obs['target'] = action

from rl.policies.encdec_policy import EncDecPolicy
import rlkit.torch.pytorch_util as ptu
import torch as th
class sim_keyboard:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.feature = config.get('feature')
        del master_env.feature_sizes['goal']
        self.size = master_env.feature_sizes['target'] = config.get('keyboard_size', 6)
        self.mode = config.get('mode')
        self.noise_p = config.get('keyboard_p')
        self.blank_p = config.get('blank_p')

        file_name = os.path.join('image','util_models', f'{self.env_name}_params_s1_sac.pkl')
        loaded = th.load(file_name, map_location=ptu.device)
        policy = loaded['trainer/policy']
        prev_vae = loaded['trainer/vae'].to(ptu.device)
        self.policy = EncDecPolicy(
            policy=policy,
            features_keys=['goal'],
            vaes=[prev_vae],
            deterministic=True,
            latent_size=4,
            incl_state=False,
        )

    def _step(self, obs, r, done, info):
        self.add_target(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.policy.reset()
        self.add_target(obs, info)
        return obs

    def add_target(self, obs, info):
        dist = norm(obs[self.feature] - obs['block_pos'])
        old_dist = norm(obs[self.feature] - obs['old_block_pos'])
        if self.mode == 'tool':
            traj = obs[self.feature] - obs['tool_pos']
            axis = np.argmax(np.abs(traj))
            index = 2 * axis + (traj[axis] > 0)
        elif self.mode == 'block':
            traj = obs[self.feature] - obs['block_pos']
            axis = np.argmax(np.abs(traj))
            index = 2 * axis + (traj[axis] > 0)
        elif self.mode == 'sip-puff':
            index = dist < old_dist
        elif self.mode == 'xy':
            traj = obs[self.feature][:2] - obs['block_pos'][:2]
            axis = np.argmax(np.abs(traj))
            index = 2 * axis + (traj[axis] > 0)
        elif self.mode == 'oracle':
            oracle_action, _ = self.policy.get_action(obs)
            axis = np.argmax(np.abs(oracle_action))
            index = 2 * axis + (oracle_action[axis] > 0)

        if np.random.uniform() < self.noise_p:
            index = np.random.randint(self.size)
        action = np.zeros(self.size)
        action[index] = 1
        if np.random.uniform() < self.blank_p:
            action = np.zeros(self.size)

        if self.mode == 'sip-puff':
            action[-3:] = obs['old_block_pos']
        obs['target'] = action

from rl.policies.block_push_oracle import BlockPushOracle
class oracle:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.feature = config.get('feature')
        del master_env.feature_sizes['goal']
        self.size = master_env.feature_sizes['target'] = config.get('keyboard_size', 7)
        self.blank_p = config.get('blank_p',0)
        self.spread = config.get('oracle_noise',0)
        self.smoothing = config.get('smoothing',0)
        self.lag = 0

        file_name = os.path.join('image','util_models', f'{self.env_name}_params_s1_sac.pkl')
        loaded = th.load(file_name, map_location=ptu.device)
        policy = loaded['trainer/policy']
        prev_vae = loaded['trainer/vae'].to(ptu.device)
        self.policy = EncDecPolicy(
            policy=policy,
            features_keys=['goal'],
            vaes=[prev_vae],
            deterministic=True,
            latent_size=4,
            incl_state=False,
        )
        self.use_tool_action = config.get('use_tool_action',False)

    def _step(self, obs, r, done, info):
        self.add_target(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.policy.reset()
        self.action = np.zeros(self.size)
        self.lag_queue = deque(np.zeros((self.lag, self.size))) if self.lag else deque()
        self.add_target(obs, info)
        return obs

    def add_target(self, obs, info):
        action, _ = self.policy.get_action(obs)
        action += np.random.normal(np.zeros(action.shape), self.spread)
        if np.random.uniform() < self.blank_p:
            action = np.zeros(action.shape)
        self.action = self.smoothing * self.action + action
        action = (1-self.smoothing)*self.action
        self.lag_queue.append(action)
        lag_action = self.lag_queue.popleft()
        action = lag_action
 
        obs['target'] = action


class joint:
    def __init__(self, master_env, config):
        master_env.observation_space = spaces.Box(-np.inf, np.inf, (master_env.observation_space.low.size + 7,))

    def _step(self, obs, r, done, info):
        obs['raw_obs'] = np.concatenate((obs['raw_obs'], obs['joint']))
        return obs, r, done, info

    def _reset(self, obs, info=None):
        obs['raw_obs'] = np.concatenate((obs['raw_obs'], obs['joint']))
        return obs

class dict_to_array:
    def __init__(self, master_env, config):
        pass

    def _step(self, obs, r, done, info):
        obs = np.concatenate((obs['raw_obs'], obs['target']))
        return obs, r, done, info

    def _reset(self, obs, info=None):
        obs = np.concatenate((obs['raw_obs'], obs['target']))
        return obs


class reward:
    """ rewards capped at 'cap' """

    def __init__(self, master_env, config):
        self.range = (config['reward_min'], config['reward_max'])
        self.master_env = master_env
        self.reward_type = config.get('reward_type')
        self.reward_temp = config.get('reward_temp')
        self.reward_offset = config.get('reward_offset')

    def _step(self, obs, r, done, info):
        if self.reward_type == 'custom':
            r = -1
            r += np.exp(-norm(info['tool_pos'] - info['target1_pos'])) / 2
            if info['target1_reached']:
                r = -.5
                r += np.exp(-norm(info['tool_pos'] - info['target_pos'])) / 2
            if info['task_success']:
                r = 0
        elif self.reward_type == 'custom_kitchen':
            r = -1
            if not info['tasks'][0] and (info['orders'][0] == 0 or info['tasks'][1]):
                r += np.exp(-10 * max(0, info['microwave_angle'] - -.7)) / 6 * 3 / 4 * 1 / 2
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['microwave_handle'])) / 6 / 4 * 1 / 2
            elif info['tasks'][0]:
                r += 1 / 6
            if not info['tasks'][1] and (info['orders'][0] == 1 or info['tasks'][0]):
                r += np.exp(-10 * max(0, .7 - info['fridge_angle'])) / 6 * 3 / 4 * 1 / 2
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['fridge_handle'])) / 6 / 4 * 1 / 2
            elif info['tasks'][1]:
                r += 1 / 6

            if not info['tasks'][2] and info['tasks'][0] and info['tasks'][1]:
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['target1_pos'])) / 6 * 1 / 2
            elif info['tasks'][2]:
                r = -1 / 2
            if not info['tasks'][3] and info['tasks'][2]:
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['target_pos'])) / 6 * 1 / 2
            elif info['tasks'][3]:
                r = -1 / 3

            if not info['tasks'][4] and info['tasks'][3] and (info['orders'][1] == 0 or info['tasks'][5]):
                r += np.exp(-norm(info['microwave_angle'] - 0)) / 6 * 3 / 4 * 1 / 2
                dist = norm(info['tool_pos'] - info['microwave_handle'])
                if dist > .25:
                    r += np.exp(-self.reward_temp * dist) / 6 / 4 * 1 / 2
                else:
                    r += np.exp(-self.reward_temp * .25) / 6 / 4 * 1 / 2
            elif info['tasks'][4]:
                r += 1 / 6
            if not info['tasks'][5] and info['tasks'][3] and (info['orders'][1] == 1 or info['tasks'][4]):
                r += np.exp(-norm(info['fridge_angle'] - 0)) / 6 * 3 / 4 * 1 / 2
                dist = norm(info['tool_pos'] - info['fridge_handle'])
                if dist > .25:
                    r += np.exp(-self.reward_temp * dist) / 6 / 4 * 1 / 2
                else:
                    r += np.exp(-self.reward_temp * .25) / 6 / 4 * 1 / 2
            elif info['tasks'][5]:
                r += 1 / 6

            if info['task_success']:
                r = 0

        elif self.reward_type == 'dist':
            r = 0
            if not info['task_success']:
                dist = np.linalg.norm(info['tool_pos'] - info['target_pos'])
                r = np.exp(-self.reward_temp * dist + np.log(1 + self.reward_offset)) - 1
        elif self.reward_type == 'custom_switch':
            r = 0
            if not info['task_success']:
                dist = np.linalg.norm(info['tool_pos'] - info['switch_pos'][info['target_index']])
                r = np.exp(-self.reward_temp * dist + np.log(1 + self.reward_offset)) - 1

        elif self.reward_type == 'sparse':
            r = -1 + info['task_success']
        elif self.reward_type == 'part_sparse':
            r = -1 + .5 * (info['task_success'] + info['door_open'])
        elif self.reward_type == 'terminal_interrupt':
            r = info['noop']
        elif self.reward_type == 'part_sparse_kitchen':
            r = -1 + sum(info['tasks']) / 6
        elif self.reward_type == 'valve_exp':
            dist = np.abs(self.master_env.base_env.angle_diff(info['valve_angle'], info['target_angle']))
            r = np.exp(-self.reward_temp * dist) - 1
        elif self.reward_type == 'blockpush_exp':
            r = -1
            dist = norm(info['block_pos']-info['target_pos']) + norm(info['tool_pos'] - info['block_pos'])/2
            old_dist = norm(info['old_block_pos']-info['target_pos']) + norm(info['old_tool_pos'] - info['old_block_pos'])/2
            under_table_penalty = max(0, info['target_pos'][2]-info['tool_pos'][2]-.1)
            sigmoid = lambda x: 1/(1 + np.exp(-x))
            r += sigmoid(self.reward_temp*(old_dist-dist-under_table_penalty))*self.reward_offset
            if info['task_success']:
                r = 0
        else:
            raise Exception

        r = np.clip(r, *self.range)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        return obs

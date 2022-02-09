import rlkit.torch.pytorch_util as ptu

from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rl.scripts.run_util import run_exp

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch import optim
from copy import deepcopy
from functools import reduce
import operator


def experiment(variant):
    import torch as th

    expl_config = deepcopy(variant['env_config'])
    expl_config['factories'] += ['session']
    env = default_overhead(expl_config)

    eval_config = deepcopy(variant['env_config'])
    eval_config['gaze_path'] = eval_config['eval_gaze_path']
    eval_env = default_overhead(eval_config)

    M = variant["layer_size"]

    feat_dim = env.observation_space.low.size + reduce(operator.mul,
                                                       getattr(env.base_env, 'goal_set_shape', (0,)), 1)
    obs_dim = feat_dim + sum(env.feature_sizes.values())
    action_dim = 7

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    expl_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(expl_policy)

    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )

    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False,
        real_user=variant['real_user']
    )
    replay_buffer = SimpleReplayBuffer(
        variant['replay_buffer_size'],
        obs_dim,
        action_dim,
        env_info_sizes={}
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=expl_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchCalibrationRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        calibration_data_collector=None,
        calibration_buffer=replay_buffer,
        real_user=variant['real_user'],
        **variant['algorithm_args']
    )
    algorithm.to(ptu.device)

    if variant.get('render', False):
        env.render('human')
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='sac_scratch')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    parser.add_argument('--mode', default='default', type=str)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    target_indices = [1, 2, 3] if args.env_name == 'OneSwitch' else None
    goal_noise_std = 0.1 if args.env_name == 'OneSwitch' else 0.15
    default_variant = dict(
        real_user=not args.sim,
        pretrain_path=f'{args.env_name}_params_s1_sac.pkl',
        latent_size=3,
        layer_size=256,
        replay_buffer_size=int(1e4 * path_length),
        keep_calibration_data=True,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=args.epochs,
            num_eval_steps_per_epoch=1000,
            num_train_loops_per_epoch=1,
            collect_new_paths=True,
            pretrain_steps=0,
            max_failures=5,
            eval_paths=False,
        ),

        env_config=dict(
            env_name=args.env_name,
            goal_noise_std=goal_noise_std,
            terminate_on_failure=True,
            env_kwargs=dict(step_limit=path_length, frame_skip=5, debug=False, target_indices=target_indices),
            action_type='joint',
            smooth_alpha=1,
            factories=[],
            adapts=['goal','dict_to_array','reward'],
            reward_type='sparse',
            reward_max=0,
            reward_min=-1,
            reward_temp=1,
            reward_offset=-0.2,
            gaze_dim=128,
            gaze_path=f'{args.env_name}_gaze_data_train.h5',
            eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5'
        )
    )
    variants = []

    search_space = {
        'algorithm_args.trajs_per_index': [0],
        'lr': [5e-4],
        'algorithm_args.relabel_failures': [True],
        'algorithm_args.num_trains_per_train_loop': [100],
        'seedid': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'freeze_decoder': [True],
        'mode': ['default'],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=default_variant,
    )
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']

        if not args.use_ray:
            variant['render'] = args.no_render

        mode_dict = {'OneSwitch':
                         {'default': {'calibrate_split': False,
                                      'calibration_indices': [1, 2, 3]},
                          'no_online': {'calibrate_split': False,
                                        'calibration_indices': [1, 2, 3],
                                        'num_trains_per_train_loop': 0},
                          'shift': {'calibrate_split': True,
                                    'calibration_indices': [1, 2, 3]},
                          'no_right': {'calibrate_split': False,
                                       'calibration_indices': [2, 3]},
                          'overcalibrate': {'calibrate_split': False,
                                            'calibration_indices': [0, 1, 2, 3, 4]}
                          },
                     'Bottle':
                         {'default': {'calibrate_split': False,
                                      'calibration_indices': [0, 1, 2, 3]},
                          'no_online': {'calibrate_split': False,
                                        'calibration_indices': [0, 1, 2, 3],
                                        'num_trains_per_train_loop': 0},
                          'shift': {'calibrate_split': True,
                                    'calibration_indices': [0, 1, 2, 3]},
                          'no_door': {'calibrate_split': False,
                                      'calibration_indices': [1, 2]},
                          'with_door': {'calibrate_split': False,
                                        'calibration_indices': [0, 3]}

                          }
                     }[variant['env_config']['env_name']][variant['mode']]

        variant['algorithm_args'].update(mode_dict)

        target = 'real_gaze' if variant['real_user'] else 'sim_target'
        variant['env_config']['adapts'].insert(1,target)

    args.process_args = process_args

    run_exp(experiment, variants, args)

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, VAE
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.policies import ConcatTanhGaussianPolicy

from rl.policies import EncDecPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.trainers import EncDecSACTrainer
from rl.replay_buffers import ModdedReplayBuffer
from rl.scripts.run_util import run_exp
from rl.misc.simple_path_loader import SimplePathLoader
from rlkit.torch.networks import Clamp
from rl.trainers import TorchEncDecAWACTrainer

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from functools import reduce
import operator
import numpy as np


def experiment(variant):
    import torch as th

    env = default_overhead(variant['env_config'])
    env.seed(variant['seedid'])
    eval_config = variant['env_config'].copy()
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    feat_dim = env.observation_space.low.size + reduce(operator.mul,
                                                       getattr(env.base_env, 'goal_set_shape', (0,)), 1)
    obs_dim = feat_dim + sum(env.feature_sizes.values())

    action_dim = env.action_space.low.size
    M = variant["layer_size"]

    encoder_obs_space = env.encoder_observation_space
    if encoder_obs_space is None:
        encoder_obs_space = env.observation_space
    encoder_obs_dim = encoder_obs_space.low.size + sum(env.feature_sizes.values())

    if not variant['from_pretrain']:
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
        policy = ConcatTanhGaussianPolicy(
            obs_dim=feat_dim + variant['latent_size'],
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )
        vae = VAE(input_size=encoder_obs_dim if variant['incl_state'] else sum(env.feature_sizes.values()),
                  latent_size=variant['latent_size'],
                  encoder_hidden_sizes=[64],
                  decoder_hidden_sizes=[64]
                  ).to(ptu.device)
    else:
        file_name = os.path.join('image/util_models', variant['pretrain_path'])
        loaded = th.load(file_name)
        qf1 = loaded['trainer/qf1']
        qf2 = loaded['trainer/qf2']
        target_qf1 = loaded['trainer/target_qf1']
        target_qf2 = loaded['trainer/target_qf2']
        policy = loaded['trainer/policy']
        vae = loaded['trainer/vae']

    expl_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vaes=[vae],
        incl_state=variant['incl_state'],
        sample=False,
        deterministic=False
    )

    eval_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vaes=[vae],
        incl_state=variant['incl_state'],
        sample=False,
        deterministic=True
    )

    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )

    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False,
    )
    trainer = EncDecSACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae,
        latent_size=variant['latent_size'],
        incl_state=variant['incl_state'],
        **variant['trainer_kwargs']
    )
    replay_buffer = ModdedReplayBuffer(
        variant['replay_buffer_size'],
        env,
        sample_base=0,
        store_latents=False
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_args']
    )
    algorithm.to(ptu.device)

    path_loader = SimplePathLoader(
        demo_path=variant['demo_paths'],
        demo_path_proportion=variant['demo_path_proportions'],
        replay_buffer=replay_buffer,
    )
    path_loader.load_demos()

    if variant['pretrain_steps']:
        awac_trainer = TorchEncDecAWACTrainer(
            policy=policy,
            policy_lr=variant['trainer_kwargs']['policy_lr'],
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            qf_lr=variant['trainer_kwargs']['qf_lr'],
            vae=vae,
            latent_size=variant['latent_size'],
            beta=variant['trainer_kwargs']['beta'],
            sample=variant['trainer_kwargs']['sample'],
            soft_target_tau=variant['trainer_kwargs']['soft_target_tau'],
            target_update_period=variant['trainer_kwargs']['target_update_period'],
            incl_state=variant['incl_state']
        )
        for _ in range(variant['pretrain_steps']):
            train_data = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
            awac_trainer.train(train_data)

    if variant.get('render', False):
        env.render('human')
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Valve')
    parser.add_argument('--exp_name', default='pretrain_sac_valve')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--incl_state', action='store_true')
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        pretrain_path=f'{args.env_name}_params_s1_sac_det.pkl',
        latent_size=2,
        layer_size=256,
        incl_state=args.incl_state,
        pretrain_steps=25000,
        algorithm_args=dict(
            num_epochs=10000,
            num_eval_steps_per_epoch=0,
            eval_paths=False,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=path_length,
            batch_size=256,
            collect_new_paths=True,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            encoder_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            sample=not args.det,
            beta=0 if args.det else 1e-4
        ),
        demo_paths=[
            os.path.join(main_dir, "demos", f"{args.env_name}_keyboard_on_policy_50_test.npy"),
        ],
        env_config=dict(
            terminate_on_failure=False,
            env_name=args.env_name,
            goal_noise_std=0,
            env_kwargs=dict(frame_skip=5, debug=False, num_targets=None, stochastic=True,
                            min_error_threshold=np.pi / 32, use_rand_init_angle=True, term_cond=None),
            action_type='joint',
            smooth_alpha=1,
            factories=[],
            adapts=['goal', 'reward'],
            gaze_dim=128,
            state_type=0,
            reward_type='valve_exp',
            reward_min=None,
            reward_max=0,
            reward_temp=5
        )
    )
    search_space = {
        'seedid': [2000],
        'from_pretrain': [False],
        'algorithm_args.num_trains_per_train_loop': [1000],
        'replay_buffer_size': [int(1e6)],
        'demo_path_proportions': [[100]],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)


    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']
        if not args.use_ray:
            variant['render'] = args.no_render


    args.process_args = process_args

    run_exp(experiment, variants, args)

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


def experiment(variant):
    import torch as th

    env = default_overhead(variant['env_config'])
    env.seed(variant['seedid'])
    eval_config = variant['env_config'].copy()
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    feat_dim = env.observation_space.low.size
    goal_dim = sum(env.feature_sizes.values())
    obs_dim = feat_dim + goal_dim
    action_dim = env.action_space.low.size
    M = variant["layer_size"]

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
        vae = VAE(input_size=obs_dim if variant['incl_state'] else goal_dim,
                  latent_size=variant['latent_size'],
                  encoder_hidden_sizes=[64],
                  decoder_hidden_sizes=[64]
                  ).to(ptu.device)
    else:
        file_name = os.path.join('util_models', variant['pretrain_path'])
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
    parser.add_argument('--env_name', default='BlockPush')
    parser.add_argument('--exp_name', default='pretrain_sac_blockpush')
    parser.add_argument('--render', action='store_true')
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
        # latent_size=8,
        layer_size=256,
        incl_state=args.incl_state,
        pretrain_steps=0,
        algorithm_args=dict(
            num_epochs=int(1e4),
            num_eval_steps_per_epoch=0,
            eval_paths=False,
            # num_expl_steps_per_train_loop=200,
            # min_num_steps_before_training=0,
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
            sample=True,
            # beta=0
        ),
        demo_paths=[
            os.path.join(main_dir, "demos", f"{args.env_name}_1000_optimal.npy"),
        ],
        env_config=dict(
            terminate_on_failure=False,
            env_name=args.env_name,
            step_limit=path_length,
            goal_noise_std=0,
            env_kwargs=dict(),
            action_type='disc_traj',
            smooth_alpha=1,
            factories=[],
            adapts=['goal','reward'],
            gaze_dim=128,
            state_type=0,
            # reward_type='part_sparse',
            reward_min=-1,
            reward_max=0,
        ),
        render=args.render,
    )
    search_space = {
        'seedid': [2000],
        'from_pretrain': [False],
        'algorithm_args.num_trains_per_train_loop': [500],
        'replay_buffer_size': [int(1e6)],
        'demo_path_proportions': [[]],
        'latent_size': [4],
        'env_config.reward_type': ['blockpush_exp'],
        'trainer_kwargs.beta': [0],
        'env_config.reward_temp': [5,10],
        # 'env_config.reward_offset': [.25, .5,]
        # 'env_config.reward_temp': [50,],
        'env_config.reward_offset': [.25,.5,.75]
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)


    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']
        if args.use_ray:
            variant['render'] = False
        if args.render:
            variant['algorithm_args'].update(dict(
                num_expl_steps_per_train_loop=200,
                min_num_steps_before_training=0,
            ))


    args.process_args = process_args

    run_exp(experiment, variants, args)

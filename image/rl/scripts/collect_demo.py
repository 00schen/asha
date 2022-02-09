from rl.policies import EncDecPolicy, DemonstrationPolicy, KeyboardPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
import rlkit.pythonplusplus as ppp

import os
from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy

import torch as th
from types import MethodType


def collect_demonstrations(variant):
    import time

    current_time = time.time_ns()
    env = default_overhead(variant['env_kwargs']['config'])
    env.seed(variant['seedid'] + 100 + current_time)

    file_name = os.path.join(variant['eval_path'])
    loaded = th.load(file_name, map_location='cpu')
    policy = EncDecPolicy(
        policy=loaded['trainer/policy'],
        features_keys=list(env.feature_sizes.keys()),
        vaes=[loaded['trainer/vae']],
        incl_state=False,
        sample=False,
        deterministic=False
    )

    # policy = FollowerPolicy(env)
    # policy = DemonstrationPolicy(policy, env, p=variant['p'])
    # policy = KeyboardPolicy()

    path_collector = FullPathCollector(
        env,
        policy
    )

    if variant.get('render', False):
        env.render('human')
    paths = []
    success_count = 0
    while len(paths) < variant['num_episodes']:
        collected_paths = path_collector.collect_new_paths(
            variant['path_length'],
            1,
        )
        for path in collected_paths:
            paths.append(path)
            success_count += path['env_infos'][-1]['task_success']
        print("total paths collected: ", len(paths), "successes: ", success_count)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--suffix', default='test')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    args, _ = parser.parse_known_args()
    main_dir = str(Path(__file__).resolve().parents[2])
    print(main_dir)

    path_length = 100
    variant = dict(
        seedid=3000,
        eval_path=os.path.join(main_dir, 'util_models', 'OneSwitch_params_s1_sac.pkl'),
        env_kwargs={'config': dict(
            env_name='OneSwitch',
            env_kwargs=dict(frame_skip=5, debug=False, num_targets=5, stochastic=True,
                            min_error_threshold=np.pi / 32, use_rand_init_angle=True),
            oracle='keyboard',
            oracle_kwargs=dict(
                threshold=.5,
            ),
            action_type='joint',
            smooth_alpha=1,

            factories=[],
            adapts=['goal',],
            state_type=0,
            apply_projection=False,
            reward_max=0,
            reward_min=-1,
            input_penalty=1,
            reward_type='sparse',
            terminate_on_failure=False,
            goal_noise_std=0,
            reward_temp=1,
            reward_offset=-0.2
        )},
        render=args.no_render and (not args.use_ray),

        on_policy=True,
        p=1,
        num_episodes=50,
        path_length=path_length,
        save_name_suffix=args.suffix,

    )
    search_space = {
        'env_kwargs.config.oracle_kwargs.epsilon': 0 if variant['on_policy'] else .7,  # higher epsilon = more noise
    }
    search_space = ppp.dot_map_dict_to_nested_dict(search_space)
    variant = ppp.merge_recursive_dicts(variant, search_space)


    def process_args(variant):
        variant['env_kwargs']['config']['seedid'] = variant['seedid']
        variant['save_name'] = \
            f"{variant['env_kwargs']['config']['env_name']}_{variant['env_kwargs']['config']['oracle']}" \
            + f"_{'on_policy' if variant['on_policy'] else 'off_policy'}_{variant['num_episodes']}" \
            + "_" + variant['save_name_suffix']


    if args.use_ray:
        import ray
        from itertools import count

        ray.init(_temp_dir='/tmp/ray_exp1', num_gpus=0)


        @ray.remote
        class Iterators:
            def __init__(self):
                self.run_id_counter = count(0)

            def next(self):
                return next(self.run_id_counter)


        iterator = Iterators.options(name="global_iterator").remote()

        process_args(variant)


        @ray.remote(num_cpus=1, num_gpus=0)
        class Sampler:
            def sample(self, variant):
                variant = deepcopy(variant)
                variant['seedid'] += ray.get(iterator.next.remote())
                return collect_demonstrations(variant)


        num_workers = 16
        variant['num_episodes'] = variant['num_episodes'] // num_workers

        samplers = [Sampler.remote() for i in range(num_workers)]
        samples = [samplers[i].sample.remote(variant) for i in range(num_workers)]
        samples = [ray.get(sample) for sample in samples]
        paths = list(sum(samples, []))
        np.save(os.path.join(main_dir, "demos", variant['save_name']), paths)
    else:
        import time

        current_time = time.time_ns()
        variant['seedid'] = current_time
        process_args(variant)
        paths = collect_demonstrations(variant)
        np.save(os.path.join(main_dir, "demos", variant['save_name']), paths)

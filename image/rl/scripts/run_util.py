import gtimer as gt
import rlkit.torch.pytorch_util as ptu
import os
from rlkit.launchers.launcher_util import setup_logger, reset_execution_environment


def run_exp(experiment, variants, args):
    if args.use_ray:
        import ray
        from ray.util import ActorPool
        from itertools import count
        ray.init(_temp_dir='/tmp/ray_exp1', num_gpus=args.gpus)

        @ray.remote
        class Iterators:
            def __init__(self):
                self.run_id_counter = count(0)

            def next(self):
                return next(self.run_id_counter)

        iterator = Iterators.options(name="global_iterator").remote()

        @ray.remote(num_cpus=1, num_gpus=1 / args.per_gpu if args.gpus else 0)
        class Runner:
            def new_run(self, variant):
                gt.reset_root()
                ptu.set_gpu_mode(args.gpus > 0)
                args.process_args(variant)
                iterator = ray.get_actor("global_iterator")
                run_id = ray.get(iterator.next.remote())
                save_path = os.path.join(args.main_dir, 'logs')
                reset_execution_environment()
                setup_logger(exp_prefix=args.exp_name, variant=variant, base_log_dir=save_path,
                             exp_id=run_id, snapshot_mode='gap_and_last', snapshot_gap=50)
                experiment(variant)

        runners = [Runner.remote() for i in range(args.gpus * args.per_gpu if args.gpus > 0 else args.per_gpu)]
        runner_pool = ActorPool(runners)
        list(runner_pool.map(lambda a, v: a.new_run.remote(v), variants))
    else:
        variant = variants[0]
        ptu.set_gpu_mode(False)
        args.process_args(variant)
        save_path = os.path.join(args.main_dir, 'logs')
        reset_execution_environment()
        setup_logger(exp_prefix=args.exp_name, variant=variant, base_log_dir=save_path, exp_id=0, )
        experiment(variant)

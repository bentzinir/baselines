import sys
import re
import multiprocessing
import os.path as osp
import gym #, gym_maze
from collections import defaultdict
import tensorflow as tf
import numpy as np
import copy

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from baselines.common.misc_util import set_default_value

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    # env_type = env.entry_point.split(':')[0].split('.')[-1]
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    extra_args['mode'] = set_default_value(extra_args, 'mode', 'maximum_span')
    extra_args['distance_threshold'] = set_default_value(extra_args, 'distance_threshold', None)
    extra_args['cover_distance_threshold'] = set_default_value(extra_args, 'cover_distance_threshold', None)

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)
    alg_kwargs['log_path'] = args.log_path

    override_params = extra_args

    if extra_args['mode'] == 'maximum_span':
        _args = copy.deepcopy(args)
        _args.num_env = 1
        env = build_env(_args, extra_args=extra_args)
    else:
        env = build_env(args, extra_args=extra_args)

    alg_kwargs['load_mca_path'] = set_default_value(alg_kwargs, 'load_mca_path', None)
    alg_kwargs['distance_threshold'] = extra_args['distance_threshold']

    # build overloaded env for the mca explorer

    mca_extra_args = copy.deepcopy(extra_args)
    mca_extra_args["ss"] = set_default_value(extra_args, 'ss', False)
    mca_env = build_env(args, extra_args=mca_extra_args)

    # os.path.join(save_path, time.strftime("%Y-%m-%d-%H-%M-%S"))

    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

        mca_env = VecVideoRecorder(mca_env, osp.join(logger.get_dir(), "mca_videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    # create non-vectorized environment for evaluating cover distance
    alg_kwargs['cover_measure_env'] = gym.make(env_id, distance_threshold=alg_kwargs['cover_distance_threshold'],
                                               mode=alg_kwargs['mode'])

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        mca_env=mca_env,
        seed=seed,
        total_timesteps=total_timesteps,
        override_params=override_params,
        **alg_kwargs
    )

    return model, env


def build_env(args, extra_args=None):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale,
                           flatten_dict_observations=flatten_dict_observations,
                           env_kwargs=extra_args
                           )

        # TODO: add back after debug
        # if env_type == 'mujoco':
        #     env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        # env_type = env.entry_point.split(':')[0].split('.')[-1]
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        return logger.configure(log_path)
    else:
        return logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        # import time
        rank = 0
        # if args.log_path:
        #     args.log_path = osp.join(args.log_path, time.strftime("%Y-%m-%d-%H-%M-%S"))
        args.log_path = configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    if args.play:
        args.num_timesteps = 0
        args.num_env = 1

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None

        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)
            if isinstance(actions, list):
                actions = actions[0]
            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            if args.vis_sleep > 0:
                import time
                time.sleep(args.vis_sleep)
            # print(f"gc:{obs['observation'][...,3:]}")
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
    env.close()

    return model


if __name__ == '__main__':
    main(sys.argv)
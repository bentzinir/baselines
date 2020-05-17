import sys
from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs
from baselines.her.metric_diversification import MetricDiversifier
from baselines.her.cover_measure import init_from_point
import gym
import numpy as np
import random
import time
from baselines.her.experiment import config
from baselines.common import tf_util
import copy
np.set_printoptions(precision=2)


def set_goal(env, cover):
    if cover is None:
        return env.reset()
    pnt = cover[random.choice(range(len(cover)))]
    return env.set_goal(goal=pnt['ag'])


def reset_env(env, cover, mode='intrinsic'):
    if mode == 'intrinsic':
        return env.reset()
    elif mode == 'extrinsic':
        assert 'cover_path' is not None, 'missing cover path argument'
        idxs = list(range(len(cover)))
        # idxs = [352, 630]
        obs = init_from_point(env, cover[random.choice(idxs)])
        # print(f"Achieved goal: {obs['achieved_goal']}")
        return obs
    elif mode == 'random':
        obs = env.reset()
        qpos = obs["qpos"]
        qvel = obs["qvel"]
        ex_init = {'o': None, 'qpos': np.zeros_like(qpos), 'qvel': np.zeros_like(qvel), 'g': None}
        env.reset(ex_init=ex_init)


def scan_cover(env, action_repetition=1, cover_path=None, **kwargs):
    cover = MetricDiversifier.load_model(cover_path)
    obs = reset_env(env, cover, mode='intrinsic')
    for i in range(100000):
        env.render()
        time.sleep(.3)
        if i % action_repetition == 0:
            a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        if i % 1 == 0:
            ob = reset_env(env, cover, mode='extrinsic')
            # print(np.linalg.norm(ob["qvel"]))
            time.sleep(.5)
    env.close()


def plain_loop(env, action_repetition=1, clip_range=0.5, **kwargs):
    reset_env(env, cover=None, mode='intrinsic')
    i = 0
    while True:
        i += 1
        env.render()
        time.sleep(.01)
        if i % action_repetition == 0:
            a = np.clip(env.action_space.sample(), -clip_range, clip_range)
        o, r, d, info = env.step(a)
        if i % 1000 == 0:
            reset_env(env, cover=None, mode='intrinsic')
            print(f"Reset")
            i = 0
    env.close()


def play_policy(env, env_id, load_path=None, cover_path=None, semi_metric=False, **kwargs):
    params = config.DEFAULT_PARAMS
    _override_params = copy.deepcopy(kwargs)
    params.update(**_override_params)
    params['env_name'] = env_id
    params = config.prepare_params(params)
    dims, coord_dict = config.configure_dims(params)
    params['ddpg_params']['scope'] = "mca"
    policy, reward_fun = config.configure_ddpg(dims=dims, params=params, active=True, clip_return=True)
    tf_util.load_variables(load_path)
    print(f"Loaded model: {load_path}")
    cover = MetricDiversifier.load_model(cover_path)
    obs = reset_env(env, cover, mode='intrinsic')
    i = 0
    while True:
        i += 1
        # print(i)
        env.render()
        time.sleep(.03)
        action, _, state, _ = policy.step(obs)
        # if i % 10 == 0:
        #     action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(f"achieved: {obs['achieved_goal']}, desired: {obs['desired_goal']}, obs: {obs['observation'][:6]}")
        if i % 50 == 0 or info['is_success']:
            if cover is None or semi_metric:
                reset_env(env, cover, mode='intrinsic')
            else:
                reset_env(env, cover, mode='extrinsic')
            obs = set_goal(env, cover)
            if info['is_success']:
                input(f"success at:{i}")
            i = 0
    env.close()


if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    extra_args = parse_cmdline_kwargs(unknown_args)
    environment = gym.make(args.env, **extra_args)
    if extra_args['option'] == 'scan_cover':
        scan_cover(environment, **extra_args)
    elif extra_args['option'] == 'plain_loop':
        plain_loop(environment, **extra_args)
    elif extra_args['option'] == 'play_policy':
        assert extra_args['load_path'] is not None
        # assert extra_args['cover_path'] is not None
        play_policy(environment, args.env, **extra_args)

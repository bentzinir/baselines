import sys
from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs
from baselines.her.metric_diversification import MetricDiversifier
from baselines.her.cover_measure import init_from_point
import gym
import numpy as np
import random
import time
np.set_printoptions(precision=4)


def reset_env(env, mode='intrinsic', cover_path=None):
    if mode == 'intrinsic':
        return env.reset()
    elif mode == 'extrinsic':
        assert 'cover_path' is not None, 'missing cover path argument'
        cover = MetricDiversifier.load_model(cover_path)
        idxs = list(range(len(cover)))
        # idxs = [352, 630]
        obs = init_from_point(env, cover[random.choice(idxs)])
        # print(f"Achieved goal: {obs['achieved_goal']}")
        return obs
    elif mode == 'random':
        obs = env.reset()
        qpos = obs["qpos"]
        qvel = obs["qvel"]
        ex_init = {'x': None, 'qpos': np.zeros_like(qpos), 'qvel': np.zeros_like(qvel), 'g': None}
        env.reset(ex_init=ex_init)


def scan_cover(env, action_repetition=1, **kwargs):
    reset_env(env, mode='intrinsic')
    for i in range(100000):
        env.render()
        time.sleep(2)
        if i % action_repetition == 0:
            a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        if i % 1 == 0:
            ob = reset_env(env, mode='extrinsic', cover_path=kwargs['cover_path'])
            # print(np.linalg.norm(ob["qvel"]))
    env.close()


def plain_loop(env, action_repetition=1, **kwargs):
    reset_env(env, mode='intrinsic')
    for i in range(100000):
        env.render()
        time.sleep(.01)
        if i % action_repetition == 0:
            a = env.action_space.sample()
            # a = np.array([-0.9639, -0.5384])
        obs, reward, done, info = env.step(a)

        if i % 100 == 0:
            reset_env(env, mode='intrinsic')
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

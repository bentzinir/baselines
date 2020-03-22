import argparse
import gym
from baselines.her.metric_diversification import MetricDiversifier
from baselines.her.metric_diversification import VisObserver
from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs
from baselines.run import build_env
import os, sys
import itertools
import numpy as np
from matplotlib import pyplot as plt
import collections
from baselines.her.metric_diversification import Bunch


def plot(cover_a, cover_b, log_directory):

    fig, ax = plt.subplots(1, 1)

    def cover_plot(cover, color):
        x = list(cover.keys())
        y = np.asarray([np.asarray(item).mean() for item in cover.values()])
        error = np.asarray([np.asarray(item).std() for item in cover.values()])

        ax.plot(x, y, f"{color}-")
        ax.fill_between(x, y - error, y + error, facecolor=color, alpha=0.5)

    cover_plot(cover_a, color='r')
    cover_plot(cover_b, color='b')
    plt.savefig(f"{log_directory}/cover.png")
    # plt.show()


def xy_cover(env, venv, reward_fun, nsteps, cover_x, cover_y, vis_coords):
    observer = VisObserver()

    ex_inits = []
    for item in cover_x:
        ex_inits.append({'x': item['x'], 'info': item['info'], 'g': item['x_feat']})

    goals = []
    for item in cover_y:
        goals.append(item['x_feat'])

    venv.reset(ex_inits)

    at_goal = [None] * venv.num_envs

    for n in range(nsteps):
        actions = [env.action_space.sample() for _ in range(venv.num_envs)]
        o, *_ = venv.step(actions)
        a_goals = o["achieved_goal"]
        observer.update([item[vis_coords] for item in goals],
                        [item[vis_coords] for item in a_goals], draw=True)

        for i, g in enumerate(goals):
            if at_goal[i] is not None:
                continue
            for j, ag in enumerate(a_goals):
                if reward_fun(ag_2=ag, g=g, info={}):
                    at_goal[i] = n
                    continue
    return at_goal


def none_init():
    return {'x': None, 'info': None, 'g': None}


def internal_radius(env, reward_fun, cover, nsamples, nsteps, vis_coords=None):
    _hit_time = [None] * nsamples
    for ns in range(nsamples):
        taus = [[] for _ in range(len(cover))]
        for k in range(len(cover)):
            info = cover[k]['info']
            if isinstance(info, collections.Mapping):
                info = Bunch(info)
            ex_init = {'x': cover[k]['x'], 'info': info, 'g': cover[k]['x_feat']}
            o = env.reset(ex_init=ex_init)
            for n in range(nsteps):
                taus[k].append(o["achieved_goal"])
                o, *_ = env.step(env.action_space.sample())

        hit = False
        for t in range(nsteps):
            if hit:
                break
            a_goals_t = [tau[t] for tau in taus]
            for pair in itertools.combinations(a_goals_t, 2):
                if reward_fun(ag_2=pair[0], g=pair[1], info={}):
                    _hit_time[ns] = t
                    hit = True
                    break

    hit_time = []
    for item in _hit_time:
        if item is None:
            hit_time.append(nsteps)
        else:
            hit_time.append(item)
    return hit_time


def _internal_radius(env, venv, reward_fun, cover, nsamples, nsteps, vis_coords=None):
    # observer = VisObserver()
    k = len(cover)
    ex_inits = [none_init()] * venv.num_envs
    for idx, item in enumerate(cover):
        info = item['info']
        if isinstance(info, collections.Mapping):
            info = Bunch(info)
        ex_inits[idx] = {'x': item['x'], 'info': info, 'g': item['x_feat']}

    _hit_time = [None] * nsamples
    for ns in range(nsamples):
        o = venv.reset(ex_inits)
        a_goals = o["achieved_goal"][:k]
        for n in range(nsteps):
            if _hit_time[ns] is not None:
                continue
            for pair in itertools.combinations(a_goals, 2):
                if reward_fun(ag_2=pair[0], g=pair[1], info={}):
                    _hit_time[ns] = n
                    break
            actions = [env.action_space.sample() for _ in range(venv.num_envs)]
            o, *_ = venv.step(actions)
            a_goals = o["achieved_goal"][:k]
            # observer.update([item[vis_coords] for item in a_goals], draw=True)
    hit_time = []
    for item in _hit_time:
        if item is None:
            hit_time.append(nsteps)
        else:
            hit_time.append(item)
    return hit_time


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    env = gym.make(args.env, **extra_args)

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    log_directory = extra_args["load"]
    directories = os.listdir(log_directory)
    k_vec = [int(item.split('K')[-1]) for item in directories if item[0]=='K']
    k_vec.sort()
    # k_vec = k_vec[:2]
    # args.num_env = 2
    # venv = build_env(args, extra_args=extra_args)
    nsteps = 100
    nsamples = 20
    random_radius = {}
    mca_radius = {}

    for k in k_vec:
        k_dirname = os.path.join(log_directory, f"K{k}")

        random_cover_fname = os.path.join(k_dirname, f"random/mca_cover/K{k}.json")
        random_cover = MetricDiversifier.load_model(random_cover_fname)
        mca_cover_fname = os.path.join(k_dirname, f"learned/mca_cover/K{k}.json")
        mca_cover = MetricDiversifier.load_model(mca_cover_fname)
        print(f"Measuring {k} lift")
        # random_radius[k] = _internal_radius(env, venv, reward_fun, random_cover, nsamples=nsamples, nsteps=nsteps)
        # mca_radius[k] = _internal_radius(env, venv, reward_fun, mca_cover, nsamples=nsamples, nsteps=nsteps)
        random_radius[k] = internal_radius(env, reward_fun, random_cover, nsamples=nsamples, nsteps=nsteps)
        mca_radius[k] = internal_radius(env, reward_fun, mca_cover, nsamples=nsamples, nsteps=nsteps)
    # venv.close()
    print(f"Done! plotting...")
    plot(random_radius, mca_radius, log_directory)


if __name__ == '__main__':
    main(sys.argv)

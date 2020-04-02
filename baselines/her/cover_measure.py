import argparse
import gym
from baselines.her.metric_diversification import MetricDiversifier
from baselines.her.metric_diversification import VisObserver
from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs
import os, sys
import itertools
import numpy as np
from matplotlib import pyplot as plt
import collections
from baselines.her.metric_diversification import Bunch
from baselines.common.misc_util import set_default_value


def reward_fun(env, ag_2, g, info, dist_th):  # vectorized
    return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info, distance_threshold=dist_th)


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


def internal_radius(env, cover, nsamples, nsteps, vis_coords=None):
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
                if reward_fun(env, ag_2=pair[0], g=pair[1], info={}):
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


def plot(results, log_directory):

    fig, ax = plt.subplots(1, 1)

    def cover_plot(cover, name):
        x = list(cover.keys())
        y = np.asarray([np.asarray(item).mean() for item in cover.values()])
        error = np.asarray([np.asarray(item).std() for item in cover.values()])

        # ax.plot(x, y, f"{color}-")
        ax.plot(x, y, label=name)
        ax.fill_between(x, y - error, y + error, alpha=0.5)

    for key, val in results.items():
        cover_plot(cover=val, name=key)
    ax.legend()
    plt.savefig(f"{log_directory}/lift.png")
    # plt.show()


def min_reach_time(env, cover, nsamples, nsteps, distance_th):
    _hit_time = [None] * nsamples
    hit_pairs = [(None, None)] * nsamples
    for ns in range(nsamples):
        time = None
        for k in range(len(cover)):
            info = cover[k]['info']
            if isinstance(info, collections.Mapping):
                info = Bunch(info)
            ex_init = {'x': cover[k]['x'], 'info': info, 'g': cover[k]['x_feat']}
            o = env.reset(ex_init=ex_init)
            k_hit = False
            for n in range(nsteps):
                if k_hit:
                    break
                if time is not None:
                    if n >= time:
                        break
                for j in range(len(cover)):
                    if j == k:
                        continue
                    if reward_fun(env, ag_2=o["achieved_goal"], g=cover[j]['x_feat'], info={}, dist_th=distance_th):
                        time = n
                        k_hit = True
                        hit_pairs[ns] = (k, j)
                        break
                o, *_ = env.step(env.action_space.sample())
        _hit_time[ns] = time

    hit_time = []
    for item in _hit_time:
        if item is None:
            hit_time.append(nsteps)
        else:
            hit_time.append(item)
    return np.asarray(hit_time), hit_pairs


def mean_reach_time(env, cover, nsamples, nsteps, distance_th):
    _hit_time = [None] * nsamples
    for ns in range(nsamples):
        hit_times = [None] * len(cover)
        for k in range(len(cover)):
            info = cover[k]['info']
            if isinstance(info, collections.Mapping):
                info = Bunch(info)
            ex_init = {'x': cover[k]['x'], 'info': info, 'g': cover[k]['x_feat']}
            o = env.reset(ex_init=ex_init)
            k_hit = False
            k_time = nsteps
            for n in range(nsteps):
                if k_hit:
                    break
                for j in range(len(cover)):
                    if j == k:
                        continue
                    if reward_fun(env, ag_2=o["achieved_goal"], g=cover[j]['x_feat'], info={}, dist_th=distance_th):
                        k_hit = True
                        k_time = n
                        break
                o, *_ = env.step(env.action_space.sample())
            hit_times[k] = k_time
        _hit_time[ns] = np.asarray(hit_times).mean()

    return np.asarray(_hit_time)


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    env = gym.make(args.env, **extra_args)

    log_directory = extra_args["load"]
    # methods = ["random", "learned"]
    methods = [""]
    nsteps = 300
    nsamples = 20
    distance_th = set_default_value(extra_args, 'cover_distance_threshold', None)

    results = {}
    for method in methods:
        results[method] = {}
        for k in range(0, 120, 1):
            # fname = f"{log_directory}/K{k}/{method}/mca_cover/K{k}.json"
            fname = f"{log_directory}/K{k}.json"
            cover = MetricDiversifier.load_model(fname)
            if cover is None:
                continue
            reach_time = mean_reach_time(env, cover, nsamples=nsamples, nsteps=nsteps, distance_th=distance_th)
            results[method][k] = reach_time
            print(f"{method}, k={k}, mean reach time = {np.asarray(reach_time).mean()}")
    plot(results, log_directory)


if __name__ == '__main__':
    main(sys.argv)

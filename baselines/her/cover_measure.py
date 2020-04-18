import gym
from baselines.her.metric_diversification import MetricDiversifier
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


def none_init():
    return {'x': None, 'info': None, 'g': None}


def plot(results, log_directory):

    fig, ax = plt.subplots(1, 1)

    def cover_plot(data, name):
        y = data["mean"]
        x = np.arange(len(y)) * data["xscale"]
        ax.plot(x, y, label=name)
        if "std" in data:
            error = data["std"]
            ax.fill_between(x, y - error, y + error, alpha=0.5)

    for key, val in results.items():
        cover_plot(data=val, name=key)
    ax.legend()
    plt.savefig(f"{log_directory}/lift.png")


def init_from_point(env, pnt):
    ex_init = {'x': pnt['x'],
               'qpos': pnt['qpos'],
               'qvel': pnt['qvel'],
               'g': pnt['x_feat']}
    return env.reset(ex_init=ex_init)


def xy_cover_single(env, cover_x, cover_y, nsteps, distance_th, self_cover, min_time, n_actions=30, vis=False):
    if len(cover_x) == 0:
        return True, 0

    idx = np.random.randint(len(cover_x))
    hit = False
    hit_time = nsteps
    for _ in range(n_actions):
        if hit:
            break
        o = init_from_point(env, cover_x[idx])
        if vis:
            env.render()
        a = env.action_space.sample()
        for n in range(nsteps):
            if n >= hit_time:
                break
            if n >= min_time:
                break
            if hit:
                break
            for j in range(len(cover_y)):
                if self_cover and j == idx:
                    continue
                if reward_fun(env, ag_2=o["achieved_goal"], g=cover_y[j]['x_feat'], info={}, dist_th=distance_th):
                    hit = True
                    hit_time = n
                    if vis:
                        print(f"difference: {np.linalg.norm(o['achieved_goal'] - cover_y[j]['x_feat'])}")
                        env.render()
                        init_from_point(env, cover_y[j])
                        env.render()
                    break
            o, *_ = env.step(a)
            if vis:
                env.render()
    return hit, hit_time


def xy_cover(env, cover_x, nsamples, nsteps, distance_th, cover_y=None):
    rates = []
    times = []
    if cover_y is None:
        cover_y = cover_x
        self_cover = True
    else:
        self_cover = False
    min_roam_time = nsteps
    for ns in range(nsamples):
        rate, roam_time = xy_cover_single(env, cover_x, cover_y, nsteps, distance_th, self_cover, min_roam_time)
        min_roam_time = np.minimum(min_roam_time, roam_time)
        rates.append(rate)
        times.append(roam_time)
    return np.asarray(rates), np.asarray(min_roam_time)


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
                env.render()
        _hit_time[ns] = time

    hit_time = []
    for item in _hit_time:
        if item is None:
            hit_time.append(nsteps)
        else:
            hit_time.append(item)
    return np.asarray(hit_time), hit_pairs


def mean_reach_time(env, cover, nsamples, nsteps, distance_th):
    sample_size = 10
    _hit_time = []
    for ns in range(nsamples):
        hit_times = [None] * sample_size
        sample_set = np.random.choice(list(range(len(cover))), sample_size, replace=False)
        for i, k in enumerate(sample_set):
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
            hit_times[i] = k_time
        _hit_time.append(np.asarray(hit_times).mean())

    return np.asarray(_hit_time)


def parse_log(logfile, field_name, normalize=False, scale=1):
    with open(logfile, "r") as fid:
        lines = fid.read().splitlines()
        values = np.asarray([float(line.split('|')[-2]) for line in lines if field_name in line])
        if normalize:
            values = (values - values.min())
            values = values / values.max()
            values *= scale
    return values


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    log_directory = extra_args["load"]

    results = dict()
    # results["hit_time"] = dict()
    # results["hit_time"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="test/hit_time_rate", normalize=True)
    # results["hit_time"]["xscale"] = 1
    #
    results["success"] = dict()
    results["success"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="test/success_rate", normalize=True, scale=10)
    results["success"]["xscale"] = 1

    results["hit time rate"] = dict()
    results["hit time rate"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="test/hit_time_rate", normalize=False)
    results["hit time rate"]["xscale"] = 1

    results["obs std"] = dict()
    results["obs std"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="stats_o/std", normalize=True, scale=10)
    results["obs std"]["xscale"] = 1

    for k in [100, 300, 500, 700]:
        fmean = f"k: {k}, RT mean"
        fstd = f"k: {k}, RT std"
        results[f"{k}"] = dict()
        results[f"{k}"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name=fmean, normalize=False)
        results[f"{k}"]["std"] = parse_log(f"{log_directory}/log.txt", field_name=fstd, normalize=False)
        results[f"{k}"]["xscale"] = 50
    plot(results, log_directory)


if __name__ == '__main__':
    main(sys.argv)

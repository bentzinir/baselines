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
    # print(f"saved figure to: {log_directory}/lift.png")
    # plt.show()


def init_from_point(env, pnt):
    ex_init = {'x': pnt['x'],
               'qpos': pnt['qpos'],
               'qvel': pnt['qvel'],
               'g': pnt['x_feat']}
    return env.reset(ex_init=ex_init)


def xy_cover_single(env, cover_x, cover_y, nsteps, distance_th, self_cover, n_actions=30, vis=False):
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
            z=1
        a = env.action_space.sample()
        for n in range(nsteps):
            if n >= hit_time:
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
                        z=1
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
    for ns in range(nsamples):
        rate, time = xy_cover_single(env, cover_x, cover_y, nsteps, distance_th, self_cover)
        rates.append(rate)
        times.append(time)
    return np.asarray(rates).mean(), np.asarray(times).mean()


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


def parse_log(logfile, field_name, skip_factor=20, normalize=False):
    with open(logfile, "r") as fid:
        lines = fid.read().splitlines()
        values = [float(line.split('|')[-2]) for line in lines if field_name in line]
        epochs = [float(line.split('|')[-2]) for line in lines if " epoch" in line]
        values = np.asarray(values)
        if normalize:
            values = (values - values.min())
            values = values / values.max()
        result = {}
        for epoch, val in zip(epochs[::skip_factor], values[::skip_factor]):
            result[epoch] = val
    return result


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    env = gym.make(args.env, **extra_args)

    log_directory = extra_args["load"]
    # methods = ["random", "learned"]
    methods = [""]
    nsteps = 20
    nsamples = 50
    distance_th = set_default_value(extra_args, 'cover_distance_threshold', None)

    results = {}
    # hit_time = parse_log(f"{log_directory}/log.txt", field_name="test/hit_time_rate", normalize=False)
    # success_rate = parse_log(f"{log_directory}/log.txt", field_name="test/success_rate")
    # results["test/hit_time_rate"] = {}
    # for key in hit_time.keys():
    #     results["test/hit_time_rate"][key] = hit_time[key] * success_rate[key]
    #
    # results["test/mean_Q"] = parse_log(f"{log_directory}/log.txt", field_name="test/mean_Q")
    # results["test/mean_Q_hit_rate"] = {}
    # for key, val in results["test/mean_Q"].items():
    #     results["test/mean_Q_hit_rate"][key] = nsteps - val

    for method in methods:
        for k in [100, 200, 300]:
            # results[f"hit rate {k}"] = {}
            results[f"roam time {k}"] = {}
        # fname = f"{log_directory}/mca_cover/epoch_{0}.json"
        # cover_y = MetricDiversifier.load_model(fname)
        for epoch in range(0, 5000, 1):
            for k in [100, 200, 300]:
                fname = f"{log_directory}/{k}/mca_cover/epoch_{epoch}.json"
                cover = MetricDiversifier.load_model(fname)
                if cover is None:
                    continue
                # m_time = mean_reach_time(env, cover, nsamples=nsamples, nsteps=nsteps, distance_th=distance_th)
                xy = xy_cover(env, cover, nsamples=nsamples, nsteps=nsteps, distance_th=distance_th)
                # yx = xy_cover(env, cover_y, nsamples=nsamples, nsteps=nsteps, distance_th=distance_th)
                # reach_time, _ = min_reach_time(env, cover, nsamples=nsamples, nsteps=nsteps, distance_th=distance_th)
                print(f"epoch={epoch}, k:{k}, hit rate / roam time = {xy}")
                # results[f"hit rate {k}"][epoch] = xy[0]
                results[f"roam time {k}"][epoch] = xy[1]
                plot(results, log_directory)


if __name__ == '__main__':
    main(sys.argv)

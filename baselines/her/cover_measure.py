from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs
import sys
import numpy as np
from matplotlib import pyplot as plt
from baselines.common.misc_util import set_default_value
import os
from pathlib import Path


def reward_fun(env, ag_2, g, info):  # vectorized
    return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)


def none_init():
    return {'x': None, 'info': None, 'g': None}


def plot(results, save_dir):

    fig, ax = plt.subplots(1, 1)

    def cover_plot(data, name):
        y = data["mean"]
        x = np.arange(0, len(y)) * data["xscale"]
        ax.plot(x, y, label=name)
        if "std" in data:
            error = data["std"]
            ax.fill_between(x, y - error, y + error, alpha=0.5)

    for key, val in results.items():
        cover_plot(data=val, name=val["name"])

    plt.title(f"Mean Hit Time", fontsize=20)
    plt.xlabel("Epochs", fontsize=20)
    plt.locator_params(nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    ax.set_facecolor('#ECE6E5')
    plt.grid(color='w', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/hit_time.png")
    plt.show()


def init_from_point(env, pnt):
    ex_init = {'o': pnt['o'],
               'qpos': pnt['qpos'],
               'qvel': pnt['qvel'],
               'g': pnt['ag']}
    return env.reset(ex_init=ex_init)


def xy_cover_single(env, cover_x, cover_y, nsteps, self_cover, n_actions=30, vis=False):
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
            if hit:
                break
            for j in range(len(cover_y)):
                if self_cover and j == idx:
                    continue
                if reward_fun(env, ag_2=o["achieved_goal"], g=cover_y[j]['x_feat'], info={}):
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


def xy_cover(env, cover_x, nsamples, nsteps, cover_y=None):
    rates = []
    times = []
    if cover_y is None:
        cover_y = cover_x
        self_cover = True
    else:
        self_cover = False
    for ns in range(nsamples):
        rate, roam_time = xy_cover_single(env, cover_x, cover_y, nsteps, self_cover)
        rates.append(rate)
        times.append(roam_time)
    return np.asarray(rates), np.asarray(times)


def min_reach_time(env, cover, nsamples, nsteps, distance_th):
    _hit_time = [None] * nsamples
    hit_pairs = [(None, None)] * nsamples
    for ns in range(nsamples):
        time = None
        for k in range(len(cover)):
            info = cover[k]['info']
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


def parse_horizon(logfile):
    with open(logfile, "r") as fid:
        lines = fid.read().splitlines()
        t_lines = [line for line in lines if 'T: ' in line]
    return float(t_lines[0].split(' ')[-1])


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def parse_log(logfile, field_name, normalize=False, dilute_fact=5, f=1):
    with open(logfile, "r") as fid:
        lines = fid.read().splitlines()
        values = np.asarray([float(line.split('|')[-2]) for line in lines if field_name in line])
        if normalize:
            values = (values - values.min())
            values = values / values.max()
        # if scale:
        #     values *= scale
        values = smooth(values, f)
    return values[::dilute_fact]


def extract_log_files(directory, patterns=[]):
    log_files = []
    for path in Path(directory).rglob('log.txt'):
        pattern_matches = [False for pattern in patterns if pattern not in str(path)]
        if not False in pattern_matches:
            log_files.append(str(path))
    return log_files


def method_to_log_pattern(method):
    if method == "plain":
        log_pattern = "alpha0/"
        legend_name = r'$\alpha =$' + f"{0}"
    elif method == "scrb":
        log_pattern = "alpha05/"
        legend_name = r'$\alpha =$' + f"{0.5}"
    else:
        log_pattern = ""
        legend_name = ""
    return log_pattern, legend_name


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    base_dir = extra_args["base_dir"]
    name = extra_args["name"]
    results = dict()

    save_dir = set_default_value(extra_args, 'save_dir', "/")

    std_type = 1
    d = 10
    f = 1
    trail = 1

    for method in ['scrb', 'plain']:
        log_pattern, legend_name = method_to_log_pattern(method)
        log_files = extract_log_files(base_dir, patterns=[name, log_pattern])
        values = []
        stds = []
        length = np.inf
        for logfile in log_files:
            value = parse_log(logfile, field_name="test/hit_time_mean", normalize=False, dilute_fact=d, f=f)[:-trail]
            std = parse_log(logfile, field_name="test/hit_time_std", normalize=False, dilute_fact=d, f=f)[:-trail]
            values.append(value)
            stds.append(std)
            if len(value) < length:
                length = len(value)
            # std = parse_log(f"{scrb_log_dir}/log.txt", field_name="test/hit_time_std", normalize=False, dilute_fact=d, f=f)[:-trail]

        values = [value[:length] for value in values]
        stds = [std[:length] for std in stds]

        if std_type == 1:
            standard_deviation = np.mean(stds, axis=0)
        else:
            standard_deviation = np.std(values, axis=0)
        results[method] = dict()
        results[method]["mean"] = np.mean(values, axis=0)
        results[method]["std"] = standard_deviation
        results[method]["xscale"] = d
        results[method]["name"] = legend_name

    plot(results, save_dir)

    # results["test success"] = dict()
    # results["test success"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="test/success_rate", normalize=False, scale=100)
    # results["test success"]["xscale"] = 1
    #
    # results["train success"] = dict()
    # results["train success"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="train/success_rate", normalize=False, scale=100)
    # results["train success"]["xscale"] = 1

    # results["train hit time rate"] = dict()
    # results["train hit time rate"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="train/hit_time_rate", normalize=True, scale=1)
    # results["train hit time rate"]["xscale"] = 1

    # results["Q"] = dict()
    # results["Q"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="test/mean_Q", normalize=False, scale=1)
    # results["Q"]["xscale"] = 1

    # results["goal std"] = dict()
    # results["goal std"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name="stats_g/std", normalize=False, scale=1)
    # results["goal std"]["xscale"] = 1


    # v = results["Q"]["mean"] + results["hit time rate"]["mean"]

    # horizon = parse_horizon(logfile=f"{log_directory}/log.txt")

    # q_rate = horizon - results["hit time rate"]["mean"]

    # q_gap = 100 * results["Q"]["mean"] / q_rate

    # results["q_gap"] = dict()
    # results["q_gap"]["mean"] = q_gap
    # results["q_gap"]["xscale"] = 1

    # for k in [100, 300, 500, 700]:
    #     fmean = f"k: {k}, RT mean"
    #     fstd = f"k: {k}, RT std"
    #     results[f"{k}"] = dict()
    #     results[f"{k}"]["mean"] = parse_log(f"{log_directory}/log.txt", field_name=fmean, normalize=False)
    #     results[f"{k}"]["std"] = parse_log(f"{log_directory}/log.txt", field_name=fstd, normalize=False)
    #     results[f"{k}"]["xscale"] = 50



if __name__ == '__main__':
    main(sys.argv)

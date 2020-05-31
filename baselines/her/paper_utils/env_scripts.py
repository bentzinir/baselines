import sys
from baselines.common.cmd_util import common_arg_parser
from baselines.run import parse_cmdline_kwargs
from baselines.her.metric_diversification import MetricDiversifier
from baselines.her.cover_measure import init_from_point
import gym, gym_maze
import numpy as np
import time
from baselines.her.paper_utils import utils as paper_utils
np.set_printoptions(precision=2)


def set_goal(env, scrb):
    if len(scrb.used_slots()) == 0:
        return env.reset()
    return env.set_goal(goal=scrb.draw(1)[0]['ag'])


def reset_env(env, scrb, mode='intrinsic'):
    if mode == 'intrinsic':
        return env.reset()
    elif mode == 'extrinsic':
        assert 'cover_path' is not None, 'missing cover path argument'
        pnt = scrb.draw(1)[0]
        if pnt is None:
            return env.reset()
        obs = init_from_point(env, pnt)
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
        time.sleep(.1)
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
        time.sleep(.1)
        if i % action_repetition == 0:
            a = np.clip(env.action_space.sample(), -clip_range, clip_range)
        o, r, d, info = env.step(a)
        if i % 1000 == 0:
            reset_env(env, cover=None, mode='intrinsic')
            print(f"Reset")
            i = 0
    env.close()


def play_policy(env, env_id, T=20, load_path=None, cover_path=None, semi_metric=False, eps_greedy=False, **kwargs):
    policy, reward_fun = paper_utils.load_policy(env_id, load_path, **kwargs)
    cover = MetricDiversifier.load_model(cover_path)
    obs = reset_env(env, cover, mode='intrinsic')
    i = 0
    while True:
        i += 1
        env.render()
        time.sleep(.01)
        action, _, state, _ = policy.step(obs)
        if eps_greedy and i % 5 == 0:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        success = info['is_success']
        timeout = i % T == 0
        done = success or timeout
        if done:
            # input(f"success: {success}, invalid: {invalid}, timeout: {timeout}")
            if cover is None or semi_metric:
                reset_env(env, cover, mode='intrinsic')
            else:
                reset_env(env, cover, mode='extrinsic')
            obs = set_goal(env, cover)

            i = 0
    env.close()


def exp1_loop(env, scrb, policy, eps_greedy, T, semi_metric, nsteps):

    obs = reset_env(env, scrb, mode='intrinsic')

    while len(scrb.open_slots()) > 0:
        pnt = scrb.init_record(o=obs['observation'].copy())
        scrb.load_new_point(pnt, d_func=policy.get_actions)
        assert not scrb.dilute_overlaps

    reached_goal = False
    t = 0
    counter = 0
    times = []
    radii = []
    while counter < nsteps:
        # 1. environment step
        action, _, state, _ = policy.step(obs)
        if reached_goal or (eps_greedy and t % 10 == 0):
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        success = info['is_success']
        reached_goal = reached_goal or success

        # 2. GPI update
        pnt = scrb.init_record(o=obs['observation'].copy())
        scrb.load_new_point(pnt, d_func=policy.get_actions)

        r_pack = env._max_episode_steps + scrb.M.min()
        times.append(counter)
        radii.append(r_pack)

        if counter % 1000 == 0:
            ...
            # scrb.save(message=counter)
            # print(f"counter: {counter}, cover size: {scrb.current_size}, packing radius: {r_pack}")

        # TODO: add back after debug
        # scrb.age += 1

        # 3. measure packing radius
        ...

        # 4. episodic reset
        if t % T == 0:
            t = 0
            reached_goal = False
            if semi_metric:
                reset_env(env, scrb, mode='intrinsic')
            else:
                reset_env(env, scrb, mode='extrinsic')
            obs = set_goal(env, scrb)
        counter += 1
        t += 1

    return times, radii


def experiment1(env, env_id, T=100, k=50, load_path=None, save_path=None, semi_metric=False, eps_greedy=False,
                dilute_overlaps=True, ntrials=5, nsteps=10000, random_mode=False, **kwargs):

    policy, reward_fun = paper_utils.load_policy(env_id, load_path, **kwargs)
    if semi_metric:
        metric_str = "semi_metric"
    else:
        metric_str = "full_metric"

    for random_mode in [True, False]:
        if random_mode:
            random_str = 'random'
            alpha=0
        else:
            random_str = 'scrb'
            alpha=0.5

        log_path = f"{save_path}/{metric_str}_{random_str}"

        results = dict()
        k_vec = [10, 20, 30, 40, 50]
        # k_vec = [50]
        for k in k_vec:
            results[k] = dict()
            k_radii = []
            for trial_idx in range(ntrials):
                scrb = MetricDiversifier(k=k, vis=False, dilute_overlaps=dilute_overlaps, vis_coords=[0, 1], save_path=log_path,
                                         reward_func=reward_fun, random_mode=random_mode)
                times, radii = exp1_loop(env, scrb, policy, eps_greedy, T, semi_metric, nsteps)
                k_radii.append(radii)
                print(f"k: {k}, trial: {trial_idx}/{ntrials}, nsteps: {nsteps}")
            results[k]["mean"] = np.asarray(k_radii).mean(axis=0)
            results[k]["std"] = np.asarray(k_radii).std(axis=0)
            results[k]["time"] = times

            paper_utils.results_to_figure(results, save_directory=log_path, alpha=alpha, message=f"{metric_str}_{random_str}")

        exp1_loop(env, scrb, policy, eps_greedy, T, semi_metric, 50)
        paper_utils.overlayed_figure(env, scrb, save_directory=log_path, message=f"{metric_str}_{random_str}")


def experiment2(env, env_id, T=100, models_path=None, save_path=None, eps_greedy=False, ntrials=5, ngoals=100, random_mode=False, vis=False, **kwargs):

    goals = [env.env.draw_goal() for _ in range(ngoals)]
    reached = np.zeros(len(goals))
    recall_at_epoch = []
    epochs = paper_utils.list_epochs(models_path)
    for epoch_idx in epochs:
        policy, reward_fun = paper_utils.load_policy(env_id, load_path=f"{models_path}/epoch_{epoch_idx}.model", **kwargs)
        for gidx, goal in enumerate(goals):
            if reached[gidx]:
                continue
            reset_env(env, scrb=None, mode='intrinsic')
            env.env.set_goal(goal=goal)
            for t in range(T):
                if reached[gidx]:
                    break
                if vis:
                    env.render()
                time.sleep(.01)
                action, _, state, _ = policy.step(obs)
                if eps_greedy and t % 5 == 0:
                    action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if info['is_success']:
                    reached[gidx] = 1
        recall_at_epoch.append(reached.mean())


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
        play_policy(env=environment, env_id=args.env, **extra_args)
    elif extra_args['option'] == 'experiment1':
        assert extra_args['load_path'] is not None, 'load path is none'
        assert args.save_path is not None, 'save path is none'
        experiment1(env=environment, env_id=args.env, save_path=args.save_path, **extra_args)
    elif extra_args['option'] == 'experiment2':
        assert extra_args['models_path'] is not None, 'models path is none'
        assert args.save_path is not None, 'save path is none'
        experiment2(env=environment, env_id=args.env, save_path=args.save_path, **extra_args)
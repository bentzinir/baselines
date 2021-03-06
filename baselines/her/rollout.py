from collections import deque

import numpy as np
import pickle

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, active, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False,
                 exploration='eps_greedy', compute_root_Q=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.hit_time_mean_history = deque(maxlen=history_len)
        self.hit_time_std_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self, ex_init=None, record=False):
        if not self.active:
            return

        if ex_init is None:
            ex_init = [{'o': None, 'qpos': None, 'qvel': None, 'g': None} for _ in range(self.venv.num_envs)]

        self.obs_dict = self.venv.reset(ex_init, record)
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

        self.initial_qpos = self.obs_dict['qpos']
        self.initial_qvel = self.obs_dict['qvel']

    def generate_rollouts(self, ex_init=None, record=False, random=False, log_hit_time=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        if not self.active:
            return
        self.reset_all_rollouts(ex_init, record=record)

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        qpos = np.empty((self.rollout_batch_size, self.dims['qpos']), np.float32)
        qvel = np.empty((self.rollout_batch_size, self.dims['qvel']), np.float32)

        qpos[:] = self.initial_qpos
        qvel[:] = self.initial_qvel

        num_envs = self.venv.num_envs

        random_action = self.policy._random_action(num_envs)

        reached_goal = [False] * num_envs
        hit_time = [None] * num_envs

        if random:
            self.exploration = 'random'
        else:
            self.exploration = 'eps_greedy'  # 'go'

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs, qposes, qvels, hit_times = [], [], [], []

        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net,
                exploration=self.exploration,
                go=np.logical_not(reached_goal),
                random_action=random_action,
            )

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            # compute new states and observations
            obs_dict_new, _, done, info = self.venv.step(u)
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']

            qpos_new = obs_dict_new['qpos']
            qvel_new = obs_dict_new['qvel']

            success = np.array([i.get('is_success', 0.0) for i in info])

            for e_idx, (suc, ht) in enumerate(zip(success, hit_time)):
                if suc and hit_time[e_idx] is None:
                    hit_time[e_idx] = t

            reached_goal = [hit is not None for hit in hit_time]

            if any(done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            qposes.append(qpos.copy())
            qvels.append(qvel.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
            qpos[...] = qpos_new
            qvel[...] = qvel_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        qposes.append(qpos.copy())
        qvels.append(qvel.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals,
                       qpos=qposes,
                       qvel=qvels,
                       # t=Ts
                       )

        if self.compute_Q:
            episode["Qs"] = Qs

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        if self.exploration != 'random':
            if self.exploration in ['go_explore', 'go']:
                successful = np.asarray([1 if hit is not None else 0 for hit in hit_time])
            elif self.exploration in ['eps_greedy']:
                successful = np.array(successes)[-1, :]
            assert successful.shape == (self.rollout_batch_size,)
            success_rate = np.mean(successful)
            self.success_history.append(success_rate)

            hit_times = np.asarray([hit if hit is not None else 0 for hit in hit_time])
            if log_hit_time:
                hit_time_mean = np.mean(hit_times)
                hit_time_std = np.std(hit_times)
                self.hit_time_mean_history.append(hit_time_mean)
                self.hit_time_std_history.append(hit_time_std)

        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        if not self.active:
            return
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.hit_time_mean_history.clear()
        self.hit_time_std_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        if not self.active:
            return
        return np.mean(self.success_history)

    def current_hit_time_mean_rate(self):
        if not self.active:
            return
        return np.mean(self.hit_time_mean_history)

    def current_hit_time_std_rate(self):
        if not self.active:
            return
        return np.mean(self.hit_time_std_history)

    def current_mean_Q(self):
        if not self.active:
            return
        return np.mean(self.Q_history)

    def save_policy(self, path):
        if not self.active:
            return
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        if not self.active:
            return
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('hit_time_mean', np.mean(self.hit_time_mean_history))]
        logs += [('hit_time_std', np.std(self.hit_time_std_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs


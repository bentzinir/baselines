import copy
import numpy as np
from baselines.her.visualizer import VisObserver


class MCA:
    def __init__(self, policy, rollout_worker, evaluator, state_model, coord_dict, #n_samples=30,
                 ss=False, sharing=False, vis_obs=False):
        self.policy = policy
        self.rollout_worker = rollout_worker
        self.evaluator = evaluator
        self.state_model = state_model
        self.best_success_rate = -1
        self.ss = ss
        self.sharing = sharing
        self.ex_experience = None
        self.coord_dict = coord_dict
        self.tmp_point = state_model.init_record()
        if vis_obs:
            self.visualizer = VisObserver()

    # def _buffer_sample(self, n, **kwargs):
    #     if self.policy.buffer.current_size == 0:
    #         return
    #     inits = []
    #     while len(inits) < n:
    #         pnts = self.policy.buffer.sample(2)
    #         if np.any(pnts['info_valid'] == 0):
    #             continue
    #         inits.append({'x': pnts['o'][0],
    #                       'qpos': pnts['qpos'][0],
    #                       'qvel': pnts['qvel'][0],
    #                       'g': pnts['ag'][1]})
    #     return inits

    @staticmethod
    def sample_2_dict(x):
        z = dict()
        for key in x[0].keys():
            z[key] = np.asarray([pnt[key][0] for pnt in x])
        return z

    def sample_from_buffer(self, n, valids_only=True):
        if not valids_only:
            return self.policy.buffer.sample(n)
        else:
            pnts = []
            while len(pnts) < n:
                pnt = self.policy.buffer.sample(1)
                if pnt['info_valid']:
                    pnts.append(pnt)
            return self.sample_2_dict(pnts)

    def init_from_buffer(self, n):
        if self.policy.buffer.current_size == 0:
            return
        batch = self.sample_from_buffer(n, valids_only=True)
        p = np.random.permutation(n)
        goals = [batch['ag'][pidx] for pidx in p]
        inits = []
        for o, ag, qpos, qvel, g in zip(batch['o'], batch['ag'], batch['qpos'], batch['qvel'], goals):
            inits.append({'x': o,
                          'qpos': qpos,
                          'qvel': qvel,
                          'g': g})
        return inits

    def update_metric_model(self):
        if self.policy.buffer.current_size == 0:
            return
        # batch = self.policy.buffer.sample(100)
        batch = self.sample_from_buffer(100, valids_only=True)
        for o, ag, qpos, qvel in zip(batch['o'], batch['ag'], batch['qpos'], batch['qvel']):
            new_point = self.state_model.init_record(x=o, x_feat=ag, qpos=qpos, qvel=qvel)
            self.state_model.load_new_point(new_point, d_func=self.policy.get_actions)

    def load_episode(self, episode):
        if episode is None:
            return
        obs = episode['o']
        if self.ss:
            agoals = episode['o']
        else:
            agoals = episode['ag']
        if 's_info' in episode:
            _, nsteps, _ = np.asarray(episode['s_info']).shape
            obs = obs[:, :nsteps, :]
            agoals = agoals[:, :nsteps, :]
            obs = np.reshape(obs, (-1, obs.shape[-1]))
            ags = np.reshape(agoals, (-1, agoals.shape[-1]))
            infos = [j for sub in episode['s_info'] for j in sub]
        else:
            obs = np.reshape(obs, (-1, obs.shape[-1]))
            ags = np.reshape(agoals, (-1, agoals.shape[-1]))
            infos = np.full(obs.shape, None)

        p = np.random.permutation(len(obs))
        # if len(p) > self.n_samples:
        #     p = p[:self.n_samples]
        obs = obs[p]
        ags = ags[p]
        infos = [infos[p_idx] for p_idx in p]

        for ob, ag, info in zip(obs, ags, infos):
            if hasattr(info, '_asdict'):
                info = info._asdict()
            new_point = self.state_model.init_record(x=ob, x_feat=ag, info=info)
            if self.tmp_point['x'] is None:
                self.tmp_point = new_point
            self.state_model.load_new_point(new_point, d_func=self.policy.get_actions)

        if 's_info' in episode:
            del episode['s_info']

    def store_ex_episode(self, episode):
        if episode is None:
            return
        if 's_info' in episode:
            del episode['s_info']
        if self.sharing:
            self.ex_experience = copy.deepcopy(episode)
        else:
            self.ex_experience = episode

    def overload_sg(self, episode, mca_episode):
        if episode is None or mca_episode is None:
            return episode
        for key, val in mca_episode.items():
            if key in episode:
                if key == 'g' or key == 'ag':
                    val = val[..., self.coord_dict["g"]]
                episode[key] = np.concatenate([episode[key], val], axis=0)
        return episode

    def overload_ss(self, mca_episode):
        if not self.sharing:
            return mca_episode
        if mca_episode is None:
            return None
        for key, val in self.ex_experience.items():
            if key == 'g' or key == 'ag':
                continue
            mca_episode[key] = np.concatenate([mca_episode[key], val], axis=0)

        # achieved goal is simply the current state
        mca_episode['ag'] = np.concatenate([mca_episode['ag'], self.ex_experience['o']], axis=0)

        # extract goal from a random state in the set of trajectories
        goals = np.reshape(self.ex_experience['o'][:, :-1, :], [-1, self.ex_experience['o'].shape[-1]])
        np.random.shuffle(goals)
        mca_episode['g'] = np.concatenate([mca_episode['g'], np.reshape(goals, self.ex_experience['o'][:, :-1, :].shape)], axis=0)

        # remove Q and root Q from mca_episode dictionary because it is no longer needed and it is not augmented,
        # therefore has inadequate shape.
        del mca_episode['Qs']
        del mca_episode['root_Qs']

        return mca_episode

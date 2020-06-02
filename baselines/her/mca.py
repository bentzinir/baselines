import numpy as np
import random


class MCA:
    def __init__(self, policy, semi_metric, rollout_worker, evaluator, state_model, coord_dict, ss=False):
        self.policy = policy
        self.semi_metric = semi_metric
        self.ncells = len(state_model)
        self.rollout_worker = rollout_worker
        self.evaluator = evaluator
        self.state_model = state_model
        self.best_success_rate = -1
        self.ss = ss
        self.ex_experience = None
        self.coord_dict = coord_dict
        # self.tmp_point = state_model.init_record()

    @staticmethod
    def sample_2_dict(x):
        z = dict()
        for key in x[0].keys():
            z[key] = np.asarray([pnt[key][0] for pnt in x])
        return z

    def sample_from_buffer(self, n, valids_only=True):
        if self.policy.buffer.current_size == 0:
            return
        if not valids_only:
            return self.policy.buffer.sample_regular(n)
        else:
            pnts = []
            counter = 0
            while len(pnts) < n:
                counter += 1
                pnt = self.policy.buffer.sample_regular(1)
                if pnt['info_valid']:
                    pnts.append(pnt)
                if counter > 100 * n:
                    print(f"Failed to sample valid points from buffer")
                    return None
            return self.sample_2_dict(pnts)

    def init_from_buffer(self, n):
        batch = self.sample_from_buffer(n, valids_only=True)
        if batch is None:
            return
        inits = []
        for o, ag, qpos, qvel, ag in zip(batch['o'], batch['ag'], batch['qpos'], batch['qvel'], batch['ag']):
            inits.append({'o': o.copy(), 'ag': ag.copy(), 'qpos': qpos.copy(), 'qvel': qvel.copy()})
        return inits

    def draw_init(self, n, alpha=0.5):
        # with probability alpha sample from the state model. Otherwise, sample from the replay buffer
        if np.random.binomial(n=1, p=alpha):
            valid_cells = [midx for midx, m in enumerate(self.state_model) if m.current_size > 0]
            if len(valid_cells) == 0:
                return
            inits = self.state_model[random.choice(valid_cells)].draw(n)
            ginits = self.state_model[random.choice(valid_cells)].draw(n)
        else:
            inits = self.init_from_buffer(n)
            ginits = self.init_from_buffer(n)
        if inits is None or ginits is None:
            return
        # stitch init and goal together
        for idx, init in enumerate(inits):
            init["g"] = ginits[idx]["ag"].copy()
            if self.semi_metric:
                for key in init.keys():
                    if key != 'g':
                        init[key] = None
        return inits

    def calculate_cell_idx(self, ags):
        n = len(ags)
        root_o_mat = np.repeat(np.expand_dims(self.rollout_worker.initial_o[0], 0), repeats=n, axis=0)
        root_ag_mat = np.repeat(np.expand_dims(self.rollout_worker.initial_ag[0], 0), repeats=n, axis=0)
        _, Qs = self.policy.get_actions(o=root_o_mat, ag=root_ag_mat, g=ags, compute_Q=True, use_target_net=True)
        return self.q2cellidx(Qs.squeeze(axis=1))

    def refresh_cells(self, n):
        if not self.semi_metric:
            print(f'state model size {self.state_model[0].current_size}')
            return
        for cidx in range(self.ncells):
            if self.state_model[cidx].current_size == 0:
                print(f"Cell {cidx} size is {self.state_model[cidx].current_size}. Skipping refreshment")
                continue
            n = np.minimum(n, self.state_model[cidx].current_size)
            pts = self.state_model[cidx].draw(n, replace=False)
            idxs_in_buffer = [pnt['idx_in_buffer'] for pnt in pts]

            assert len([True for pnt in pts if pnt['ag'] is None]) == 0

            ags = np.asarray([pnt['ag'] for pnt in pts])
            current_cidxs = self.calculate_cell_idx(ags=ags)
            nrefreshes = 0
            for current_idx, idx_in_buffer in zip(current_cidxs, idxs_in_buffer):
                if current_idx != cidx and (idx_in_buffer in self.state_model[cidx].used_slots()):
                    self.state_model[cidx].invalidate_idx(idx_in_buffer)
                    nrefreshes += 1
            print(f'Cell {cidx} size is {self.state_model[cidx].current_size}. #refreshments {nrefreshes}')

    def update_metric_model(self, n):
        batch = self.sample_from_buffer(n, valids_only=True)
        if batch is None:
            return
        # in semi metric mode we decide of cell idx by calculating distance from root
        if self.semi_metric:
            cidxs = self.calculate_cell_idx(ags=batch['ag'])
            state_model_dfunc = None
        else:
            cidxs = np.zeros(len(batch['o']), dtype=np.int)
            state_model_dfunc = self.policy.get_actions

        assert len(cidxs) == len(batch['o'])
        assert cidxs.min() >= 0
        assert cidxs.max() < self.rollout_worker.T
        counter, nupdates = 0, 0
        for o, ag, qpos, qvel, cidx in zip(batch['o'], batch['ag'], batch['qpos'], batch['qvel'], cidxs):
            assert ag is not None
            new_point = self.state_model[int(cidx)].init_record(o=o.copy(), ag=ag.copy(), qpos=qpos.copy(), qvel=qvel.copy())
            updated = self.state_model[cidx].load_new_point(new_point, d_func=state_model_dfunc)
            nupdates += updated
            counter += 1
            print(f"\r>> Updating metric model {counter}/{int(n)}", end='')
        print(f' ... Done!, # of updates: {nupdates}')

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

    def q2cellidx(self, q_vals):
        return np.clip(q_vals, 1e-4, self.rollout_worker.T - 1e-4).astype(np.int)

    def update_age(self):
        for state_model in self.state_model:
            state_model.age += 1

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import json
import random


class VisObserver:
    def __init__(self, ndim=2):
        self.ndim = ndim
        fig = plt.figure()
        if ndim == 2:
            ax = fig.add_axes([0, 0, 1, 1])
            self.scat = ax.scatter(x=[], y=[], c='r', marker='+')
        elif ndim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = Axes3D(fig)
            self.scat_ax = ax
            self.scat = ax.scatter(xs=[], ys=[], c='r', marker='+')
        plt.tight_layout()

    def update(self, points, draw=False):
        if self.ndim == 2:
            self.scat.set_offsets(points)
        elif self.ndim == 3:
            pts = np.asarray(points)
            self.scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            self.scat_ax.autoscale()
            self.scat_ax.set_xlim([pts[:, 0].min(), pts[:, 0].max()])
            self.scat_ax.set_ylim([pts[:, 1].min(), pts[:, 1].max()])
            self.scat_ax.set_zlim([pts[:, 2].min(), pts[:, 2].max()])
        if draw:
            plt.draw()
            plt.pause(0.0001)

    def save(self, save_path, message, verbose=False):
        if not os.path.exists(os.path.join(save_path, "mca_figures")):
            os.makedirs(os.path.join(save_path, "mca_figures"))
        fig_name = os.path.join(save_path, "mca_figures", f"{message}.png")
        plt.savefig(fig_name)
        if verbose:
            print(f"saving figure: {fig_name}")


class MetricDiversifier:
    def __init__(self, k, reward_func, load_p=1, vis=False, dilute_overlaps=True, vis_coords=None, load_model=None, save_path=None, feature_w=None, random_mode=False, **kwargs):
        self.init_buffers(k)
        self.feature_w = feature_w
        self.reward_func = reward_func
        self.save_path = save_path
        self.dilute_overlaps = dilute_overlaps
        self.random_mode = random_mode
        self.load_p = load_p
        self.vis = vis
        self.vis_coords = vis_coords
        self.counter = 0
        self.observer = None
        if load_model is not None:
            self.load_model(load_model)
            print(f"Loaded cover: {load_model}, model size: {self.current_size}")

    def init_buffers(self, k):
        self.k = k
        self.k_approx = k  # k // 10
        self.M = -np.inf * np.ones((self.k, self.k))
        self.age = np.inf * np.ones(self.k)
        self.buffer = [None for _ in range(self.k)]
        [self.invalidate_idx(idx) for idx in range(self.k)]

    def _buffer_2_array(self, val, idxs):
        array = np.asarray([self.buffer[idx][val] for idx in idxs])
        return array

    def _buffer_2_dict(self):
        z = {}
        i = 0
        for b in self.buffer:
            if b is None:
                continue
            z[i] = {}
            for key, val in b.items():
                if type(val) == np.ndarray:
                    val = val.tolist()
                elif val == np.inf or val == -np.inf:
                    val = None
                elif isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                z[i][key] = val
            i += 1
        return z

    def quasimetric(self, x1_o, x1_ag, x2_o, x2_ag, d_func=None, feat_distance=True):
        '''
        :param x1_o: set of points
        :param x2_o: set of points
        :param d_func: a quasimetric distance function
        :param feat_distance: decide whether distance is measured on features or on entire state
        :return: the point in the set that is closest to x
        '''
        if d_func is None:
            if feat_distance:
                if self.feature_w is None:
                    x1x2_distance = np.linalg.norm(x1_ag - x2_ag, ord=2, axis=1)
                else:
                    distance_mat = ((x1_ag - x2_ag) ** 2)
                    weight_mat = np.repeat(np.expand_dims(self.feature_w, 0), repeats=distance_mat.shape[0], axis=0)
                    x1x2_distance = (weight_mat * distance_mat).sum(axis=1)
            else:
                x1x2_distance = np.linalg.norm(x1_o - x2_o, ord=2, axis=1)
        else:
            _, Q = d_func(o=x1_o, ag=x1_ag, g=x2_ag, compute_Q=True, use_target_net=True)
            if Q.ndim == 2:
                Q = Q.squeeze()
            x1x2_distance = - Q
        return x1x2_distance

    def prepare_dfunc_inputs(self, pnt, set_idxs=None):
        if set_idxs is None:
            set_idxs = self.used_slots()
        set_o_mat = self._buffer_2_array(val='o', idxs=set_idxs)
        set_ag_mat = self._buffer_2_array(val='ag', idxs=set_idxs)
        pnt_o_mat = np.repeat(np.expand_dims(pnt['o'], 0), repeats=len(set_idxs), axis=0)
        pnt_ag_mat = np.repeat(np.expand_dims(pnt['ag'], 0), repeats=len(set_idxs), axis=0)
        if len(self.used_slots()) == 0:
            set_o_mat = np.empty_like(pnt_o_mat)
            set_ag_mat = np.empty_like(pnt_ag_mat)
        return set_o_mat, set_ag_mat, pnt_o_mat, pnt_ag_mat

    def refresh_entry(self, idx, d_func):
        rpnt = self.buffer[idx]
        set_o_mat, set_ag_mat, pnt_o_mat, pnt_ag_mat = self.prepare_dfunc_inputs(rpnt)

        # update "from" distances pnt -> set
        self.M[idx] = self.quasimetric(x1_o=pnt_o_mat, x1_ag=pnt_ag_mat, x2_o=set_o_mat, x2_ag=set_ag_mat,
                                       d_func=d_func)

        # update "to" distances set -> pnt
        self.M[:, idx] = self.quasimetric(x1_o=set_o_mat, x1_ag=set_ag_mat, x2_o=pnt_o_mat, x2_ag=pnt_ag_mat,
                                          d_func=d_func)

        # update self distance to +inf to exclude from downstream calculation
        self.M[idx, idx] = np.inf

        self.age[idx] = 0

    def adjust_set(self, new_pnt, d_func):

        for idx in self.used_slots():
            pnt = self.buffer[idx]
            if pnt['ag'] is None:
                assert False

        nrefs = np.minimum(len(self.used_slots()), self.k_approx)

        ref_idx_set = np.random.choice(self.used_slots(), nrefs, replace=False)

        # refresh outdated matrix entries
        for idx in ref_idx_set:
            if self.age[idx] > 10:
                self.refresh_entry(idx, d_func)

        set_o_mat, set_ag_mat, newpnt_o_mat, newpnt_ag_mat = self.prepare_dfunc_inputs(new_pnt)

        distances_to_new_pnt = self.quasimetric(x1_o=set_o_mat, x1_ag=set_ag_mat, x2_o=newpnt_o_mat, x2_ag=newpnt_ag_mat, d_func=d_func)

        ####################################
        # Greedy Packing Improvement (GPI) #
        ####################################
        b_idx = -1
        b_delta = -np.inf
        for j in ref_idx_set:
            j_distances = distances_to_new_pnt.copy()
            j_distances[j] = np.inf
            delta_j = j_distances.min() - self.M[:, j].min()
            if delta_j > b_delta and delta_j > 0:
                b_delta = delta_j
                b_idx = j
        ##################################

        # Random mode
        if self.random_mode:
            b_idx = random.choice(self.used_slots())

        # Early escape if no update is needed
        if b_idx == -1:
            return False

        # Do not take new point if it overlaps any set point (besides b_idx)
        idxs = self.used_slots()
        idxs.remove(b_idx)
        if self.pnt_set_overlap(new_pnt, idxs=idxs):
            b_idx = -1

        if b_idx >= 0:

            distances_from_newpnt = self.quasimetric(x1_o=newpnt_o_mat, x1_ag=newpnt_ag_mat, x2_o=set_o_mat, x2_ag=set_ag_mat, d_func=d_func)

            self.occupy_idx(new_pnt, b_idx, ref_idxs=self.used_slots(), distances_to_newpnt=distances_to_new_pnt, distances_from_newpnt=distances_from_newpnt)

        return b_idx >= 0

    def occupy_idx(self, new_pnt, insert_idx, ref_idxs, distances_to_newpnt=None, distances_from_newpnt=None):

        self.M[insert_idx, ref_idxs] = distances_from_newpnt

        self.M[ref_idxs, insert_idx] = distances_to_newpnt

        self.M[insert_idx, insert_idx] = np.inf

        # replace in buffer
        self.buffer[insert_idx] = new_pnt

        self.age[insert_idx] = 0

    def invalidate_idx(self, idx):
        self.M[idx] = -np.inf
        self.M[:, idx] = -np.inf
        self.M[idx, idx] = np.inf
        self.buffer[idx] = None
        self.age[idx] = np.inf

    def pnt_set_overlap(self, pnt, idxs=None):
        if not self.dilute_overlaps:
            return False
        if idxs is None:
            idxs = self.used_slots()
        for refidx in idxs:
            if self.reward_func(ag_2=pnt['ag'], g=self.buffer[refidx]['ag'], info={}):
                return True
        return False

    def load_new_point(self, new_pnt, d_func=None):
        if new_pnt is None:
            return False

        # Not fully occupied
        if len(self.open_slots()) > 0:
            # check for overlaps. If ok - load
            if self.pnt_set_overlap(new_pnt):
                return False
            else:
                set_o_mat, set_ag_mat, newpnt_o_mat, newpnt_ag_mat = self.prepare_dfunc_inputs(new_pnt)
                distances_from = self.quasimetric(x1_o=newpnt_o_mat, x1_ag=newpnt_ag_mat, x2_o=set_o_mat,
                                                  x2_ag=set_ag_mat, d_func=d_func)
                distances_to = self.quasimetric(x1_o=set_o_mat, x1_ag=set_ag_mat, x2_o=newpnt_o_mat,
                                                x2_ag=newpnt_ag_mat, d_func=d_func)
                self.occupy_idx(new_pnt, insert_idx=random.choice(self.open_slots()), ref_idxs=self.used_slots(),
                                distances_from_newpnt=distances_from, distances_to_newpnt=distances_to)
                return True

        # Fully occupied
        else:
            return self.adjust_set(new_pnt, d_func)

    def draw(self, n, replace=True):
        if self.current_size == 0:
            return None
        s_idxs = np.random.choice(self.used_slots(), n, replace=replace)
        batch = []
        for s_idx in s_idxs:
            record = {key: self.buffer[s_idx][key].copy() for key in self.buffer[s_idx].keys()}
            record['idx_in_buffer'] = s_idx
            batch.append(record)
        return batch

    def _update_figure(self):
        pts = [self.buffer[idx]['o'][self.vis_coords] for idx in self.used_slots()]
        if len(pts) == 0:
            return
        self.observer.update(pts)

    def visualize(self):
        if self.vis and self.vis_coords is not None:
            if self.observer is None:
                self.observer = VisObserver(len(self.vis_coords))
            self._update_figure()

    def save(self, save_path=None, message=None):
        if save_path is None:
            save_path = self.save_path
        if message is None:
            message = f"K{self.current_size}"
        # save visualization
        self.visualize()
        if self.observer is not None:
            self.observer.save(save_path, message)
        # save cover model
        self.save_model(save_path, message)

    def open_slots(self):
        return [idx for idx, b in enumerate(self.buffer) if b is None]

    def used_slots(self):
        return [idx for idx, b in enumerate(self.buffer) if b is not None]

    @property
    def current_size(self):
        return len([True for b in self.buffer if b is not None])
        # return len(self.buffer)

    @staticmethod
    def init_record(o=None, ag=None, qpos=None, qvel=None):
        if ag is None:
            ag = o
        if qpos is None:
            qpos = np.empty(1)
        if qvel is None:
            qvel = np.empty(1)
        return {'o': np.asarray(o), 'ag': np.asarray(ag), 'qpos': qpos, 'qvel': qvel}

    def save_model(self, save_path, message=None, verbose=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if message is None:
            message = 'cover'
        f_name = f"{save_path}/{message}.json"
        with open(f_name, 'w') as outfile:
            json_str = json.dumps(self._buffer_2_dict(), indent=4, sort_keys=True)
            outfile.write(json_str)
        if verbose:
            print(f"saving cover: {f_name}, (size:{self.current_size})")

    def load_model(self, load_path, d_func=None):
        if load_path is None:
            return
        if not os.path.exists(load_path):
            return None
        with open(load_path, 'r') as infile:
            json_buffer = json.load(infile)
            k = len(json_buffer)
            self.init_buffers(k)
        for key, val in json_buffer.items():
            new_pnt = self.init_record(**val)
            idx = random.choice(self.open_slots())
            self.occupy_idx(new_pnt, idx, ref_idxs=[], distances_to_newpnt=None, distances_from_newpnt=None)
        for idx in self.used_slots():
            self.refresh_entry(idx, d_func)


if __name__ == '__main__':

    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(ag_2, g, info={}, distance_threshold=0.05):
        d = goal_distance(ag_2, g)
        return (d < distance_threshold).astype(np.float32)

    def quasimetric(o, ag, g, distance_threshold=0.05, horizon=100, **kwargs):
        dist = np.linalg.norm(ag - g, ord=2, axis=1)
        nsteps_to_goal = dist // distance_threshold
        Q = horizon - nsteps_to_goal
        return [], Q

    def gaussian_mixture():
        dists_prior = [0.5, 0.5]
        dist_idx = np.random.choice([0, 1], 1, p=dists_prior)[0]
        if dist_idx == 0:  # uniform
            x = np.random.uniform(low=0.0, high=1.0, size=2)
        elif dist_idx == 1:  # normal
            x = np.random.multivariate_normal(mean=[0.1, 0.5], cov=0.001 * np.eye(2))
        return x

    x = np.asarray([1, 1], dtype=np.float32)

    def random_walk(x_, scale=0.01):
        x = x_ + np.random.uniform(low=-scale, high=scale, size=2)
        x = np.clip(x, 0, 1)
        return x

    uniformizer = MetricDiversifier(k=10, vis=True, load_p=1, vis_coords=[0, 1],
                                    prop_adjust_interval=1000,
                                    save_path='logs/2020-01-01',
                                    reward_func=compute_reward,
                                    )
    counter = 0
    while True:
        # x = gaussian_mixture()
        x = random_walk(x)
        pnt = uniformizer.init_record(o=x.copy())
        uniformizer.load_new_point(pnt, d_func=quasimetric)
        counter += 1
        if counter % 10000 == 0:
            uniformizer.save(message=counter)
            print(f"epoch: {counter}, cover size: {uniformizer.current_size}, min dist: {uniformizer.M.min()}")
        uniformizer.age += 1
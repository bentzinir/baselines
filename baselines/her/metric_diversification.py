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

    def save(self, save_path, message):
        if not os.path.exists(os.path.join(save_path, "mca_figures")):
            os.makedirs(os.path.join(save_path, "mca_figures"))
        fig_name = os.path.join(save_path, "mca_figures", f"{message}.png")
        plt.savefig(fig_name)
        print(f"saving figure: {fig_name}")


class MetricDiversifier:
    def __init__(self, k, random_cover=False, load_p=1, vis=False, vis_coords=None, load_model=None, save_path=None, **kwargs):
        self.k = k
        self.k_approx = k // 10
        self.M = -np.inf * np.ones((self.k, self.k))
        self.age = np.inf * np.ones(self.k)
        self.buffer = [None for _ in range(self.k)]
        [self.invalidate_idx(idx) for idx in range(self.k)]
        self.random_cover = random_cover
        self.save_path = save_path
        self.load_p = load_p
        self.vis = vis
        self.vis_coords = vis_coords
        self.counter = 0
        self.observer = None
        if load_model is not None:
            self.buffer = self.load_model(load_model)
            print(f"Loaded model: {load_model}")
            print(f"Model size: {self.current_size}")

    def _buffer_2_array(self, val, idxs=None):
        if idxs is None:
            idxs = list(range(self.current_size))
        return np.asarray([self.buffer[idx][val] for idx in idxs])

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

    @staticmethod
    def quasimetric(x1_o, x1_ag, x2_o, x2_ag, d_func=None, feat_distance=True):
        '''
        :param x1_o: set of points
        :param x2_o: set of points
        :param d_func: a quasimetric distance function
        :param feat_distance: decide whether distance is measured on features or on entire state
        :return: the point in the set that is closest to x
        '''
        if d_func is None:
            if feat_distance:
                # x1x2_distance = np.linalg.norm(x1_ag - x2_ag, ord=2, axis=1)
                # TODo: remove after debug
                distance_mat = ((x1_ag - x2_ag) ** 2)
                weight_mat = np.ones_like(distance_mat)
                weight_mat[..., 3:] = 10
                x1x2_distance = (weight_mat * distance_mat).sum(axis=1)
            else:
                x1x2_distance = np.linalg.norm(x1_o - x2_o, ord=2, axis=1)
        else:
            _, Q = d_func(o=x1_o, ag=x1_ag, g=x2_ag, compute_Q=True, use_target_net=True)
            x1x2_distance = - Q.squeeze()
        return x1x2_distance

    def prepare_dfunc_inputs(self, pnt):
        set_o_mat = self._buffer_2_array(val='o', idxs=list(range(self.current_size)))
        set_ag_mat = self._buffer_2_array(val='ag', idxs=list(range(self.current_size)))
        pnt_o_mat = np.repeat(np.expand_dims(pnt['o'], 0), repeats=self.current_size, axis=0)
        pnt_ag_mat = np.repeat(np.expand_dims(pnt['ag'], 0), repeats=self.current_size, axis=0)
        return set_o_mat, set_ag_mat, pnt_o_mat, pnt_ag_mat

    def adjust_set(self, new_pnt, d_func):

        for idx in self.used_slots():
            pnt = self.buffer[idx]
            if pnt['ag'] is None:
                assert False

        ref_idx_set = np.random.choice(self.used_slots(), self.k_approx, replace=False)

        assert len(self.open_slots()) == 0

        # refresh outdated matrix entries
        for idx in ref_idx_set:
            if self.age[idx] > 10:
                pnt = self.buffer[idx]
                set_o_mat, set_ag_mat, pnt_o_mat, pnt_ag_mat = self.prepare_dfunc_inputs(pnt)
                # update "from" distances pnt -> set
                self.M[idx] = self.quasimetric(x1_o=pnt_o_mat, x1_ag=pnt_ag_mat, x2_o=set_o_mat, x2_ag=set_ag_mat, d_func=d_func)
                self.M[idx, idx] = np.inf

                # update "to" distances set -> pnt
                self.M[:, idx] = self.quasimetric(x1_o=set_o_mat, x1_ag=set_ag_mat, x2_o=pnt_o_mat, x2_ag=pnt_ag_mat, d_func=d_func)
                self.M[idx, idx] = np.inf

                self.age[idx] = 0

        set_o_mat, set_ag_mat, newpnt_o_mat, newpnt_ag_mat = self.prepare_dfunc_inputs(new_pnt)

        distances_to_new_pnt = self.quasimetric(x1_o=set_o_mat, x1_ag=set_ag_mat, x2_o=newpnt_o_mat, x2_ag=newpnt_ag_mat, d_func=d_func)

        ##################################
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

        # FORCED UPDATE
        if b_idx == -1:
            if np.random.binomial(n=1, p=0.001):
                b_idx = random.choice(range(self.current_size))
                # print(f"Forced update")

        if b_idx >= 0:

            self.M[b_idx] = self.quasimetric(x1_o=newpnt_o_mat, x1_ag=newpnt_ag_mat, x2_o=set_o_mat, x2_ag=set_ag_mat, d_func=d_func)

            self.M[:, b_idx] = distances_to_new_pnt

            self.M[b_idx, b_idx] = np.inf

            # replace in buffer
            self.buffer[b_idx] = new_pnt

        return b_idx >= 0

    def invalidate_idx(self, idx):
        self.M[idx] = -np.inf
        self.M[:, idx] = -np.inf
        self.M[idx, idx] = np.inf
        self.buffer[idx] = None
        self.age[idx] = np.inf

    def load_new_point(self, new_pnt, d_func=None):
        if new_pnt is None:
            return False
        # load new_pnt if the buffer is empty
        if 'g' in new_pnt.keys():
            a = 1
        if self.current_size < self.k:
            self.buffer[random.choice(self.open_slots())] = new_pnt
            # self.buffer.append(new_pnt)
            return False
        if not np.random.binomial(n=1, p=self.load_p):
            return False

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
        pts = [self.buffer[idx]['o'][self.vis_coords] for idx in range(self.current_size)]
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
        return {'o': o.copy(), 'ag': ag.copy(), 'qpos': qpos.copy(), 'qvel': qvel.copy()}

    def save_model(self, save_path, message=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if message is None:
            message = 'cover'
        f_name = f"{save_path}/{message}.json"
        with open(f_name, 'w') as outfile:
            json_str = json.dumps(self._buffer_2_dict(), indent=4, sort_keys=True)
            outfile.write(json_str)
        print(f"saving cover: {f_name}, (size:{self.current_size})")

    @staticmethod
    def load_model(load_path):
        if load_path is None:
            return
        if not os.path.exists(load_path):
            return None
        with open(load_path, 'r') as infile:
            json_buffer = json.load(infile)
        buffer = deque(maxlen=len(json_buffer))
        for key, val in json_buffer.items():
            buffer.append(val)
        for i in range(len(buffer)):
            buffer[i]['o'] = np.asarray(buffer[i]['o'])
            buffer[i]['ag'] = np.asarray(buffer[i]['ag'])
        return buffer


if __name__ == '__main__':

    random_cover = False
    save_path = 'logs/2020-01-01'
    if random_cover:
        save_path = f"{save_path}/random"
    else:
        save_path = f"{save_path}/learned"

    uniformizer = MetricDiversifier(k=1000, vis=True, load_p=1, vis_coords=[0, 1],
                                    # load_model='/home/nir/work/git/baselines/logs/01-01-2020/mca_cover/0_model.json'
                                    prop_adjust_interval=1000,
                                    random_cover=random_cover, save_path=save_path
                                    )

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

    counter = 0
    while True:
        # x = gaussian_mixture()
        x = random_walk(x)
        pnt = uniformizer.init_record(o=x)
        uniformizer.load_new_point(pnt)
        counter += 1
        if counter % 10000 == 0:
            uniformizer.save(save_path=save_path, message=counter)
            print(f"random cover;{random_cover}, epoch: {counter}, cover size: {uniformizer.current_size}")

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import json
import collections


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class VisObserver:
    def __init__(self, ndim=2):
        self.ndim = ndim
        fig = plt.figure()
        if ndim == 2:
            ax = fig.add_axes([0, 0, 1, 1])
            self.scat = ax.scatter(x=[], y=[], c='r', marker='+')
            self.proposal_scat = plt.scatter(x=[], y=[], c='b', marker='+')
        elif ndim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = Axes3D(fig)
            self.scat_ax = ax
            self.scat = ax.scatter(xs=[], ys=[], c='r', marker='+')
            self.proposal_scat = ax.scatter(xs=[], ys=[], c='b', marker='+')
        # plt.show(block=False)
        plt.tight_layout()

    def update(self, points, proposal_points=None, draw=False):
        if self.ndim == 2:
            self.scat.set_offsets(points)
            if proposal_points:
                self.proposal_scat.set_offsets(proposal_points)
                self.proposal_scat.set_color('b')
            else:
                self.proposal_scat.set_color('c')
        elif self.ndim == 3:
            pts = np.asarray(points)
            self.scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            self.scat_ax.autoscale()
            self.scat_ax.set_xlim([pts[:, 0].min(), pts[:, 0].max()])
            self.scat_ax.set_ylim([pts[:, 1].min(), pts[:, 1].max()])
            self.scat_ax.set_zlim([pts[:, 2].min(), pts[:, 2].max()])
            if proposal_points:
                prop_pts = np.asarray(proposal_points)
                self.proposal_scat._offsets3d = (prop_pts[:, 0], prop_pts[:, 1], prop_pts[:, 2])
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
    def __init__(self, k, reward_fun, random_cover=False, load_p=1, vis=False, vis_coords=None, load_model=None,
                 phase_interval=500, dilute_at_goal=False, **kwargs):
        self.kmin = k  # 5
        self.k = k
        self.reward_fun = reward_fun
        self.buffer = deque(maxlen=self.k)
        self.proposal = False
        self.phase_interval = phase_interval
        self.x_proposal = self.init_record(x=None)
        self.random_cover = random_cover
        self.dilute_at_goal = dilute_at_goal
        self.load_p = load_p
        self.vis = vis
        self.vis_coords = vis_coords
        self.counter = 0
        self.observer = None
        if load_model is not None:
            self.buffer = self.load_model(load_model)
            print(f"Loaded model: {load_model}")
            print(f"Model size: {self.current_size}")

        assert load_p == 1, 'deprecated'

    def _buffer_2_array(self, val, idxs=None):
        if idxs is None:
            idxs = list(range(self.current_size))
        return np.asarray([self.buffer[idx][val] for idx in idxs])

    def _buffer_2_dict(self):
        z = {}
        for i, b in enumerate(self.buffer):
            z[i] = {}
            for key, val in b.items():
                if type(val) == np.ndarray:
                    val = val.tolist()
                    # val = f'[{", ".join(map(str, val))}]'
                elif val == np.inf or val == -np.inf:
                    val = None
                elif isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                elif isinstance(val, collections.Mapping) and key == 'info':  # this is for Mujoco sim state
                    for skey, sval in val.items():
                        if type(sval) == np.ndarray:
                            val[skey] = sval.tolist()
                z[i][key] = val
        return z

    @staticmethod
    def set_to_point_distance(set_pts, x, d_func=None):
        '''
        :param set_pts: a set of points
        :param x: new point
        :param d_func: a quasimetric distance function
        :return: the point in the set that is closest to x
        '''
        if d_func is None:
            distances_2_x = np.linalg.norm(x - set_pts, ord=2, axis=1)
        else:
            x_mat = np.repeat(np.expand_dims(x, 0), repeats=set_pts.shape[0], axis=0)
            _, Q = d_func(o=set_pts, ag=None, g=x_mat, compute_Q=True)
            distances_2_x = - Q.squeeze()
        return distances_2_x

    def _set_distance(self, d_func):
        '''
        :param idxs: indexes of points in the buffer
        :param d_func: a quasimetric distance function
        :return: a member of idxs that is most reachable by any other point
        '''

        set_idxs = list(range(self.current_size))
        for idx in range(self.current_size):
            p = self.buffer[idx]['x_feat']
            _X = self._buffer_2_array(val='x', idxs=set_idxs)
            distance_to_p = self.set_to_point_distance(_X, p, d_func=d_func)
            distance_to_p[idx] = np.inf
            self.buffer[idx]['distance'] = distance_to_p.min()
            self.buffer[idx]['nn'] = distance_to_p.argmin()
            self.buffer[idx]['c'] += 1
        return

    def propose_point(self, new_pnt, new_pnt_at_goal, verbose=False):
        if self.counter < self.phase_interval:
            self.counter += 1
            if new_pnt['distance'] > self.x_proposal['distance'] and not new_pnt_at_goal:
                self.x_proposal = new_pnt
        else:
            self.counter = 0
            if self.current_size < self.k and self.x_proposal['x'] is not None:
                if verbose:
                    print(f"Appending: {self.current_size} -> {self.current_size + 1}")
                # self.fit_buffer_size(1)
                self.buffer.append(self.x_proposal)
                self.x_proposal = self.init_record()
            self.proposal = False

    def adjust_set(self, new_pnt, distances_2_new_pnt, d_func):
        if self.counter < self.phase_interval:
            self.counter += 1
            # Calculate pairwise distances
            self._set_distance(d_func)

            ##################################
            b_idx = -1
            b_delta = -np.inf
            for j in range(self.current_size):
                j_distances = distances_2_new_pnt.copy()
                j_distances[j] = np.inf
                delta_j = j_distances.min() - self.buffer[j]['distance']
                if delta_j > b_delta and delta_j > 0:
                    b_delta = delta_j
                    b_idx = j
            ##################################

            if b_idx >= 0:
                # pop
                del self.buffer[b_idx]
                # append
                self.buffer.append(new_pnt)
                # self.counter = 0
        else:
            self.counter = 0
            if self.dilute_at_goal:
                self.dilute()
            self.proposal = True

    def _load_active(self, new_pnt, d_func):
        # load new_pnt if the buffer is empty
        if self.current_size < self.kmin:
            self.buffer.append(new_pnt)
            return

        # if not np.random.binomial(n=1, p=self.load_p):
        #     return

        self.counter += 1

        # calculate the distance of new_pnt to all points
        X = self._buffer_2_array(val='x', idxs=list(range(self.current_size)))

        distances_2_new_pnt = self.set_to_point_distance(X, new_pnt['x_feat'], d_func=d_func)

        new_pnt['distance'] = distances_2_new_pnt.min()

        # check that new_pnt is not "at goal" w.r.t any existing point
        new_pnt_at_goal = self._set_pnt_reward(new_pnt)

        if self.proposal:
            self.propose_point(new_pnt, new_pnt_at_goal)
        else:
            self.adjust_set(new_pnt, distances_2_new_pnt, d_func)

    def _load_inactive(self, new_point, verbose=False):
        # load new_pnt if the buffer is empty
        if self.current_size < self.kmin:
            self.buffer.append(new_point)
            return
        if not np.random.binomial(n=1, p=0.005):
            return
        if self.current_size > self.k:
            self.buffer.popleft()
        # self.fit_buffer_size(1)
        self.buffer.append(new_point)
        if self.dilute_at_goal:
            self.dilute(verbose=False)
        if verbose:
            print(f"Buffer size: {self.current_size}")

    def load_new_point(self, new_point, d_func=None):
        '''

        :param new_point: a dictionary representing the new point
        :param d_func:
        :return:
        '''
        if new_point is None:
            return
        if self.random_cover:
            self._load_inactive(new_point)
        else:
            self._load_active(new_point, d_func)

    def _set_pnt_reward(self, pnt):
        for i in range(self.current_size):
            if self.reward_fun(pnt['x_feat'], self.buffer[i]['x_feat'], info=None):
                return True
        return False

    def dilute(self, verbose=True):
        _size = self.current_size
        if self.current_size <= self.kmin:
            return False
        dilute = False
        dilutions = []
        for i in range(self.current_size):
            for j in range(i+1, self.current_size):
                if self.reward_fun(self.buffer[i]['x_feat'], self.buffer[j]['x_feat'], info=None):
                    if self.buffer[i]['distance'] < self.buffer[j]['distance']:
                        dilutions.append(i)
                    else:
                        dilutions.append(j)
                    dilute = True
        if not dilute:
            return False
        dilutions = list(set(dilutions))
        dilutions.sort(reverse=True)
        for idx in dilutions:
            del self.buffer[idx]
        # self.fit_buffer_size()
        if verbose:
            print(f"Diluting: {_size} -> {self.current_size}")
        return dilute

    def draw(self, n, farthest=False):
        if self.current_size == 0:
            return None
        if farthest:
            s_idxs = np.argsort([b['distance'] for b in self.buffer])[-n:]
        else:
            s_idxs = np.random.choice(list(range(self.current_size)), n)
        g_idxs = np.random.choice(list(range(self.current_size)), n)
        batch = []
        for s_idx, g_idx in zip(s_idxs, g_idxs):
            s_record = self.buffer[s_idx]
            g_record = self.buffer[g_idx]
            info = s_record['info']
            if isinstance(info, collections.Mapping):
                info = Bunch(s_record['info'])
            batch.append({'x': s_record['x'], 'info': info, 'g': g_record['x_feat']})
        return batch

    def _update_figure(self):
        pts = [self.buffer[idx]['x'][self.vis_coords] for idx in range(self.current_size)]
        prop_pts = None
        if self.proposal and self.x_proposal['x'] is not None:
            prop_pts = [self.x_proposal['x'][self.vis_coords]]
            self.observer.update(pts, prop_pts)
        else:
            self.observer.update(pts, prop_pts)

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

    @property
    def current_size(self):
        return len(self.buffer)

    @staticmethod
    def init_record(x=None, x_feat=None, info=None, distance=-np.inf):
        if x_feat is None:
            x_feat = x
        return {'x': x, 'x_feat': x_feat, 'c': 0, 'info': info, 'distance': distance, 'nn': None}

    def save_model(self, save_path, message=None):
        save_dir = os.path.join(save_path, "mca_cover")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if message is None:
            message = 'cover'
        f_name = f"{save_dir}/{message}.json"
        with open(f_name, 'w') as outfile:
            json_str = json.dumps(self._buffer_2_dict(), indent=4, sort_keys=True)
            outfile.write(json_str)
        print(f"saving cover: {f_name}")

    @staticmethod
    def load_model(load_path):
        with open(load_path, 'r') as infile:
            json_buffer = json.load(infile)
        buffer = deque(maxlen=len(json_buffer))
        for key, val in json_buffer.items():
            buffer.append(val)
        for i in range(len(buffer)):
            buffer[i]['x'] = np.asarray(buffer[i]['x'])
            buffer[i]['x_feat'] = np.asarray(buffer[i]['x_feat'])
        return buffer


if __name__ == '__main__':

    def reward_fun(s, g, **kwargs):
        if np.linalg.norm(s - g, ord=2, axis=-1) < 0.1:
            return 1
        else:
            return 0

    random_cover = True
    save_path = 'logs/2020-01-01-TwoGaussians'
    if random_cover:
        save_path = f"{save_path}/random"
    else:
        save_path = f"{save_path}/learned"

    uniformizer = MetricDiversifier(k=100, reward_fun=reward_fun, vis=True, load_p=1, vis_coords=[0, 1],
                                    # load_model='/home/nir/work/git/baselines/logs/01-01-2020/mca_cover/0_model.json'
                                    prop_adjust_interval=1000,
                                    random_cover=random_cover, save_path=save_path
                                    )

    dists_prior = [0.5, 0.5]
    counter = 0
    while True:
        dist_idx = np.random.choice([0, 1], 1, p=dists_prior)[0]
        if dist_idx == 0:  # uniform
            x = np.random.uniform(low=0.0, high=1.0, size=2)
        elif dist_idx == 1:  # normal
            x = np.random.multivariate_normal(mean=[0.1, 0.5], cov=0.001 * np.eye(2))

        pnt = uniformizer.init_record(x=x)
        uniformizer.load_new_point(pnt)
        counter += 1
        if counter % 10000 == 0:
            uniformizer.save(save_path=save_path, message=counter)
            print(f"random cover;{random_cover}, epoch: {counter}, cover size: {uniformizer.current_size}")

import multiprocessing as mp
from baselines.her.metric_diversification import MetricDiversifier
from baselines.common.vec_env import CloudpickleWrapper
import numpy as np


def make_state_model_vec(k_vec, vis, vis_coords, load_path, log_path, random_cover, load_prob, phase_length, dilute_at_goal):
    def make_thunk(k):
        return lambda: MetricDiversifier(k=k,
                                         vis=vis,
                                         vis_coords=vis_coords,
                                         load_model=load_path,
                                         save_path=f"{log_path}/{k}/mca_cover",
                                         random_cover=random_cover,
                                         load_p=load_prob,
                                         phase_length=phase_length,
                                         dilute_at_goal=dilute_at_goal)
    return StateModelVec([make_thunk(k) for k in k_vec])


def worker(remote, parent_remote, state_model_fn_wrappers):
    def update_metric_model(state_model, policy):
        if policy.buffer.current_size == 0:
            return False
        batch = policy.buffer.sample(100)
        for o, ag, qpos, qvel in zip(batch['o'], batch['ag'], batch['qpos'], batch['qvel']):
            new_point = state_model.init_record(x=o, x_feat=ag, qpos=qpos, qvel=qvel)
            state_model.load_new_point(new_point, d_func=policy.get_actions)
        return True

    parent_remote.close()
    state_models = [state_model_fn_wrapper() for state_model_fn_wrapper in state_model_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'draw':
                remote.send([state_model.draw(data) for state_model in state_models])
            elif cmd == 'update':
                remote.send([update_metric_model(state_model, policy=data) for state_model in state_models])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('StateModelVecEnv worker: got KeyboardInterrupt')
    finally:
        for state_model in state_models:
            state_model.close()


class StateModelVec:
    def __init__(self, model_fns, context='spawn'):
        self.n_remotes = len(model_fns)
        model_fns = np.array_split(model_fns, self.n_remotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_remotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(model_fn))) for
                   (work_remote, remote, model_fn) in zip(self.work_remotes, self.remotes, model_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def draw(self, n):
        print("Draw")
        for remote in self.remotes:
            remote.send(('draw', n))
        states = [remote.recv() for remote in self.remotes]
        return states

    def update(self, policy):
        for remote in self.remotes:
            remote.send(('update', policy))
        dones = [remote.recv() for remote in self.remotes]
        return dones
from . import VecEnvWrapper
import numpy as np
import gym

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        obs_shape = self.observation_space.shape
        if type(self.observation_space) is gym.spaces.dict.Dict:
            obs_shape = self.observation_space["observation"].shape
        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = TfRunningMeanStd(shape=obs_shape, scope='ob_rms') if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None
        else:
            from baselines.common.running_mean_std import RunningMeanStd
            self.ob_rms = RunningMeanStd(shape=obs_shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        def filt(val):
            self.ob_rms.update(val)
            return np.clip((val - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        if self.ob_rms:
            if type(obs) is dict:
                obs["observation"] = filt(obs["observation"])
            else:
                obs = filt(obs)
            return obs
        else:
            return obs

    def reset(self, ex_inits=None):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset(ex_inits=ex_inits)
        return self._obfilt(obs)

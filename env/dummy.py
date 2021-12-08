# -*- coding: utf-8 -*-
import gym
import multiprocessing as mp
import numpy as np
from typing import Dict, List

from agent import BaseAgent


def dict_stack(insts: List[Dict]):
    if len(insts) == 0:
        return {}
    return {key: np.stack([d[key] for d in insts]) for key in insts[0].keys()}


class ReplayBuffer:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self.meta = {'obs': [], 'act': [], 'rew': [], 'obs_next': [], 'done': []}
        # for k in list(self._meta.keys()):
        #     if isinstance(self._meta[k], list):
        #         self._meta[k] = []

    def add(self, obs, act, rew, obs_next, done):
        self.meta['obs'].append(obs)
        self.meta['act'].append(act)
        self.meta['rew'].append(rew * 1.0)
        self.meta['obs_next'].append(obs_next)
        self.meta['done'].append(done)
        self._index += 1

    @property
    def data(self):
        return {key: np.stack(value) for key, value in self.meta.items()}

    def __getattr__(self, name):
        return self.meta[name]

    def __len__(self) -> int:
        return self._index


class Collector:
    def __init__(self, env: gym.Env, agent: BaseAgent):
        self.env = env
        self.agent = agent
        self.buffer = ReplayBuffer()

    def reset(self) -> None:
        obs = self.env.reset()
        self.agent.reset(obs)
        self.buffer.reset()
        return obs

    def collect(self, traffic_signal: mp.Value):
        # start_time = time.time()
        obs = self.reset()
        while True:
            act = self.agent(obs)
            obs_next, rew, done, info = self.env.step(act)
            done = done or (not traffic_signal.value)
            self.buffer.add(obs, act, rew, obs_next, done)
            if done: break
            obs = obs_next
        # duration = max(time.time() - start_time, 1e-9)
        # print(f'{os.getpid()} : {round(duration, 2)}')
        return {
            'len': len(self.buffer),
            'rew': sum(self.buffer.rew),
            'suc': info['success'],
            # 'time': duration
        }

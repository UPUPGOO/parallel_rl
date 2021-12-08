# -*- coding: utf-8 -*-
import gym
from env.subproc import SubprocEnv
from agent import SimpleAgent
import torch


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, device='cpu'):
        super(Actor, self).__init__()
        self.net = torch.nn.Linear(state_dim, action_dim)
        self.device = device

    def forward(self, s):
        return self.net(s)


process_num = 3
env_fns = [lambda: gym.make('Pendulum-v0') for _ in range(process_num)]
tmp_env = env_fns[0]()
state_dim = tmp_env.observation_space.shape[0]
action_dim = tmp_env.action_space.shape[0]
tmp_env.close()
del tmp_env

env = SubprocEnv(env_fns, SimpleAgent(Actor(state_dim, action_dim)), buffer_size=10000)

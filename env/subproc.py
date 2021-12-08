# -*- coding: utf-8 -*-
import gym
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import warnings
from typing import List, Callable, Optional, Union
import datetime

from agent import BaseAgent
from env.batch import Batch
from env.dummy import Collector, dict_stack

def get_unique_num():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + str(np.random.randint(10, 100))

class SharedBufferCell:
    def __init__(self, name: str, dtype: np.dtype, dim: int, max_size: int, create: bool = False):
        nbytes = np.empty((max_size, dim), dtype).nbytes
        self.shm = SharedMemory(name=name, create=create, size=nbytes)
        shape = (max_size, dim) if dim > 1 else (max_size,)
        self.array = np.ndarray(shape=shape, dtype=dtype, buffer=self.shm.buf)
        self.create = create

    def store(self, start_index: int, value: np.ndarray):
        self.array[start_index:start_index + len(value)] = value[:]

    @property
    def data(self):
        return self.array

    def __del__(self):
        del self.array
        self.shm.close()
        if self.create:
            self.shm.unlink()


class SharedBuffer:
    def __init__(self, max_size: int, format_dict: dict, create: bool = False):
        self.cells = {}
        for name, format in format_dict.items():
            # if isinstance(format, dict):
            #     self.cells[name] = SharedBuffer(max_size, format, create)
            # else:
            shm_name, dtype, dim = format
            self.cells[name] = SharedBufferCell(shm_name, dtype, dim, max_size, create)
        self._max_size = max_size
        self._format_dict = format_dict

    def store(self, start_index: int, **kwargs):
        for name, value in kwargs.items():
            # if isinstance(value, dict):
            #     self.cells[name].store(start_index, **value)
            # else:
            self.cells[name].store(start_index, value)

    def __len__(self) -> int:
        return self._max_size

    @property
    def data(self):
        return {name: cell.data for name, cell in self.cells.items()}

    @property
    def max_size(self):
        return self._max_size

    @property
    def format_dict(self):
        return self._format_dict

    @staticmethod
    def create_format_dict(inst_dict: dict, prefix: str = None):
        if prefix is None:
            prefix = get_unique_num()
        format_dict = {}
        for name, inst in inst_dict.items():
            shm_name = f'{prefix}.{name}'
            # if isinstance(inst, dict):
            #     format_dict[name] = SharedBuffer.create_format_dict(inst, shm_name)
            # else:
            inst = np.array(inst)
            format_dict[name] = (shm_name, inst.dtype, inst.size)
        return format_dict


def _worker(
        rank: int,
        parent: mp.connection.Connection,
        p: mp.connection.Connection,
        env_fn: Callable[[], gym.Env],
        agent: BaseAgent,
        buffer_fn: Callable[[], SharedBuffer],
        buffer_counter: mp.Value,
        episode_array: mp.Array,
        traffic_signal: mp.Value,
):
    parent.close()
    env = env_fn()
    collector = Collector(env, agent)
    buffer = buffer_fn()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "collect":
                n_step, n_episode = data
                assert n_step < buffer.max_size, 'The allocated shared buffer is too small!!!'
                collected_step, collected_episode = 0, 0
                metrics = []
                while True:
                    res = collector.collect(traffic_signal)
                    with buffer_counter.get_lock():
                        start_index = buffer_counter.value
                        size = start_index + res['len']
                        if size < buffer.max_size:
                            buffer_counter.value = size
                        else:
                            size = start_index
                    if size > start_index:
                        buffer.store(start_index, **collector.buffer.data)
                        collected_step += res['len']
                        collected_episode += 1
                        if traffic_signal.value:
                            with episode_array.get_lock():
                                episode_array[rank] += 1
                            metrics.append(res)
                    if not traffic_signal.value: break
                    if (buffer_counter.value >= n_step and (np.array(episode_array[:]) >= n_episode).all()):
                        traffic_signal.value = False
                        break
                    if size == start_index:
                        warnings.warn(f'The allocated shared buffer is almost full.\n'
                                      f'collected({buffer_counter.value}), needed({n_step}), allocated({buffer.max_size})\n')
                        traffic_signal.value = False
                        break
                if len(metrics) == 0:
                    warnings.warn('No complete trajectory is collected.\n'
                                  f'collected({buffer_counter.value}), needed({n_step}), allocated({buffer.max_size})\n'
                                  f'episode num: {episode_array[:]}\n'
                                  f'traffic signal: {traffic_signal.value}')
                    metrics.append(res)
                p.send({
                    'episode': collected_episode,
                    'step': collected_step,
                    **{name: np.mean(value) for name, value in dict_stack(metrics).items()}
                })
            elif cmd == "seed":
                p.send(env.seed(data) if hasattr(env, "seed") else None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "callback":
                res = env.callback(*data)
                p.send(res)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocEnv:
    def __init__(self, env_fns: List[Callable[[], gym.Env]], agent: BaseAgent, buffer_size: int):
        self.env_num = len(env_fns)
        agent.to_device('cpu')
        agent.share_memory()

        tmp_env = env_fns[0]()
        obs = tmp_env.reset()
        act = tmp_env.action_space.sample()
        obs_next, rew, done, info = tmp_env.step(act)
        format_dict = SharedBuffer.create_format_dict({'obs': obs, 'act': act, 'rew': rew, 'obs_next': obs_next, 'done': done})
        self.buffer = SharedBuffer(buffer_size, format_dict, create=True)
        buffer_fn = lambda: SharedBuffer(buffer_size, format_dict, create=False)
        tmp_env.close()
        del tmp_env

        self.buffer_counter = mp.Value('i', 0, lock=True)
        self.episode_array = mp.Array('i', self.env_num, lock=True)
        self.traffic_signal = mp.Value('b', True, lock=True)
        self.reset_buffer()

        self.env_pipes, self.env_processes = [], []
        for i in range(self.env_num):
            parent_remote, child_remote = mp.Pipe()
            args = (i, parent_remote, child_remote, env_fns[i], agent, buffer_fn, self.buffer_counter, self.episode_array, self.traffic_signal)
            process = mp.Process(target=_worker, args=args, daemon=True)
            process.start()
            child_remote.close()
            self.env_pipes.append(parent_remote)
            self.env_processes.append(process)

    def reset_buffer(self):
        with self.buffer_counter.get_lock():
            self.buffer_counter.value = 0
        with self.episode_array.get_lock():
            self.episode_array[:] = np.zeros(self.env_num, dtype=int)
        with self.traffic_signal.get_lock():
            self.traffic_signal.value = True

    @property
    def data(self) -> Batch:
        with self.buffer_counter.get_lock():
            size = self.buffer_counter.value
        return Batch(self.buffer.data)[:size]

    def collect(self, n_step: int = None, n_episode: int = None):
        assert not (n_step is None and n_episode is None)
        if n_step is None: n_step = self.buffer.max_size
        if n_episode is None:  n_episode = 1
        assert n_step > 0 and n_episode > 0
        [p.send(["collect", (n_step, n_episode)]) for p in self.env_pipes]
        return dict_stack([p.recv() for p in self.env_pipes])

    def seed(self, seed: Optional[Union[int, List[int]]] = None):
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        [p.send(["seed", s]) for s, p in zip(seed_list, self.env_pipes)]
        return [p.recv() for p in self.env_pipes]

    def callback(self, func_name: str, params: dict = None, id: Optional[Union[int, List[int], np.ndarray]] = None):
        id = self._wrap_id(id)
        [self.env_pipes[i].send(["callback", (func_name, params)]) for i in id]
        return [self.env_pipes[i].recv() for i in id]

    def close(self):
        try:
            [p.send(["close", None]) for p in self.env_pipes]
            # mp may be deleted so it may raise AttributeError
            [p.recv() for p in self.env_pipes]
            [p.join() for p in self.env_processes]
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        [p.terminate() for p in self.env_processes]

    def _wrap_id(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> Union[List[int], np.ndarray]:
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        return id

    def __getattr__(self, key: str):
        [p.send(["getattr", key]) for p in self.env_pipes]
        return [p.recv() for p in self.env_pipes]

    def __len__(self):
        return self.env_num

    def __del__(self):
        self.close()

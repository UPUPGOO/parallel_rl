# -*- coding: utf-8 -*-
import numpy as np
import pprint
from typing import Any, Union, Iterator


class Batch:
    def __init__(self, batch_dict: dict = None, **kwargs):
        if batch_dict is not None:
            self.__dict__.update(batch_dict)
        if len(kwargs) > 0:
            self.__init__(kwargs)

    def __setattr__(self, key: str, value: np.ndarray) -> None:
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        return getattr(self.__dict__, key)

    def __getitem__(self, index: Union[str, int, slice, np.ndarray]) -> "Batch":
        if isinstance(index, str):
            return self.__dict__[index]
        return Batch({key: value[index] for key, value in self.items()})

    def __len__(self):
        keys = list(self.keys())
        if len(keys) == 0:
            return 0
        return len(self[keys[0]])

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + "(\n"
        flag = False
        for k, v in self.__dict__.items():
            rpl = "\n" + " " * (6 + len(k))
            obj = pprint.pformat(v).replace("\n", rpl)
            s += f"    {k}: {obj},\n"
            flag = True
        if flag:
            s += ")"
        else:
            s = self.__class__.__name__ + "()"
        return s

    def split(self, size: int, shuffle: bool = True, merge_last: bool = False) -> Iterator["Batch"]:
        length = len(self)
        assert 1 <= size  # size can be greater than length, return whole batch
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx:idx + size]]

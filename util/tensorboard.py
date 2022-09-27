"""
Adapted from https://github.com/haotian-liu/yolact_edge/blob/master/utils/tensorboard_helper.py
"""

import os

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class SummaryHelper(object):
    def __init__(self):
        self._w = None
        self.log_dir = None
        self._step = 0
        self._last_log_step = 0
        self.image_interval = 100
    
    def set_log_dir(self, log_dir):
        if get_rank() != 0: return
        if self._w is not None:
            raise NotImplementedError("SummaryWriter has been initialized already!")
        self._w = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
    
    def add_scalar(self, key, value, step=None):
        if self._w is None: return
        if step is None: step = self._step
        self._w.add_scalar(key, value, step)

    def add_text(self, key, value, step=None):
        if self._w is None: return
        if step is None: step = self._step
        self._w.add_text(key, value, step)

    def add_images(self, key, value):
        if self._w is None: return
        self._w.add_images(key, value, self._step)
    
    def add_image(self, key, value, dataformats='HWC'):
        if self._w is None: return
        self._w.add_image(key, value, self._step, dataformats=dataformats)

    def step(self):
        self.set_step(self._step + 1)

    def set_image_interval(self, interval):
        self.image_interval = interval

    def set_step(self, step):
        self._step = step
        

TB = SummaryHelper()
import os
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

import torch
import numpy as np

from preprocess import load_sparse, load_subgraphs




def load_adj_cooccur(path, device=torch.device('cuda')):
    filename = os.path.join(path, 'adj_cooccur.npz')
    adj_cooccur = torch.from_numpy(load_sparse(filename)).to(device=device, dtype=torch.float32)
    return adj_cooccur

def load_adj_temporal(path, device=torch.device('cuda')):
    filename1 = os.path.join(path, 'adj_temporal.npz')
    adj_temporal = torch.from_numpy(load_sparse(filename1)).to(device=device, dtype=torch.float32)
    return adj_temporal


class EHRDataset:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cuda')):
        super().__init__()
        self.path = data_path
        # self.code_x, self.visit_lens, self.y, self.divided, self.neighbors = self._load(label)
        self.code_x, self.visit_lens, self.y, self.neighbors = self._load(label)
        # 加载预计算的子图数据
        self.cooccur_subgraphs, self.temporal_subgraphs = self._load_subgraphs()
        self._size = self.code_x.shape[0]
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        if label == 'm':
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h':
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        # divided = load_sparse(os.path.join(self.path, 'divided.npz'))
        neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'))
        # return code_x, visit_lens, y, divided, neighbors

        return code_x, visit_lens, y, neighbors
    
    def _load_subgraphs(self):
        """加载预计算的子图数据"""
        try:
            return load_subgraphs(self.path)
        except FileNotFoundError:
            print(f"Warning: Subgraph data not found in {self.path}")
            print("Returning empty lists. Consider running preprocessing to generate subgraphs.")
            return [], []

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index): # 切分数据
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        # divided = torch.from_numpy(self.divided[slices]).to(device)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        

        return code_x, visit_lens, y, neighbors

class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.lrs = self._generate_lr(milestones, lrs)
        self.current_epoch = 0

    def _generate_lr(self, milestones, lrs):
        milestones = [1] + milestones + [self.epochs + 1]
        lrs = [self.init_lr] + lrs
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i], )) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str

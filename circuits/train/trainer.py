"""
Training loop from Karpathy's minGPT
"""

import time
import os
import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from yacs.config import CfgNode as CN


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 0 # 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0

        C.learning_rate = 5e-4
        C.decay_lr = True
        C.warmup_iters = 1000
        C.lr_decay_iters = 20000
        C.min_lr = 1e-5

        C.micro_batch_size = None

        C.start_token = None
        return C

    def __init__(self, config, model, data_dir):
        self.config = config
        self.model = model
        self.optimizer = None
        # self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.batch_size = config.batch_size
        self.block_size = config.block_size

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.current_lr = config.learning_rate

        if config.micro_batch_size is None:
            self.micro_batch_size = self.batch_size
        else:
            self.micro_batch_size = config.micro_batch_size
        self.num_micro_batches = self.batch_size // self.micro_batch_size
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data

        use_start_token = False
        block_size = self.block_size
        if self.config.start_token is not None:
            use_start_token = True
            block_size -= 1
            start_token = np.array([self.config.start_token])

        ix = torch.randint(len(data) - block_size, (self.micro_batch_size,))
        if use_start_token:
            x = torch.stack([torch.from_numpy(( np.concatenate((start_token, data[i:i+block_size])) ).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i:i+1+block_size]).astype(np.int64)) for i in ix])
        else:
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_lr(self, iter):
        warmup_iters = self.config.warmup_iters
        learning_rate = self.config.learning_rate
        lr_decay_iters = self.config.lr_decay_iters
        min_lr = self.config.min_lr

        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters

        if iter > lr_decay_iters:
            return min_lr

        decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        model.zero_grad(set_to_none=True)
        while True:
            if self.config.decay_lr:
                self.current_lr = self.get_lr(self.iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
            else:
                self.current_lr = self.config.learning_rate
            
            for _ in range(self.num_micro_batches):
                x, y = self.get_batch('train')

                # forward the model
                logits, self.loss = model(x, y)

                self.loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            model.zero_grad(set_to_none=True)

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            x, y = self.get_batch('val')
            logits, self.loss = self.model(x, y)
        self.model.train()
        return self.loss.item()
    

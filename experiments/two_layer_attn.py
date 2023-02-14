import os
import time
from functools import partial

import numpy as np

import torch

import tiktoken

from matplotlib import pyplot as plt

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from circuits.models.two_attn_layer import TwoLayerAttnTransformer
from circuits.train.trainer import Trainer
from circuits.train.utils import set_seed, setup_logging

from yacs.config import CfgNode as CN


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/small_2layer'

    # model
    C.model = TwoLayerAttnTransformer.get_default_config()
    C.model.vocab_size = 50257
    C.model.n_embd = 512#768
    C.model.n_head = 8#12

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.block_size = 256#2048
    C.trainer.batch_size = 32
    # C.trainer.micro_batch_size = 4

    C.trainer.learning_rate = 2e-4
    C.trainer.decay_lr = True
    C.trainer.warmup_iters = 1000
    C.trainer.lr_decay_iters = 20000
    C.trainer.min_lr = 1e-5
    return C


def batch_end_callback(trainer, writer, config):
    if trainer.iter_num % 10 == 0:
        # log training loss
        writer.add_scalar('train_loss', trainer.loss, trainer.iter_num)
        writer.add_scalar('learning_rate', trainer.current_lr, trainer.iter_num)

    if trainer.iter_num % 500 == 0:
        # log validation loss
        val_loss = trainer.validate()
        writer.add_scalar('val_loss', val_loss, trainer.iter_num)
        print(f"iter {trainer.iter_num} val loss: {val_loss}")
    
    if trainer.iter_num % 2000 == 0:
        print('saving latest model at iter', trainer.iter_num)
        # save the latest model
        ckpt_path = os.path.join(config.system.work_dir, f"latest_model_{trainer.iter_num}.pt")
        torch.save(trainer.model.state_dict(), ckpt_path)


def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    config = get_config()
    print(config)

    set_seed(config.system.seed)
    setup_logging(config)

    # new writer for each run based on time
    writer = SummaryWriter(os.path.join(config.system.work_dir, 'tensorboard', time.strftime("%Y-%m-%d_%H-%M-%S")))

    data_dir = os.path.join("../data", "openwebtext")
    if not os.path.exists(data_dir):
        raise ValueError("data not found, please run openwebtext.py")

    # construct the model
    config.model.block_size = config.trainer.block_size
    model = TwoLayerAttnTransformer(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, data_dir=data_dir)

    trainer.add_callback('on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    trainer.run()


if __name__ == '__main__':
    train()

import os
import time
from functools import partial

import numpy as np

import torch

import tiktoken

from torch.utils.tensorboard import SummaryWriter

from circuits.models.zero_layer import ZeroLayerTransformer
from circuits.train.trainer import Trainer
from circuits.train.utils import set_seed, setup_logging

from yacs.config import CfgNode as CN


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/zero_layer_bpe'

    # model
    C.model = ZeroLayerTransformer.get_default_config()
    C.model.n_embd = 128
    C.model.vocab_size = 50257

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 1e-4
    C.trainer.block_size = 128
    return C


def prep_data(data_dir):
    with open('../data/tiny_shakespeare.txt', 'r') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))


def batch_end_callback(trainer, writer, config):
    if trainer.iter_num % 10 == 0:
        # log training loss
        writer.add_scalar('train_loss', trainer.loss, trainer.iter_num)
        # print(f"iter {trainer.iter_num} train loss: {trainer.loss}")

    if trainer.iter_num % 100 == 0:
        # log validation loss
        val_loss = trainer.validate()
        writer.add_scalar('val_loss', val_loss, trainer.iter_num)
        print(f"iter {trainer.iter_num} val loss: {val_loss}")
    
    if trainer.iter_num % 10000 == 0:
        print('saving latest model at iter', trainer.iter_num)
        # save the latest model
        ckpt_path = os.path.join(config.system.work_dir, f"latest_model_{trainer.iter_num}.pt")
        torch.save(trainer.model.state_dict(), ckpt_path)


def train():
    config = get_config()
    print(config)

    set_seed(config.system.seed)
    setup_logging(config)

    # new writer for each run based on time
    writer = SummaryWriter(os.path.join(config.system.work_dir, 'tensorboard', time.strftime("%Y-%m-%d_%H-%M-%S")))

    data_dir = os.path.join("../data", "shakespeare_bpe")
    # prep data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        prep_data(data_dir=data_dir)

    # construct the model
    config.model.block_size = config.trainer.block_size
    model = ZeroLayerTransformer(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, data_dir=data_dir)

    trainer.add_callback('on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    trainer.run()


if __name__=="__main__":
    train()

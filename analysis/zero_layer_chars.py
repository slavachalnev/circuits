# TODO: the training section needs to be moved to the train folder.

import os
import time
import pickle
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
    C.system.work_dir = './out/zero_layer_chars'

    # model
    C.model = ZeroLayerTransformer.get_default_config()
    C.model.n_embd = 16

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 1e-4
    C.trainer.block_size = 128
    return C


def prep_data(data_dir):
    with open('../data/tiny_shakespeare.txt', 'r') as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    train_ids = [stoi[c] for c in train_data]
    val_ids = [stoi[c] for c in val_data]

    # save the data
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)


def batch_end_callback(trainer, writer, config):
    if trainer.iter_num % 10 == 0:
        # log training loss
        writer.add_scalar('train_loss', trainer.loss, trainer.iter_num)
    
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

    data_dir = os.path.join("../data", "shakespeare_chars")
    # prep data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        prep_data(data_dir=data_dir)
    
    # load dataset metadata
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta = pickle.load(open(meta_path, 'rb'))

    # construct the model
    config.model.vocab_size = meta['vocab_size']
    config.model.block_size = config.trainer.block_size
    model = ZeroLayerTransformer(config.model)

    # construct the trainer
    trainer = Trainer(config.trainer, model, data_dir=data_dir)

    trainer.add_callback(
        'on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    trainer.run()
    writer.close()


def compute_ngram_counts(data):
    ngram_counts = {}
    prev = None
    for char in data:
        if prev is not None:
            ngram = (prev, char)
            if ngram in ngram_counts:
                ngram_counts[ngram] += 1
            else:
                ngram_counts[ngram] = 1
        prev = char
    return ngram_counts


def analyse(model_paths, data_dir):
    # load dataset
    meta = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    actual_ngram_counts = compute_ngram_counts(train_data)

    # fill out the actual ngram matrix
    vocab_size = meta['vocab_size']
    actual_ngram_matrix = np.zeros((vocab_size, vocab_size))
    for i in range(vocab_size):
        for j in range(vocab_size):
            if (i, j) in actual_ngram_counts:
                actual_ngram_matrix[i,j] = actual_ngram_counts[(i, j)]
    
    # normalise the rows
    actual_ngram_matrix = actual_ngram_matrix / np.sum(actual_ngram_matrix, axis=1, keepdims=True)

    kl_divergences = []
    for model_path in model_paths:
        # inspect model weights
        state_dict = torch.load(model_path)
        embedding_weights = state_dict['embedding.weight']
        output_weights = state_dict['unembedding.weight']

        ngram_logits = torch.matmul(embedding_weights, output_weights.T)
        ngram_probs = torch.softmax(ngram_logits, dim=1).to('cpu').numpy()

        # compute the KL divergence
        actual_ngram_matrix[actual_ngram_matrix == 0] = 1e-10 # avoid log(0)
        kl_divergence = np.sum(actual_ngram_matrix * np.log(actual_ngram_matrix / ngram_probs))

        print('KL divergence from predicted to actual probs:', kl_divergence)
        kl_divergences.append(kl_divergence)
    
    plt.plot(kl_divergences)
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.set_defaults(train=False)
    args = parser.parse_args()

    if args.train:
        train()
    else:
        models_dir = "./out/zero_layer_chars"
        model_paths = [os.path.join(models_dir, f"latest_model_{i}.pt") for i in range(0, 70000, 10000)]
        analyse(model_paths=model_paths, data_dir="../data/shakespeare_chars")
    
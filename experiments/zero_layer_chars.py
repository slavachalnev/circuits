import os
import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# for cli args
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
    return C


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 32
        return C

    def __init__(self, data, config=None):
        if config is None:
            config = self.get_default_config()
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


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

    # validate on the first 1000 characters, train on the rest
    train_text = open('../data/tiny_shakespeare.txt', 'r').read()
    train_dataset = CharDataset(data=train_text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = ZeroLayerTransformer(config.model)

    # construct the trainer
    trainer = Trainer(config.trainer, model, train_dataset)

    trainer.add_callback(
        'on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    trainer.run()
    writer.close()


def compute_ngram_counts(document):
    ngram_counts = {}
    prev = None
    for char in document:
        if prev is not None:
            ngram = prev + char
            if ngram in ngram_counts:
                ngram_counts[ngram] += 1
            else:
                ngram_counts[ngram] = 1
        prev = char
    return ngram_counts


def analyse(model_paths):

    # load dataset
    with open('../data/tiny_shakespeare.txt', 'r') as f:
        data = f.read()
    dataset = CharDataset(data=data)
    actual_ngram_counts = compute_ngram_counts(data)

    # fill out the actual ngram matrix
    actual_ngram_matrix = np.zeros((dataset.vocab_size, dataset.vocab_size))
    for i in range(dataset.vocab_size):
        for j in range(dataset.vocab_size):
            ngram = dataset.itos[i] + dataset.itos[j]
            if ngram in actual_ngram_counts:
                actual_ngram_matrix[i,j] = actual_ngram_counts[ngram]
    
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
        model_paths = [os.path.join(models_dir, f"latest_model_{i}.pt") for i in range(0, 170000, 10000)]
        analyse(model_paths=model_paths)
    
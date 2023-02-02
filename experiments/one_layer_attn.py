import os
import time
from functools import partial

import numpy as np

import torch

import tiktoken

from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from circuits.models.one_attn_layer import OneLayerAttnTransformer
from circuits.train.trainer import Trainer
from circuits.train.utils import set_seed, setup_logging

from yacs.config import CfgNode as CN


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/one_layer_openwebtext_big'

    # model
    C.model = OneLayerAttnTransformer.get_default_config()
    C.model.vocab_size = 50257
    C.model.n_embd = 768
    C.model.n_head = 12

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.block_size = 2048
    C.trainer.batch_size = 32
    C.trainer.micro_batch_size = 4

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
    model = OneLayerAttnTransformer(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, data_dir=data_dir)

    trainer.add_callback('on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    trainer.run()


def get_weights_for_head(weights, head, n_heads, d_model, norm_emb=True):
    """ Get the weights for a single head. """
    d_head = d_model // n_heads

    w_v = weights['attn.attn.in_proj_weight'][2*d_model:]
    w_o = weights['attn.attn.out_proj.weight'] 

    w_v_h = w_v[head*d_head: (head+1)*d_head, :]
    w_o_h = w_o[:, head*d_head: (head+1)*d_head]


    ln1w = weights['attn.ln.weight'].unsqueeze(1).numpy()
    ln1b = weights['attn.ln.bias'].unsqueeze(1).numpy()

    w_e = weights['embedding.weight'].numpy().T
    if norm_emb:
        w_e = (w_e - np.average(w_e)) / np.std(w_e)
        w_e = w_e * ln1w + ln1b

    return {
        'w_e': w_e,
        'w_v': w_v_h.numpy(),
        'w_o': w_o_h.numpy(),
        'w_u': weights['unembedding.weight'].numpy(),
        'ln1w': ln1w,
        'ln1b': ln1b,
        'lnfw': weights['ln_f.weight'].unsqueeze(1).numpy(),
        'lnfb': weights['ln_f.bias'].unsqueeze(1).numpy(),
    }


def get_eigenvalues(weights, head, n_heads, d_model):
    """ Get the eigenvalues for the w_v @ w_e @ w_u @ w_o matrix. """
    w = get_weights_for_head(weights, head, n_heads, d_model)
    m = w['w_v'] @ w['w_e'] @ w['w_u'] @ w['w_o']
    return np.linalg.eigvals(m)


def source_to_out(source, tokenizer, weights, d_model, n_heads, head):
    """ OV circuit for a single head. """
    d_head = d_model // n_heads

    tok = tokenizer.encode(source)
    if len(tok) > 1:
        raise ValueError("source must be a single token")
    
    h_w = get_weights_for_head(weights, head, n_heads, d_model)

    x = h_w['w_e'][:, tok]

    # x = (x - np.average(x)) / np.std(x)
    # x = x*h_w['ln1w'] + h_w['ln1b']
    v = h_w['w_v'] @ x
    o = h_w['w_o'] @ v
    o = (o - np.average(o)) / np.std(o)
    o = o*h_w['lnfw'] + h_w['lnfb']  # ???
    y = h_w['w_u'] @ o

    torch_y = torch.from_numpy(y).squeeze(1)
    top = torch.topk(torch_y, 5)
    print(tokenizer.decode_tokens_bytes(top.indices.tolist()))
    print(top.values.tolist())


if __name__=="__main__":
    # train()

    enc = tiktoken.get_encoding("gpt2")
    weights = torch.load("out/from_odin/big_22000.pt", map_location='cpu')
    # weights = torch.load("out/from_odin/small_8000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    # # construct the model
    # config.model.block_size = config.trainer.block_size
    # model = OneLayerAttnTransformer(config.model)
    # model.load_state_dict(weights)

    # idxs = enc.encode(" hello")
    # in_batch = torch.tensor(idxs).unsqueeze(0)
    # generated = model.generate(in_batch, max_new_tokens=10)
    # print(enc.decode_tokens_bytes(generated[0].tolist()))


    graphs = []
    for head in range(n_heads):
        eigen = get_eigenvalues(weights=weights, head=head, n_heads=n_heads, d_model=d_model)
        xs = eigen.real
        ys = eigen.imag
        graphs.append((xs, ys))

    n_rows = n_heads // 2
    fig, ax = plt.subplots(2, n_rows, subplot_kw={'projection': 'polar'})
    for i, (xs, ys) in enumerate(graphs):
        axis = ax[i//n_rows, i%n_rows]
        axis.scatter(np.angle(xs + 1j*ys), np.log(np.abs(xs + 1j*ys)))
        axis.set_xticks([])
        # axis.set_yticks([])
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.set_title('')
    fig.tight_layout()
    plt.show()


    # for h in range(n_heads):
    #     print("head", h)
    #     source_to_out(
    #         " perfect",
    #         tokenizer=enc,
    #         weights=weights,
    #         d_model=d_model,
    #         n_heads=n_heads,
    #         head=h,
    #     )

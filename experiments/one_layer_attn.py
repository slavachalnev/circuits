import os
import time
from functools import partial

import numpy as np

import torch

import tiktoken

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
    C.model.n_embd =512 #768
    C.model.n_head =8 #12

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.block_size = 256#2048
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


def source_to_out_token(source, tokenizer, weights, d_model, n_heads):
    """ OV circuit. Computes all heads simultaneously. """
    d_head = d_model // n_heads

    tok = tokenizer.encode(source)
    if len(tok) > 1:
        raise ValueError("source must be a single token")
    
    x = weights['embedding.weight'][tok]

    w_v = weights['attn.attn.in_proj_weight'][2*d_model:] 
    w_o = weights['attn.attn.out_proj.weight'] # shape (n_heads*d_model/n_heads, d_model)

    v = torch.matmul(x, w_v)

    outs = []
    for h in range(n_heads):
        v_h = v[:, h*d_head: (h+1)*d_head]
        w_o_h = w_o[h*d_head: (h+1)*d_head, :]
        o = torch.matmul(v_h, w_o_h)
        outs.append(o.squeeze(0))
    
    out = torch.stack(outs, dim=0)

    # unembed!
    w_u = weights['unembedding.weight']
    out_tokens = torch.matmul(out, w_u.T)

    top = torch.topk(out_tokens, 5, dim=1)
    top_tokens = top.indices

    # decoder needs list of int
    tokens = [h.tolist() for h in top_tokens]

    for i, h in enumerate(tokens):
        print(tokenizer.decode_tokens_bytes(h))
        print(top.values[i].tolist())
        print()


if __name__=="__main__":
    # train()

    enc = tiktoken.get_encoding("gpt2")
    weights = torch.load("out/from_odin/small_8000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    # # construct the model
    # config.model.block_size = config.trainer.block_size
    # model = OneLayerAttnTransformer(config.model)
    # model.load_state_dict(weights)

    # idxs = enc.encode("Hello, my name is")
    # in_batch = torch.tensor(idxs).unsqueeze(0)
    # generated = model.generate(in_batch, max_new_tokens=10)
    # print(enc.decode_tokens_bytes(generated[0].tolist()))



    with torch.no_grad():
        source_to_out_token(
            " github",
            tokenizer=enc,
            weights=weights,
            d_model=d_model,
            n_heads=n_heads,
        )

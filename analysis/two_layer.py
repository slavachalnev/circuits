import numpy as np
import torch
import tiktoken

from circuits.train.train_two_layer import get_config
from analysis.utils import get_weights_for_head


if __name__=='__main__':
    enc = tiktoken.get_encoding("gpt2")

    weights = torch.load("../from_odin/big_2layer_28000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    for h in range(n_heads):
        h_w = get_weights_for_head(weights, layer=1, head=h, n_heads=n_heads, d_model=d_model)
        print('hello')

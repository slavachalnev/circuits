import numpy as np
import torch
import tiktoken

import matplotlib.pyplot as plt

from circuits.train.train_two_layer import get_config
from analysis.utils import get_weights_for_head, positional_attention_for_head


def k_composition(h_0, h_1):
    """
    Detects k-composition between two heads.

    It looks at the Frobenius norm of the product divided by the norm of the
    individual heads.
    """
    w_ov = h_0['w_o'] @ h_0['w_v']
    w_qk = h_1['w_q'].T @ h_1['w_k']

    f_qkov = np.linalg.norm(w_qk @ w_ov)
    f_qk = np.linalg.norm(w_qk)
    f_ov = np.linalg.norm(w_ov)

    return f_qkov / (f_qk * f_ov)


if __name__=='__main__':
    enc = tiktoken.get_encoding("gpt2")

    weights = torch.load("../from_odin/big_2layer_28000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    for h in range(n_heads):
        print()
        print('layer 0, head', h)
        h_w = get_weights_for_head(weights, layer=1, head=h, n_heads=n_heads, d_model=d_model)
        positional_attention_for_head(h_w, plot=False)

    compositionality = np.zeros((n_heads, n_heads))
    for h0 in range(n_heads):
        for h1 in range(n_heads):
            h_w_0 = get_weights_for_head(weights, layer=1, head=h0, n_heads=n_heads, d_model=d_model)
            h_w_1 = get_weights_for_head(weights, layer=2, head=h1, n_heads=n_heads, d_model=d_model)
            compositionality[h0, h1] = k_composition(h_0=h_w_0, h_1=h_w_1)

    print(compositionality)

    # plot the compositionality matrix as a heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(compositionality)
    plt.show()

    

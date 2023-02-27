import numpy as np
import torch
import tiktoken

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def q_composition(h_0, h_1):
    """ Detects q-composition between two heads. """
    w_ov = h_0['w_o'] @ h_0['w_v']
    w_qk = h_1['w_q'].T @ h_1['w_k']

    f_qkov = np.linalg.norm(w_qk.T @ w_ov)
    f_qk = np.linalg.norm(w_qk)
    f_ov = np.linalg.norm(w_ov)

    return f_qkov / (f_qk * f_ov)

def v_composition(h_0, h_1):
    """ Detects v-composition between two heads. """
    w_ov_0 = h_0['w_o'] @ h_0['w_v']
    w_ov_1 = h_1['w_o'] @ h_1['w_v']

    f_ovov = np.linalg.norm(w_ov_0 @ w_ov_1)
    f_ov_0 = np.linalg.norm(w_ov_0)
    f_ov_1 = np.linalg.norm(w_ov_1)

    return f_ovov / (f_ov_0 * f_ov_1)


if __name__=='__main__':
    enc = tiktoken.get_encoding("gpt2")

    weights = torch.load("../from_odin/big_2layer_long_108000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    pos = []
    for h in range(n_heads):
        print()
        print('layer 0, head', h)
        h_w = get_weights_for_head(weights, layer=0, head=h,
            n_heads=n_heads, d_model=d_model, apply_layernorm=False)
        avg = positional_attention_for_head(h_w, plot=False)

        pos.append(avg[1]) # previous token

    k_comp = np.zeros((n_heads, n_heads))
    q_comp = np.zeros((n_heads, n_heads))
    v_comp = np.zeros((n_heads, n_heads))
    for h0 in range(n_heads):
        for h1 in range(n_heads):
            h_w_0 = get_weights_for_head(weights, layer=0, head=h0, n_heads=n_heads, d_model=d_model)
            h_w_1 = get_weights_for_head(weights, layer=1, head=h1, n_heads=n_heads, d_model=d_model)
            k_comp[h0, h1] = k_composition(h_0=h_w_0, h_1=h_w_1)
            q_comp[h0, h1] = q_composition(h_0=h_w_0, h_1=h_w_1)
            v_comp[h0, h1] = v_composition(h_0=h_w_0, h_1=h_w_1)

    # print(k_comp)

    vmin = min(q_comp.min(), k_comp.min(), v_comp.min())
    vmax = max(q_comp.max(), k_comp.max(), v_comp.max())

    # fig = plt.figure(figsize=(40, 5))
    # gs = GridSpec(1, 4, width_ratios=[10, 10, 10, 4])
    fig = plt.figure(figsize=(30, 5))
    gs = GridSpec(1, 4, width_ratios=[12, 12, 12, 1])

    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(q_comp, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_ylabel('Layer 0 Heads')
    ax1.set_xlabel('Layer 1 Heads')
    ax1.set_title('Q-Composition')

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(k_comp, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Layer 0 Heads')
    ax2.set_xlabel('Layer 1 Heads')
    ax2.set_title('K-Composition')

    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(v_comp, cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_ylabel('Layer 0 Heads')
    ax3.set_xlabel('Layer 1 Heads')
    ax3.set_title('V-Composition')

    # Add the subplot on the right for the nx1 matrix
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(np.array(pos).reshape(-1, 1), cmap='viridis')#, aspect='auto')
    ax4.set_ylabel('Layer 0 Heads')
    ax4.set_title('prev tok heads')

    ax4.set_xticks([])
    # ax2.set_yticks([])

    # Show the plot
    plt.show()

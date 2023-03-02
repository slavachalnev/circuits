import numpy as np
import torch
import tiktoken

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from circuits.train.train_two_layer import get_config
from analysis.utils import get_weights_for_head, positional_attention_for_head, head_forward_pass


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

def compute_qkv_composition(weights, n_heads, d_model):
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


def get_attention(weights, tokens, h_heads, d_model):
    """ Forward pass, return attention and value norms. """

    tokens = [50257] + tokens  # add start token

    layer_0 = []
    for h in range(h_heads):
        layer_0.append(get_weights_for_head(weights,
                                            layer=0,
                                            head=h,
                                            n_heads=h_heads,
                                            d_model=d_model,
                                            apply_layernorm=False,
                                            ))
    layer_1 = []
    for h in range(h_heads):
        layer_1.append(get_weights_for_head(weights,
                                            layer=1,
                                            head=h,
                                            n_heads=h_heads,
                                            d_model=d_model,
                                            apply_layernorm=False,
                                            ))
    
    x = weights['embedding.weight'].numpy()[tokens, :]

    x_ln = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
    x_ln = x_ln * weights['b0.ln.weight'].numpy() + weights['b0.ln.bias'].numpy()

    layer_0_res = []
    for h in range(h_heads):
        layer_0_res.append(head_forward_pass(x_ln, layer_0[h]))
    
    for h in range(h_heads):
        x += layer_0_res[h][0]
    
    x_ln = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
    x_ln = x_ln * weights['b1.ln.weight'].numpy() + weights['b1.ln.bias'].numpy()

    layer_1_res = []
    for h in range(h_heads):
        layer_1_res.append(head_forward_pass(x_ln, layer_1[h]))
    
    for h in range(h_heads):
        x += layer_1_res[h][0]
    
    attention = [out[1] for out in layer_0_res] + [out[1] for out in layer_1_res]
    attention = np.array(attention)
    attention = np.transpose(attention, (1, 2, 0))

    value_norms = [out[2] for out in layer_0_res] + [out[2] for out in layer_1_res]
    value_norms = [np.linalg.norm(v, axis=-1) for v in value_norms]
    value_norms = np.array(value_norms)
    value_norms = value_norms.T

    weighted_attn = np.zeros_like(attention)
    for i in range(attention.shape[0]):
        weighted_attn[i] = attention[i] * value_norms
        weighted_attn[i] /= np.max(weighted_attn[i])

    return weighted_attn


def plot_attention_on_text(encoder, weights, n_heads, d_model):
    """ Plot attention on text. """

    # requires https://github.com/anthropics/PySvelte
    import pysvelte

    toks = encoder.encode(
        "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say " + \
        "that they were perfectly normal, thank you very much. They were the " + \
        "last people you'd expect to be involved in anything strange or " + \
        "mysterious, because they just didn't hold with such nonsense. Mr Dursley " + \
        "was the director of a firm called Grunnings, which made drills. He was a " + \
        "big, beefy man with hardly any neck, although he did have a very large " + \
        "moustache. Mrs Dursley was thin and blonde"
    )
    words = ["<START>"] + [encoder.decode([t]) for t in toks]
    attn = get_attention(weights, toks, n_heads, d_model)

    pysvelte.AttentionMulti(tokens=words, attention=attn,
                            head_labels=['0:0', '0:1', '0:2', '0:3', '0:4', '0:5',
                                         '0:6', '0:7', '0:8', '0:9', '0:10', '0:11',
                                         '1:0', '1:1', '1:2', '1:3', '1:4', '1:5',
                                         '1:6', '1:7', '1:8', '1:9', '1:10', '1:11'],
                            ).publish("./potter.html")


if __name__=='__main__':
    enc = tiktoken.get_encoding("gpt2")

    weights = torch.load("../from_odin/big_2layer_long_108000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    compute_qkv_composition(weights, n_heads, d_model)
    # plot_attention_on_text(encoder=enc, weights=weights, n_heads=n_heads, d_model=d_model)

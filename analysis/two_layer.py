import numpy as np
import torch
import tiktoken

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from circuits.train.train_two_layer import get_config
from circuits.train.train_one_layer import Trainer
from circuits.models.two_attn_layer import TwoLayerAttnTransformer
from circuits.train.utils import set_seed
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


def save_attn_patterns(model, x):
    # saving qk values is equivalent to saving attn pattern
    x = model.embedding(x)

    d0 = model.b0(x)
    attn0 = d0['qk']

    d1 = model.b1(d0['res'])
    attn1 = d1['qk']

    return attn0, attn1


def run_with_fixed_attn(model, attns, x, targets, prev_vals=None):
    x = model.embedding(x)

    if prev_vals is None:
        print('prev vals is None')
        d0 = model.b0(x, qk=attns[0], add_to_res=False)
        # x = x + torch.zeros_like(d0['res'])
        d1 = model.b1(x, qk=attns[1], add_to_res=False)
        # x = torch.zeros_like(d1['res'])
    else:
        d0 = model.b0(x, qk=attns[0], add_to_res=False)
        x = x + prev_vals[0]
        d1 = model.b1(x, qk=attns[1], add_to_res=False)
        x = x + prev_vals[1]

    x = model.ln_f(x)

    logits = model.unembedding(x)
    loss = model.loss(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss, [d0['h_out'], d1['h_out']]


def marginal_loss_reduction(weights, config):
    """
    Algorithm for measuring marginal loss reduction of Nth order terms
    """
    # initialise model
    model = TwoLayerAttnTransformer(config=config.model)
    model.load_state_dict(weights)
    model.eval()

    # initialise trainer
    trainer = Trainer(config=config.trainer, model=model, data_dir="../data/openwebtext")
    x, y = trainer.get_batch(split='val')
    logits, loss = model(x, y)
    # print('loss is ', loss.item())

    with torch.no_grad():
        a1, a2 = save_attn_patterns(model, x)
    print(a1.shape, a2.shape)

    # uniform distribution loss
    with torch.no_grad():
        uniform_logits = torch.ones_like(logits)
        print('uniform loss ', model.loss(uniform_logits.view(-1, uniform_logits.size(-1)), y.view(-1)).item())
    
    average_loss = {0: 0, 1: 0, 2: 0}
    n_samples = 5
    for sample in range(n_samples):
        print('sample is ', sample)
        x, y = trainer.get_batch(split='val')
        print(x[:5])
        vals = None
        for order in range(3):
            print('order is ', order)
            with torch.no_grad():
                logits, loss, vals = run_with_fixed_attn(model, [a1, a2], x, y, prev_vals=vals)
            print('loss is ', loss.item())
            average_loss[order] += loss.item()

    for order in range(3):
        print('average loss for order ', order, ' is ', average_loss[order] / n_samples)





if __name__=='__main__':
    enc = tiktoken.get_encoding("gpt2")

    weights = torch.load("../from_odin/big_2layer_long_108000.pt", map_location='cpu')

    # for weight in weights:
    #     print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd
    config.model.block_size = config.trainer.block_size
    # set_seed(config.system.seed)

    # compute_qkv_composition(weights, n_heads, d_model)

    # term importance analysis
    marginal_loss_reduction(weights, config)

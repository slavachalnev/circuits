import numpy as np
import torch
import matplotlib.pyplot as plt


def get_subtract_avg_matrix(dim):
    """
    Get a matrix M such that M @ x is the same as x - avg(x).
    Which is the same as zeroing out the diagonal of x.
    """
    # z zeros out diagonal
    z = np.eye(dim) - np.ones((dim, dim)) / dim
    return z


def get_weights_for_head(weights, layer, head, n_heads, d_model):
    """ Get the weights for a single head. """
    d_head = d_model // n_heads

    w_v = weights[f'b{layer}.attn.in_proj_weight'][2*d_model:]
    w_o = weights[f'b{layer}.attn.out_proj.weight'] 

    w_v_h = w_v[head*d_head: (head+1)*d_head, :]
    w_o_h = w_o[:, head*d_head: (head+1)*d_head]

    w_q = weights[f'b{layer}.attn.in_proj_weight'][:d_model]
    w_k = weights[f'b{layer}.attn.in_proj_weight'][d_model:2*d_model]

    w_q_h = w_q[head*d_head: (head+1)*d_head, :]
    w_k_h = w_k[head*d_head: (head+1)*d_head, :]

    lnw = weights[f'b{layer}.ln.weight'].unsqueeze(1).numpy()
    lnb = weights[f'b{layer}.ln.bias'].unsqueeze(1).numpy()

    p_e = weights[f'b{layer}.pos.pe'].numpy()

    # roll layernorm into w_q, w_k, w_v
    M = get_subtract_avg_matrix(d_model)
    w_q_h = w_q_h @ M
    w_q_h = w_q_h * lnw.T

    w_k_h = w_k_h @ M
    w_k_h = w_k_h * lnw.T

    w_v_h = w_v_h @ M
    w_v_h = w_v_h * lnw.T

    return {
        # 'w_e': w_e,
        'w_v': w_v_h.numpy(),
        'w_o': w_o_h.numpy(),
        # 'w_u': w_u,
        # 'lnfw': lnfw,
        # 'lnfb': lnfb,
        'lnw': lnw,
        'lnb': lnb,
        'w_q': w_q_h.numpy(),
        'w_k': w_k_h.numpy(),
        'p_e': p_e,
    }

# def get_embedding_weights(weights):
#     """ Get the embedding weights. """

#     w_e = weights['embedding.weight'].numpy().T
#     if norm_emb:
#         w_e = (w_e - np.average(w_e, axis=0, keepdims=True)) / np.std(w_e, axis=0, keepdims=True)
#         w_e = w_e * ln1w + ln1b

#     return {
#         'w_e': w_e,
#         'w_u': w_u,
#     }


def positional_attention_for_head(head_weights, plot=False):
    """ compute matrix of preferred relative positions. """
    p_e = head_weights['p_e']
    qk = head_weights['w_q'].T @ head_weights['w_k']

    res = p_e @ qk @ p_e.T

    # mask with zeros
    mask = np.triu(np.ones_like(res), k=1)
    res = res * (1 - mask)

    # to torch, apply softmax, and convert back to numpy
    res = torch.from_numpy(res)
    res = torch.softmax(res, dim=0)
    res = res.numpy()

    # the higher the value, the more the head attends to positions.
    diag_averages = []
    n = res.shape[0]
    for i in range(n):
        diag_averages.append(np.trace(res, offset=-i)/ (n - i*0.99))
    print('positional max: ', np.max(diag_averages))
    print('positional argmax:', np.argmax(diag_averages))
    print('diagonal averages:', diag_averages[:5])

    if plot:
        # plot heatmap of res
        plt.imshow(res)
        plt.show()


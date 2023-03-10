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


def get_weights_for_head(weights, layer, head, n_heads, d_model, apply_layernorm=True):
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

    if apply_layernorm:
        # roll layernorm into w_q, w_k, w_v
        M = get_subtract_avg_matrix(d_model)
        w_q_h = w_q_h @ M
        w_q_h = w_q_h * lnw.T

        w_k_h = w_k_h @ M
        w_k_h = w_k_h * lnw.T

        w_v_h = w_v_h @ M
        w_v_h = w_v_h * lnw.T

    return {
        'w_v': w_v_h.numpy(),
        'w_o': w_o_h.numpy(),
        'lnw': lnw,
        'lnb': lnb,
        'w_q': w_q_h.numpy(),
        'w_k': w_k_h.numpy(),
        'p_e': p_e,
    }

def get_embedding_weights(weights, d_model, norm_emb=False, final_layernorm=True):
    """ Get the embedding weights. """

    w_e = weights['embedding.weight'].numpy().T
    if norm_emb:
        lnw = weights['b0.ln.weight'].unsqueeze(1).numpy()
        lnb = weights['b0.ln.bias'].unsqueeze(1).numpy()
        w_e = (w_e - np.average(w_e, axis=0, keepdims=True)) / np.std(w_e, axis=0, keepdims=True)
        w_e = w_e * lnw + lnb
    
    lnfw = weights['ln_f.weight'].unsqueeze(1).numpy()
    lnfb = weights['ln_f.bias'].unsqueeze(1).numpy()

    w_u = weights['unembedding.weight'].numpy()
    if final_layernorm:
        # Roll final layernorm into the unembedding matrix.
        # first we subtract the mean by zeroing out the diagonal dimension.
        M = get_subtract_avg_matrix(d_model)
        w_u = w_u @ M
        # multiply by the layer norm weights
        w_u = w_u * lnfw.T

    return {
        'w_e': w_e,
        'w_u': w_u,
        'lnfw': lnfw,
        'lnfb': lnfb,
    }

def get_ov_eigenvalues(wh, we):
    """
    Get the eigenvalues for the w_u @ w_o @ w_v @ w_e matrix. Equivalent to
    the eigenvalues of the w_v @ w_e @ w_u @ w_o matrix.
    """
    m = wh['w_v'] @ we['w_e'] @ we['w_u'] @ wh['w_o']
    return np.linalg.eigvals(m)


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
    
    return diag_averages


def head_forward_pass(x, weights):
    # x shape is (seq_len, d_model)
    # weights is a dict of weights for the head

    x_pos = x + weights['p_e'][:x.shape[0], :]

    q = weights['w_q'] @ x_pos.T  # q shape is (d_head, seq_len)
    k = weights['w_k'] @ x_pos.T  # k shape is (d_head, seq_len)

    v = weights['w_v'] @ x.T  # v shape is (d_head, seq_len)

    # compute attention
    a = q.T @ k  # a shape is (seq_len, seq_len)
    a = a / np.sqrt(q.shape[0])

    infs = np.full(a.shape, -np.inf)
    mask = np.triu(infs, k=1)
    a = a + mask

    # to torch, softmax, back to numpy haha
    a = torch.from_numpy(a)
    a = torch.softmax(a, dim=1)
    a = a.numpy()

    # compute output
    o = a @ v.T  # o shape is (seq_len, d_head)
    out = o @ weights['w_o'].T  # out shape is (seq_len, d_model)

    return out, a, v.T

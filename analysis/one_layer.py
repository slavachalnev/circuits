import numpy as np
import torch
import tiktoken

from matplotlib import pyplot as plt
from tqdm import tqdm

from circuits.models.one_attn_layer import OneLayerAttnTransformer
from circuits.train.train_one_layer import get_config


def get_weights_for_head(weights, head, n_heads, d_model,
                         norm_emb=True, final_ln=True):
    """ Get the weights for a single head. """
    d_head = d_model // n_heads

    w_v = weights['attn.attn.in_proj_weight'][2*d_model:]
    w_o = weights['attn.attn.out_proj.weight'] 

    w_v_h = w_v[head*d_head: (head+1)*d_head, :]
    w_o_h = w_o[:, head*d_head: (head+1)*d_head]

    w_q = weights['attn.attn.in_proj_weight'][:d_model]
    w_k = weights['attn.attn.in_proj_weight'][d_model:2*d_model]

    w_q_h = w_q[head*d_head: (head+1)*d_head, :]
    w_k_h = w_k[head*d_head: (head+1)*d_head, :]

    ln1w = weights['attn.ln.weight'].unsqueeze(1).numpy()
    ln1b = weights['attn.ln.bias'].unsqueeze(1).numpy()

    w_e = weights['embedding.weight'].numpy().T
    if norm_emb:
        w_e = (w_e - np.average(w_e, axis=0, keepdims=True)) / np.std(w_e, axis=0, keepdims=True)
        w_e = w_e * ln1w + ln1b

    w_u = weights['unembedding.weight'].numpy()
    if final_ln:
        lnfw = weights['ln_f.weight'].unsqueeze(1).numpy()
        lnfb = weights['ln_f.bias'].unsqueeze(1).numpy()
    # if final_ln:
    #     raise NotImplementedError
    #     # The idea is to roll normalization into the unembedding matrix.
    #     # first we subtract the mean by zeroing out the diagonal dimension.
    #     pass
    #     # w_u = w_u @ weights['ln.weight'].numpy().T
    #     # w_u = w_u + weights['ln.bias'].numpy()
    
    p_e = weights['attn.pos.pe'].numpy()

    return {
        'w_e': w_e,
        'w_v': w_v_h.numpy(),
        'w_o': w_o_h.numpy(),
        'w_u': w_u,
        'lnfw': lnfw,
        'lnfb': lnfb,
        'ln1w': ln1w,
        'ln1b': ln1b,
        'w_q': w_q_h.numpy(),
        'w_k': w_k_h.numpy(),
        'p_e': p_e,
    }


def get_eigenvalues(weights, head, n_heads, d_model):
    """ Get the eigenvalues for the w_v @ w_e @ w_u @ w_o matrix. """
    w = get_weights_for_head(weights, head, n_heads, d_model)
    m = w['w_v'] @ w['w_e'] @ w['w_u'] @ w['w_o']
    return np.linalg.eigvals(m)


def source_to_out(source, tokenizer, head_weights):
    """ OV circuit for a single head. """
    tok = tokenizer.encode(source)
    if len(tok) > 1:
        raise ValueError("source must be a single token")
    
    x = head_weights['w_e'][:, tok]

    v = head_weights['w_v'] @ x
    o = head_weights['w_o'] @ v
    y = head_weights['w_u'] @ o

    torch_y = torch.from_numpy(y).squeeze(1)
    top = torch.topk(torch_y, 5)
    print(tokenizer.decode_tokens_bytes(top.indices.tolist()))
    print(top.values.tolist())


def source_to_dest(source, tokenizer, head_weights, head):
    """ QK circuit for a single head. """
    tok = tokenizer.encode(source)
    if len(tok) > 1:
        raise ValueError("source must be a single token")

    x = head_weights['w_e'][:, tok]
    k = head_weights['w_k'] @ x
    kq = head_weights['w_q'].T @ k
    dst = head_weights['w_e'].T @ kq

    qk_averages = np.load(f'qk_big_nolnf_nobias/head_{head}.npy')
    # subtract the average qk value for each query
    dst = dst.squeeze(1) - np.array(qk_averages)

    # reweight by token frequency
    freq = np.load('openwebtext_gpt2_averages.npy')
    dst = dst * (freq**0.1)

    tdst = torch.from_numpy(dst)
    top = torch.topk(tdst, 5)
    print(tokenizer.decode_tokens_bytes(top.indices.tolist()))
    print(top.values.tolist())

def save_qk_averages_for_head(head_weights, head):
    """ Compute and save the average qk value for each query. """
    qk_averages = []
    for i in tqdm(range(head_weights['w_e'].shape[1])):
        x = head_weights['w_e'][:, i]
        q = head_weights['w_q'] @ x
        qk = head_weights['w_k'].T @ q
        src = head_weights['w_e'].T @ qk
        qk_averages.append(src.mean())
    np.save(f"qk_avgs/head_{head}", np.array(qk_averages))

def positional_attention_for_head(head_weights, plot=False):
    """ compute matrix of preferred relative positions. """
    p_e = head_weights['p_e']
    qk = head_weights['w_q'].T @ head_weights['w_k']

    cut_p_e = p_e
    res = cut_p_e @ qk.T @ cut_p_e.T

    # the higher the value, the more the head attends to positions.
    print('positional:', res.max() - res.min())

    if plot:
        # plot heatmap of res
        plt.imshow(res)
        plt.show()


if __name__=="__main__":
    enc = tiktoken.get_encoding("gpt2")

    # weights = torch.load("../from_odin/small_yeslnf_26000.pt", map_location='cpu')
    weights = torch.load("../from_odin/big_yeslnf_22000.pt", map_location='cpu')

    for weight in weights:
        print(weight, weights[weight].shape)

    config = get_config()
    n_heads = config.model.n_head
    d_model = config.model.n_embd

    # # compute average qk values for each head.
    # for h in range(n_heads):
    #     h_w = get_weights_for_head(weights, h, n_heads, d_model)
    #     save_qk_averages_for_head(h_w, h)

    # # construct a model and generate some text
    # config.model.block_size = config.trainer.block_size
    # model = OneLayerAttnTransformer(config.model)
    # model.load_state_dict(weights)
    # idxs = enc.encode(" hello there. general")
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
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.set_title('')
    fig.tight_layout()
    plt.show()

    print()
    word = " couldn"
    print('word:', word)

    for h in range(n_heads):
        h_w = get_weights_for_head(weights, h, n_heads, d_model)

        print()
        print("head", h)

        # positional attention for head
        positional_attention_for_head(h_w)

        print("source to out")
        source_to_out(
            word,
            tokenizer=enc,
            head_weights=h_w,
        )
        print("source to dest")
        source_to_dest(source=word, tokenizer=enc, head_weights=h_w, head=h)


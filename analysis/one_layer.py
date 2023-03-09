import numpy as np
import torch
import tiktoken

from matplotlib import pyplot as plt
from tqdm import tqdm

from circuits.models.one_attn_layer import OneLayerAttnTransformer
from circuits.train.train_one_layer import get_config
from utils import positional_attention_for_head, get_weights_for_head, get_embedding_weights


def get_eigenvalues(wh, we):
    """ Get the eigenvalues for the w_v @ w_e @ w_u @ w_o matrix. """
    m = wh['w_v'] @ we['w_e'] @ we['w_u'] @ wh['w_o']
    return np.linalg.eigvals(m)


def source_to_out(source, tokenizer, head_weights, embedding_weights):
    """ OV circuit for a single head. """
    tok = tokenizer.encode(source)
    if len(tok) > 1:
        raise ValueError("source must be a single token")
    
    x = embedding_weights['w_e'][:, tok]

    v = head_weights['w_v'] @ x
    o = head_weights['w_o'] @ v
    y = embedding_weights['w_u'] @ o

    torch_y = torch.from_numpy(y).squeeze(1)
    top = torch.topk(torch_y, 5)
    print(tokenizer.decode_tokens_bytes(top.indices.tolist()))
    print(top.values.tolist())


def source_to_dest(source, tokenizer, head_weights, embedding_weights, head,
                   subtract_start=True):
    """ QK circuit for a single head. """
    tok = tokenizer.encode(source)
    if len(tok) > 1:
        raise ValueError("source must be a single token")
    
    def get_dst(t):
        x = embedding_weights['w_e'][:, t]
        k = head_weights['w_k'] @ x
        kq = head_weights['w_q'].T @ k
        return embedding_weights['w_e'].T @ kq

    dst = get_dst(tok)
    dst = dst.squeeze(1)

    if subtract_start:
        qk_start = get_dst(-1)
        dst = dst - qk_start
        dst = dst[:-1]  # remove the start token
    else:
        qk_averages = np.load(f'qk_big_nolnf_nobias/head_{head}.npy')
        # subtract the average qk value for each query
        dst = dst - np.array(qk_averages)

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


if __name__=="__main__":
    enc = tiktoken.get_encoding("gpt2")

    weights = torch.load("../from_odin/big_drop_2_48000.pt", map_location='cpu')

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

    # extract the weights for each head
    head_weights = []
    for head in range(n_heads):
        h_w = get_weights_for_head(weights=weights, layer=0, head=head,
                            n_heads=n_heads, d_model=d_model, apply_layernorm=False)
        head_weights.append(h_w)

    embedding_weights = get_embedding_weights(weights=weights, d_model=d_model,
                                              norm_emb=True, final_layernorm=True)

    # eigenvalues for each head
    graphs = []
    for head in range(n_heads):
        eigen = get_eigenvalues(wh=head_weights[head], we=embedding_weights)
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
    word = " perfect"
    print('word:', word)

    for h in range(n_heads):
        print()
        print("head", h)

        # positional attention for head
        positional_attention_for_head(head_weights[h])

        print("source to out")
        source_to_out(
            word,
            tokenizer=enc,
            head_weights=head_weights[h],
            embedding_weights=embedding_weights,
        )
        print("source to dest")
        source_to_dest(source=word, tokenizer=enc, head_weights=head_weights[h],
                       embedding_weights=embedding_weights, head=h)


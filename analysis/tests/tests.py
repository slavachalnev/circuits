
import numpy as np
import torch
import tiktoken

from analysis.one_layer import get_weights_for_head
from circuits.models.one_attn_layer import OneLayerAttnTransformer
from circuits.train.train_one_layer import get_config


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

    return out


def split_forward_pass(weights, tokens, n_heads, d_model):
    heads = []
    for head in range(n_heads):
        heads.append(get_weights_for_head(weights, head, n_heads, d_model,
                                          norm_emb=True, final_ln=True))
    h0 = heads[0]

    # embedding
    x = weights['embedding.weight'].numpy()[tokens, :]
    # x shape is (seq_len, d_model)

    x_ln = h0['w_e'].T[tokens, :]
    print('x_ln is ', x_ln)

    # # compute layernormed x
    # x_ln = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)
    # x_ln = x_ln * h0['ln1w'].T + h0['ln1b'].T

    # perform forward pass
    head_outs = []
    for head in range(n_heads):
        # perform forward pass for head
        head_outs.append(head_forward_pass(x_ln, heads[head]))
        pass

    # combine heads
    for h_out in head_outs:
        x = x + h_out
    
    # final layer norm
    x = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)
    x = x * h0['lnfw'].T + h0['lnfb'].T
    
    # unembedding
    x = x @ heads[0]['w_u'].T
    return x


def test_split_heads():
    """
    Splitting PyTorch's MultiHeadAttention into individual heads is messy so we make 
    sure that the split-heads forward pass is the same as the normal forward pass.
    """

    enc = tiktoken.get_encoding("gpt2")
    text = " hi my name is"
    x = enc.encode(text)
    x = np.array(x, dtype=np.int32)
    print(x)
    print(type(x))

    # load weights
    weights = torch.load("../../from_odin/big_yeslnf_22000.pt", map_location='cpu')
    for weight in weights:
        print(weight, weights[weight].shape)
    
    # perform forward pass on split heads
    split_res = split_forward_pass(weights, x, 12, 768)
    split_res = torch.from_numpy(split_res)
    split_predictions = torch.argmax(split_res, dim=1)
    print('split predictions', split_predictions)
    print('split predictions', enc.decode(split_predictions.numpy()))

    # initialize normal model
    config = get_config()
    config.model.block_size = 2048
    config.model.n_embd = 768
    config.model.n_head = 12
    model = OneLayerAttnTransformer(config.model)
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        normal_res, _ = model(torch.from_numpy(x).unsqueeze(dim=0), None)
        normal_res = normal_res.squeeze(dim=0).detach()

    normal_predictions = torch.argmax(normal_res, dim=1)
    print('normal predictions', normal_predictions)
    print('normal predictions', enc.decode(normal_predictions.numpy()))

    assert torch.allclose(split_res.to(dtype=torch.float32), normal_res, atol=1e-5)

    # idxs = enc.encode("Hello there, how")
    # in_batch = torch.tensor(idxs).unsqueeze(0)
    # out_idxs = model.generate(idx=in_batch, max_new_tokens=20)
    # print(enc.decode_tokens_bytes(out_idxs[0].tolist()))


if __name__=="__main__":
    test_split_heads()
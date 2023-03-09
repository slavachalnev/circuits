
import numpy as np
import torch
import tiktoken

from analysis.utils import get_subtract_avg_matrix, head_forward_pass, get_weights_for_head, get_embedding_weights
from circuits.models.one_attn_layer import OneLayerAttnTransformer
from circuits.train.train_one_layer import get_config


def split_forward_pass(weights, tokens, n_heads, d_model):
    # extract the weights for each head
    head_weights = []
    for head in range(n_heads):
        h_w = get_weights_for_head(weights=weights, layer=0, head=head,
                            n_heads=n_heads, d_model=d_model, apply_layernorm=False)
        head_weights.append(h_w)

    embedding_weights = get_embedding_weights(weights=weights, d_model=d_model,
                                              norm_emb=False, final_layernorm=False)

    # embedding
    x = weights['embedding.weight'].numpy()[tokens, :]
    # x shape is (seq_len, d_model)

    x_ln = embedding_weights['w_e'].T[tokens, :]
    print('x_ln is ', x_ln)

    # compute layernormed x
    x_ln = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)
    x_ln = x_ln * head_weights[0]['lnw'].T + head_weights[0]['lnb'].T

    # perform forward pass
    head_outs = []
    for head in range(n_heads):
        # perform forward pass for head
        head_outs.append(head_forward_pass(x_ln, head_weights[head])[0])
        pass

    # combine heads
    for h_out in head_outs:
        x = x + h_out
    
    # final layer norm
    x = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)
    x = x * embedding_weights['lnfw'].T + embedding_weights['lnfb'].T
    
    # unembedding
    x = x @ embedding_weights['w_u'].T
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
    weights = torch.load("../../from_odin/big_drop_2_48000.pt", map_location='cpu')
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


def test_subtract_avg():
    """
    Test that subtracting the average of a vector is the same as zeroing out the diagonal of a matrix.
    """
    dim = 768
    x = np.random.randn(dim)
    z = get_subtract_avg_matrix(dim)
    assert np.allclose(x - np.mean(x), z @ x)


if __name__=="__main__":
    test_split_heads()
    test_subtract_avg()

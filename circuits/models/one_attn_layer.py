import math

import torch
import torch.nn as nn

from yacs.config import CfgNode as CN

from circuits.models.model import Model


class SinusoidalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class AttentionOnlyBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, pos_pdrop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            n_embed,
            num_heads=n_head,
            batch_first=True,
            bias=False,
            )
        self.pos = SinusoidalEncoding(d_model=n_embed, dropout=pos_pdrop, max_len=block_size)
        self.ln = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln(x)

        # posigitonal encoding as per shortformer
        # https://aclanthology.org/2021.acl-long.427.pdf
        px = self.pos(x)

        # compute attention mask
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)

        h, _ = self.attn(query=px, key=px, value=x, attn_mask=mask)
        return x + h


class OneLayerAttnTransformer(Model):

    @staticmethod
    def get_default_config():
        C = CN()

        C.vocab_size = None
        C.block_size = None

        # model dimensions
        C.n_embd = 512
        C.n_head = 8

        # dropout hyperparameters
        C.pos_embd_pdrop = 0.0

        return C
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.attn = AttentionOnlyBlock(
            n_embed=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            pos_pdrop=config.pos_embd_pdrop,
            )
        # self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.unembedding = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        x = self.embedding(x)
        x = self.attn(x)

        # final layer norm
        # x = self.ln_f(x)

        # unembedding
        logits = self.unembedding(x)

        # loss
        loss = None
        if targets is not None:
            loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


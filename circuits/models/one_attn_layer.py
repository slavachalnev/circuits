
import torch
import torch.nn as nn

from yacs.config import CfgNode as CN

from circuits.models.model import Model


class AttentionOnlyBlock(nn.Module):
    def __init__(self, n_embed, n_head, attn_pdrop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, num_heads=n_head, batch_first=True)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
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

        return C
    
    def __init__(self, config):
        super().__init__()

        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.attn = AttentionOnlyBlock(n_embed=config.n_embd, n_head=config.n_head)
        self.unembedding = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # embedding
        x = self.embedding(x)

        # position embedding
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0) # shape (1, t)
        pos_emb = self.pos_embedding(pos)
        x = x + pos_emb

        # attention layer
        x = self.attn(x)

        # unembedding
        logits = self.unembedding(x)

        # loss
        loss = None
        if targets is not None:
            loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


import math

import torch
import torch.nn as nn

from yacs.config import CfgNode as CN

from circuits.models.model import Model, AttentionOnlyBlock


class TwoLayerAttnTransformer(Model):

    @staticmethod
    def get_default_config():
        C = CN()

        C.vocab_size = None
        C.block_size = None

        # model dimensions
        C.n_embd = None
        C.n_head = None

        # dropout hyperparameters
        C.pos_embd_pdrop = 0.0

        return C
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.b0 = AttentionOnlyBlock(
            n_embed=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            pos_pdrop=config.pos_embd_pdrop,
            )

        self.b1 = AttentionOnlyBlock(
            n_embed=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            pos_pdrop=config.pos_embd_pdrop,
            )
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.unembedding = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        x = self.embedding(x)

        x = self.b0(x)['res']
        x = self.b1(x)['res']

        # final layer norm
        x = self.ln_f(x)

        # unembedding
        logits = self.unembedding(x)

        # loss
        loss = None
        if targets is not None:
            loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


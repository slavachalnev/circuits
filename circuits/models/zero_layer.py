import torch
import torch.nn as nn

from yacs.config import CfgNode as CN

from circuits.models.model import Model


class ZeroLayerTransformer(Model):
    """Input embedding followed by unembedding."""

    @staticmethod
    def get_default_config():
        C = CN()

        C.vocab_size = None
        C.block_size = None

        # model dimensions
        C.n_embd = 128

        # # dropout hyperparameters
        # C.embd_pdrop = 0.0
        # C.resid_pdrop = 0.0
        # C.attn_pdrop = 0.0
        return C

    def __init__(self, config):
        super().__init__()

        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # unembedding
        # self.unembedding = nn.Parameter(torch.Tensor(config.n_embd, config.vocab_size))
        # nn.init.xavier_uniform_(self.unembedding)
        self.unembedding = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # embedding
        x = self.embedding(x)

        # unembedding
        # logits = torch.matmul(x, self.unembedding)
        logits = self.unembedding(x)

        # loss
        loss = None
        if targets is not None:
            loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
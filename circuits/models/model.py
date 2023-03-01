from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Base class for all transformer models."""
    # TODO: add a default config

    def __init__(self):
        super().__init__()

    def get_vocab_size(self):
        """Return the size of the vocabulary."""
        raise NotImplementedError

    def forward(self, x, targets=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        raise NotImplementedError
    
    def _init_weights(self, module):
        """Initialize the weights."""
        # TODO: think about scaling the residual stream projections

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self, train_config):
        """ Initialize optimizer. Weight decay is not applied to LayerNorm, bias, and embedding weights. """

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('weight'):
                    if isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)
                else:
                    no_decay.add(fpn)
        
        # validate that all parameters are accounted for
        all_params = set(self.named_parameters())
        assert len(all_params) == len(decay) + len(no_decay)

        # initialize optimizer
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {'params': [param_dict[pn] for pn in sorted(decay)], 'weight_decay': train_config.weight_decay},
            {'params': [param_dict[pn] for pn in sorted(no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """Generate text from the model."""
        self.eval()
        print('idx is ', idx)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            print('probs is ', probs)
            print('max probs idx is ', torch.argmax(probs, dim=-1))
            

            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


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
    def __init__(self, n_embed, n_head, block_size, pos_pdrop=0.0, attn_pdrop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            n_embed,
            num_heads=n_head,
            batch_first=True,
            bias=False,
            dropout=attn_pdrop,
            )
        self.pos = SinusoidalEncoding(d_model=n_embed, dropout=pos_pdrop, max_len=block_size)
        self.ln = nn.LayerNorm(n_embed)
        self.drop = nn.Dropout(pos_pdrop)

    def forward(self, x, qk=None, add_to_res=True):
        h_in = self.ln(x)

        # posigitonal encoding as per shortformer
        # https://aclanthology.org/2021.acl-long.427.pdf
        px = self.pos(h_in)
        px = self.drop(px)

        if qk is not None:
            px = qk

        # compute attention mask
        mask = torch.triu(torch.ones(h_in.shape[1], h_in.shape[1]), diagonal=1).bool().to(h_in.device)

        h_out, _ = self.attn(query=px, key=px, value=h_in, attn_mask=mask)
        
        if add_to_res:
            res = x + h_out
        else:
            res = x
        
        return {'res': res, 'qk': px, 'h_out': h_out}


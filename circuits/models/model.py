from typing import Optional, Tuple

import torch
import torch.nn as nn


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


import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BitShiftMamba(nn.Module):
    def __init__(self, length=32, d_model=8, d_state=4, d_conv=2, expand=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.length = length
        self.embedding = nn.Embedding(2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # single logit per bit

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.embedding.weight, 0.0, 0.1)

    def forward(self, x):
     h = self.embedding(x)
     h = self.dropout(h)
     for layer in self.mamba_layers:
        h = layer(h)  # No residual
     h = self.norm(h)
     logits = self.head(h).squeeze(-1)
     return logits

import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiMambaBlock(nn.Module):
    """
    One bidirectional Mamba block:
      - Runs a forward Mamba on input (no time flip).
      - Runs a backward Mamba on time-flipped input and un-flips the output.
      - Each branch uses pre-LN, Mamba, dropout and residual.
      - Outputs average of forward & reverse branch (with optional residual scaling).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        share_norm: bool = False,
        share_ffn: bool = False,
    ):
        super().__init__()
        # forward branch
        self.pre_ln_f = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.post_ln_f = nn.LayerNorm(d_model)
        self.ffn_f = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # reverse branch
        if share_norm:
            self.pre_ln_r = self.pre_ln_f
            self.post_ln_r = self.post_ln_f
        else:
            self.pre_ln_r = nn.LayerNorm(d_model)
            self.post_ln_r = nn.LayerNorm(d_model)

        self.mamba_r = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if share_ffn:
            self.ffn_r = self.ffn_f
        else:
            self.ffn_r = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward_branch(self, x, pre_ln, mamba, post_ln, ffn, flip_time=False):
        """
        x: (B, S, D)
        flip_time: if True, flip along time dim before applying mamba and unflip after
        """
        x_in = x
        if flip_time:
            x_proc = torch.flip(x, dims=[1])
        else:
            x_proc = x

        h = pre_ln(x_proc)
        h = mamba(h)
        h = self.dropout(h)
        if flip_time:
            h = torch.flip(h, dims=[1])

        # residual + post norm
        h = x_in + self.residual_scale * h
        h = post_ln(h)

        # feedforward with residual
        y = ffn(h)
        y = self.dropout(y)
        y = h + self.residual_scale * y
        y = post_ln(y)  # apply post norm again (keeps pattern similar to transformer block)
        return y

    def forward(self, x):
        out_f = self.forward_branch(x, self.pre_ln_f, self.mamba_f, self.post_ln_f, self.ffn_f, flip_time=False)
        out_r = self.forward_branch(x, self.pre_ln_r, self.mamba_r, self.post_ln_r, self.ffn_r, flip_time=True)
        return 0.5 * (out_f + out_r)


class BiMambaEncoder(nn.Module):
    """
    Bidirectional Mamba Encoder that:
      - Accepts x of shape (B, S) with {0,1} (long) or (B, S, D) floats.
      - If input is (B, S) it uses nn.Embedding(2, d_model).
      - Adds learnable positional embeddings.
      - Stacks num_layers BiMambaBlock.
      - Returns (B, S, d_model) normalized output.
    """
    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 4,
        seq_len: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        share_norm: bool = False,
        share_ffn: bool = False,
        use_embedding_for_bits: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_embedding_for_bits = use_embedding_for_bits

        if use_embedding_for_bits:
            self.token_emb = nn.Embedding(2, d_model)
        else:
            self.token_emb = None

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                residual_scale=residual_scale,
                share_norm=share_norm,
                share_ffn=share_ffn,
            ) for _ in range(num_layers)
        ])

        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        if self.token_emb is not None:
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: either
          - (B, S) with dtype long containing 0/1 (bits)  --> uses token embedding
          - (B, S, D) float already embedded (D must equal d_model)
        returns:
          (B, S, d_model)
        """
        if x.dim() == 2:
            if self.token_emb is None:
                raise ValueError("Input is (B,S) but token embedding is disabled.")
            # bits case
            h = self.token_emb(x.long())                  # (B, S, D)
        elif x.dim() == 3:
            if x.size(2) != self.d_model:
                raise ValueError(f"Input last dim {x.size(2)} != d_model {self.d_model}")
            h = x
        else:
            raise ValueError("Input must be (B,S) or (B,S,D)")

        # add positional embeddings (truncate if sequence shorter)
        L = h.size(1)
        if L > self.seq_len:
            raise ValueError(f"Sequence length {L} > seq_len {self.seq_len}")
        h = h + self.pos_emb[:, :L, :]

        # pass through stacked BiMambaBlocks
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        return self.head(h).squeeze(-1)

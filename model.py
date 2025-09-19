import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Optional gradient checkpointing (now re-enabled via --grad-checkpoint flag).

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Clone to ensure unique, contiguous storage (helps with some DDP broadcast edge cases)
        self.register_buffer('pe', pe.clone().contiguous())

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FlashEncoderLayer(nn.Module):
    """Transformer encoder layer using scaled_dot_product_attention (SDPA) with closer parity to
    nn.TransformerEncoderLayer (GELU activation, dropout structure, LayerNorm order) so training
    dynamics align with the baseline implementation.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout_p = attn_dropout

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout_resid1 = nn.Dropout(dropout)
        self.dropout_resid2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Parameter initialization similar to TransformerEncoderLayer (Xavier for projections)
        for mod in [self.W_qkv, self.W_o, self.linear1, self.linear2]:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # src: (S,B,E)
        S, B, E = src.shape
        qkv = self.W_qkv(src)  # (S,B,3E)
        qkv = qkv.view(S, B, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (S,B,H,D)
        q = q.permute(1, 2, 0, 3).reshape(B * self.nhead, S, self.head_dim)
        k = k.permute(1, 2, 0, 3).reshape(B * self.nhead, S, self.head_dim)
        v = v.permute(1, 2, 0, 3).reshape(B * self.nhead, S, self.head_dim)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=True
        )  # (B*H, S, D)
        attn_out = attn_out.reshape(B, self.nhead, S, self.head_dim).permute(2, 0, 1, 3).contiguous()
        attn_out = attn_out.view(S, B, E)
        attn_out = self.W_o(attn_out)
        attn_out = self.dropout_resid1(attn_out)
        src = self.norm1(src + attn_out)

        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        ff = self.dropout_resid2(ff)
        src = self.norm2(src + ff)
        return src

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5,
                 use_checkpoint: bool = False, use_flash_attn: bool = False):
        super().__init__()
        self.model_type = 'Transformer'
        self.use_flash_attn = use_flash_attn
        self.use_checkpoint = use_checkpoint
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        if self.use_flash_attn:
            self.layers = nn.ModuleList([
                FlashEncoderLayer(ninp, nhead, nhid, dropout, attn_dropout=0.1)
                for _ in range(nlayers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                for _ in range(nlayers)
            ])
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.decoder.bias is not None:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        # src: (seq_len, batch)
        x = self.encoder(src) * math.sqrt(self.ninp)
        x = self.pos_encoder(x)
        if self.use_flash_attn:
            for layer in self.layers:  # Flash path ignores src_mask; causal handled internally
                if self.use_checkpoint and self.training:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
        else:
            for layer in self.layers:
                if self.use_checkpoint and self.training:
                    # Wrap layer to bind mask argument
                    x = checkpoint(lambda inp, l=layer, m=src_mask: l(inp, m), x)
                else:
                    x = layer(x, src_mask)
        return self.decoder(x)

    @property
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

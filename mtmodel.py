import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import TransformerWrapper, Encoder, Decoder
from einops import rearrange, repeat

from typing import List


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

class HelperModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

class LengthPredictor(HelperModule):
    def build(self,
            dim: int,
            nb_out: int,
            dim_mult: int = 4,
            nb_blocks: int = 6,
            dropout: float = 0.0,
        ):

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim_mult*dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_mult*dim, dim),
                nn.SiLU(),
            )
            for i in range(nb_blocks)
        ])
        self.out_fc = nn.Linear(dim, nb_out)

    def forward(self, x: torch.Tensor):
        h = x
        for l in self.layers:
            h = h + l(h)
        h = self.out_fc(h)
        return h, h.argmax(dim=-1)

# TODO: add dropout
class Transformer(HelperModule):
    def build(self,
            src_num_tokens: int,
            tgt_num_tokens: int,
            src_max_seq_len: int,
            tgt_max_seq_len: int,
            dim: int,
            depth: int,
            nb_heads: int = 8,
            pre_tgt_pred_dim: int = 64,
            tgt_pred_dim: int = 128,
            downsample_len: int = 2,
        ):
        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len

        self.src_len_emb = nn.Embedding(src_max_seq_len, tgt_pred_dim)
        self.tgt_len_emb = nn.Embedding(tgt_max_seq_len // downsample_len, dim)
        self.tgt_len_proj1 = nn.Linear(dim, pre_tgt_pred_dim)
        self.tgt_len_proj2 = nn.Linear(src_max_seq_len*pre_tgt_pred_dim, tgt_pred_dim)

        self.encoder = TransformerWrapper(
            num_tokens=src_num_tokens,
            max_seq_len=src_max_seq_len,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                head=nb_heads,
                rotary_pos_emb=True
            )
        )

        self.decoder = TransformerWrapper(
            num_tokens=tgt_num_tokens,
            max_seq_len=tgt_max_seq_len,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                head=nb_heads,
                cross_attend=True,
                rotary_pos_emb=True
            )
        )

        self.downsample_len = downsample_len
        self.len_predictor = LengthPredictor(
            dim=tgt_pred_dim,
            nb_out=tgt_max_seq_len // downsample_len,
            nb_blocks=6,
        )

    def forward(self, 
            src: torch.Tensor, src_len: int, 
            tgt: torch.Tensor, tgt_len: int,
            src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
            src_h: torch.Tensor = None,
        ):
        if not exists(src_mask):
            src_mask = torch.ones_like(src).bool()
        
        len_loss = None
        if not exists(src_h):
            src_h = self.encoder(src, mask=src_mask, return_embeddings=True) 
            # TODO: is the first mask needed? proj is pointwise
            src_v = self.tgt_len_proj1(src_h.detach() * src_mask.unsqueeze(-1)) * src_mask.unsqueeze(-1) # detach to avoid updating encoder with target loss prediction
            src_v = rearrange(src_v, 'n l d -> n (l d)')
            src_v = self.tgt_len_proj2(src_v)

            src_len_emb = self.src_len_emb(src_len)
            src_v = src_v + src_len_emb

            tgt_len_pred_logits, tgt_len_pred = self.len_predictor(src_v) # paper has a typo for this part

            len_loss = F.cross_entropy(tgt_len_pred_logits, torch.div(tgt_len, self.downsample_len, rounding_mode='trunc'))
            tgt_len_emb = self.tgt_len_emb(torch.div(tgt_len, self.downsample_len, rounding_mode='trunc')) 

            src_h = torch.cat([tgt_len_emb.unsqueeze(1), src_h], dim=1)

        src_mask = torch.cat([torch.ones(src_h.shape[0], 1).bool().to(src_mask.device), src_mask], dim=-1)
        tgt_h = self.decoder(tgt, context=src_h, mask=tgt_mask, context_mask=src_mask)
        return tgt_h, src_h, len_loss

    @torch.cuda.amp.autocast()
    @torch.inference_mode()
    def sample_step(self,
            src: torch.Tensor, src_len: int,
            tgt: torch.Tensor, 
            src_mask: torch.Tensor = None,
            src_h : torch.Tensor = None,
        ):
        if not exists(src_mask):
            src_mask = torch.ones_like(src).bool()

        if not exists(src_h):
            src_h = self.encoder(src, mask=src_mask, return_embeddings=True) 
            src_v = self.tgt_len_proj1(src_h.detach() * src_mask.unsqueeze(-1)) * src_mask.unsqueeze(-1) 
            src_v = rearrange(src_v, 'n l d -> n (l d)')
            src_v = self.tgt_len_proj2(src_v)

            src_len_emb = self.src_len_emb(src_len)
            src_v = src_v + src_len_emb

            _, tgt_len_pred = self.len_predictor(src_v) # paper has a typo for this part
            tgt_len_emb = self.tgt_len_emb(torch.div(tgt_len_pred, self.downsample_len, rounding_mode='trunc')) 

            src_h = torch.cat([tgt_len_emb.unsqueeze(1), src_h], dim=1)

        src_mask = torch.cat([torch.ones(src_h.shape[0], 1).bool().to(src_mask.device), src_mask], dim=-1)
        tgt_h = self.decoder(tgt, context=src_h, context_mask=src_mask) # TODO: how to mask tgt?

        return tgt_h, src_h

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Transformer(6000, 6000, 256, 256, 768, 6).to(device)
    src = torch.randint(0, 6000, (4, 256)).to(device)
    src_len = torch.tensor([255, 255, 255, 255]).to(device)
    tgt = torch.randint(0, 6000, (4, 256)).to(device)
    tgt_len = torch.tensor([255, 255, 255, 255]).to(device)

    tgt_logits, src_emb, len_loss = net(src=src, src_len=src_len, tgt=tgt, tgt_len=tgt_len)
    print(tgt_logits.shape)
    print(src_emb.shape)
    print(len_loss)

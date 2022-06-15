import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import TransformerWrapper, Encoder, Decoder

def exists(val):
    return val is not None

class HelperModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

class ResidualBlock(HelperModule):
    def build(self):
        pass

    def forward(self, x: torch.Tensor):
        pass

class LengthPredictor(HelperModule):
    def build(self,
            nb_out: int,
            nb_layers: int = 6,
        ):
        self.layers = nn.Sequential(*[ResidualBlock() for _ in range(nb_layers)])
        self.out_fc = nn.LazyLinear(nb_out)

    def forward(self, x: torch.Tensor):
        h = self.layers(h)
        h = self.out(h)
        return h, h.argmax(dim=-1)

class Transformer(HelperModule):
    def build(self,
            src_num_tokens: int,
            tgt_num_tokens: int,
            src_max_seq_len: int,
            tgt_max_seq_len: int,
            dim: int,
            tgt_pred_dim: int,
            depth: int,
            head: int,
            downsample_len: int = 2,
        ):
        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len

        self.src_len_emb = nn.Embedding(src_max_seq_len, tgt_pred_dim)
        self.tgt_len_emb = nn.Embedding(tgt_max_seq_len // downsample_len, dim)
        self.pre_tgt_pred = nn.Linear(dim, tgt_pred_dim)
        self.encoder = TransformerWrapper(
            num_tokens=src_num_tokens,
            max_seq_len=src_max_seq_len,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                head=head,
                rotary_pos_emb=True
            )
        )

        self.decoder = TransformerWrapper(
            num_tokens=tgt_num_tokens,
            max_seq_len=tgt_max_seq_len,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                head=head,
                cross_attend=True,
                rotary_pos_emb=True
            )
        )

        self.downsample_len = downsample_len
        self.len_predictor = LengthPredictor(
            nb_out=tgt_max_seq_len // downsample_len,
            nb_layers=6,
        )

    def forward(self, 
            src: torch.Tensor, src_len: int, 
            tgt: torch.Tensor, tgt_len: int,
            src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None
        ):
        if src_len > self.src_max_seq_len:
            raise ValueError(f"Provided source length ({src_len}) exceeds maximum length of model ({self.src_max_seq_len}).")
        if tgt_len > self.tgt_max_seq_len:
            raise ValueError(f"Provided target length ({tgt_len}) exceeds maximum length of model ({self.tgt_max_seq_len}).")
        if src_len >= 0:
            raise ValueError(f"Provided source length ({src_len}) is not greater than 0.")
        if tgt_len >= 0:
            raise ValueError(f"Provided target length ({tgt_len}) is not greater than 0.")

        src_h = self.encoder(src, mask=src_mask) # return embeddings?
        src_v = self.pre_tgt_pred(src_h.detach()) # detach to avoid updating encoder with target loss prediction

        src_len_emb = self.src_len_emb(src_len)
        src_v = src_v + src_len_emb

        tgt_len_pred_logits, tgt_len_pred = self.len_predictor(src_v) # think paper has a typo for this part

        len_loss = F.cross_entropy(tgt_len // downsample_len, tgt_len_pred_logits)
        tgt_len_emb = self.tgt_len_emb(tgt_len // 2)

        src_h = torch.cat([tgt_len_emb.unsqueeze(1), src_h], dim=1)
        tgt_h = self.decoder(tgt, context=src_h, mask=tgt_mask, context_mask=src_mask)
        return tgt_h

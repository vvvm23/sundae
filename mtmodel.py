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

class ResidualBlock(HelperModule):
    def build(self,
            in_channels: int,
            out_channels: int,
        ):
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv_res = nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.in_conv(x)
        h = self.conv1(h)
        h = self.conv2(h)
        x = F.relu(h + self.conv_res(x))
        return x

# it is not clear in the paper how the authors implemented the length predictor architecture
# only details are it having "6 residual blocks"
# guessing required..
# TODO: lazy layer would probably fail given different sized inputs
# TODO: how to resolve? mean pooling after res?
class LengthPredictor(HelperModule):
    def build(self,
            channels_in: int,
            nb_out: int,
            nb_layers: int = 6,
            channel_mult: List[int] = None,
        ):
        channel_mult = default(channel_mult, [1, 1, 2, 2, 4, 4])
        if not len(channel_mult) == nb_layers:
            raise ValueError(f"Length of channel multipliers ({channel_mult}) did not match number of layers ({nb_layers})!")
        channel_mult.insert(0, 1)

        self.layers = nn.Sequential(*[
            ResidualBlock(channels_in*channel_mult[i], channels_in*channel_mult[i+1]) 
            for i in range(nb_layers)
        ])
        self.out_fc = nn.LazyLinear(nb_out)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'n l c -> n c l')
        h = self.layers(x)
        h = rearrange(h, 'n c l -> n (c l)')
        h = self.out_fc(h)
        return h, h.argmax(dim=-1)

class Transformer(HelperModule):
    def build(self,
            src_num_tokens: int,
            tgt_num_tokens: int,
            src_max_seq_len: int,
            tgt_max_seq_len: int,
            dim: int,
            depth: int,
            nb_heads: int = 8,
            tgt_pred_dim: int = 128,
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
            channels_in=tgt_pred_dim,
            nb_out=tgt_max_seq_len // downsample_len,
            nb_layers=6,
        )

    def forward(self, 
            src: torch.Tensor, src_len: int, 
            tgt: torch.Tensor, tgt_len: int,
            src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
            src_h: torch.Tensor = None,
        ):
        # TODO: add checks for tensor case
        # if src_len > self.src_max_seq_len:
            # raise ValueError(f"Provided source length ({src_len}) exceeds maximum length of model ({self.src_max_seq_len}).")
        # if tgt_len > self.tgt_max_seq_len:
            # raise ValueError(f"Provided target length ({tgt_len}) exceeds maximum length of model ({self.tgt_max_seq_len}).")
        # if src_len <= 0:
            # raise ValueError(f"Provided source length ({src_len}) is not greater than 0.")
        # if tgt_len <= 0:
            # raise ValueError(f"Provided target length ({tgt_len}) is not greater than 0.")
        src_len = src_len - 1
        tgt_len = tgt_len - 1

        if not exists(src_h):
            src_h = self.encoder(src, mask=src_mask, return_embeddings=True) # return embeddings?
            src_v = self.pre_tgt_pred(src_h.detach()) # detach to avoid updating encoder with target loss prediction

            src_len_emb = self.src_len_emb(src_len)
            src_len_emb = repeat(src_len_emb, 'n c -> n l c', l=src_v.shape[1])
            src_v = src_v + src_len_emb

            tgt_len_pred_logits, tgt_len_pred = self.len_predictor(src_v) # think paper has a typo for this part

            len_loss = F.cross_entropy(tgt_len_pred_logits, torch.div(tgt_len, self.downsample_len, rounding_mode='trunc'))
            tgt_len_emb = self.tgt_len_emb( torch.div(tgt_len, self.downsample_len, rounding_mode='trunc')) 

            src_h = torch.cat([tgt_len_emb.unsqueeze(1), src_h], dim=1)

        tgt_h = self.decoder(tgt, context=src_h, mask=tgt_mask, context_mask=src_mask)
        return tgt_h, src_h, len_loss

if __name__ == '__main__':
    device = torch.device('cuda')

    net = Transformer(6000, 6000, 256, 256, 768, 6).to(device)
    src = torch.randint(0, 6000, (4, 256)).to(device)
    src_len = torch.tensor([256, 256, 256, 256]).to(device)
    tgt = torch.randint(0, 6000, (4, 256)).to(device)
    tgt_len = torch.tensor([256, 256, 256, 256]).to(device)

    tgt_logits, src_emb, len_loss = net(src=src, src_len=src_len, tgt=tgt, tgt_len=tgt_len)
    print(tgt_logits.shape)
    print(src_emb.shape)
    print(len_loss)

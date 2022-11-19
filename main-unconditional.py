#!/usr/bin/env python
import poptorch
import argparse

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm

import toml
import contextlib
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path

from data import get_text8

from x_transformers import TransformerWrapper, Encoder

def build_ipu_mapping(net, cfg):
    return net

def get_random_text(shape, vocab_size):
    return torch.randint(vocab_size, shape)

def corrupt_text(batched_text, vocab_size):
    corruption_prob_per_sequence = torch.rand((batched_text.shape[0], 1))
    rand = torch.rand(batched_text.shape)
    mask = (rand < corruption_prob_per_sequence).to(batched_text.device)

    random_text = get_random_text(batched_text.shape, vocab_size).to(batched_text.device)
    return mask * random_text + ~mask * batched_text

class WrappedModel(torch.nn.Module):
    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg

    def forward(self, batched_text):
        samples = corrupt_text(batched_text, self.cfg.data['vocabulary_size'])
        all_logits = []
        for _ in range(self.cfg.unroll_steps):
            logits, samples = self.denoise_step(samples)
            all_logits.append(logits)
        logits = torch.cat(all_logits, axis=0)

        targets = batched_text.repeat(self.cfg.unroll_steps, 1)
        accuracy = (logits.argmax(dim=-1) == targets).sum() / targets.numel()
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets)

        return poptorch.identity_loss(loss), accuracy*100

    def denoise_step(self, sample):
        logits = self.net(sample)
        return logits, Categorical(logits=logits).sample().detach()


@contextlib.contextmanager
def print_block(*args, **kwargs):
    print(*args, **kwargs, end=' ')
    yield
    print("Done.")

def main(args):

    batch_size = 100
    device_iterations = 100
    replication_factor = 4
    gradient_accumlation = 10
    nb_epochs = 1

    with print_block("Loading config."):
        cfg = SimpleNamespace(**toml.load(args.cfg_path))

    with print_block("Setting up opts."):
        opts = poptorch.Options()
        opts.deviceIterations(device_iterations)
        opts.Training.gradientAccumulation(gradient_accumlation)
        opts.replicationFactor(replication_factor)

    with print_block("Loading dataset."):
        train_dataset, eval_dataset = get_text8(cfg.data['root'], seq_len=cfg.data['sequence_length'])

    with print_block("Initialising pipelined network."):
        net = TransformerWrapper(
            num_tokens = cfg.data['vocabulary_size'],
            max_seq_len = cfg.data['sequence_length'],
            attn_layers = Encoder(
                dim = cfg.model['embedding_dim'],
                depth = cfg.model['nb_layers'],
                head = cfg.model['nb_heads'],
                use_scalenorm = cfg.model['use_scalenorm'],
                ff_glu = cfg.model['use_glu'],
                rotary_pos_emb=cfg.model['use_rotary'],
            )
        )
        net = WrappedModel(net, cfg).train()
        net.net.attn_layers.layers[0] = poptorch.BeginBlock(net.net.attn_layers.layers[2], ipu_id=1)
        net.net.attn_layers.layers[1] = poptorch.BeginBlock(net.net.attn_layers.layers[4], ipu_id=2)
        net.net.attn_layers.layers[0] = poptorch.BeginBlock(net.net.attn_layers.layers[6], ipu_id=3)

    with print_block("Setting up PopTorch dataloader"):
        train_loader = poptorch.DataLoader(
            options=opts,
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            mode=poptorch.DataLoaderMode.Async,
            num_workers=16,
        )

    with print_block("Setting up PopTorch training model."):
        optimiser = torch.optim.AdamW(net.parameters(), lr=cfg.trainer['learning_rate'])
        poptorch_model = poptorch.trainingModel(net, options=opts, optimizer=optimiser)

    for ei in range(nb_epochs):
        print(f"epoch {ei+1}")
        for batch in tqdm(train_loader):
            print(batch)
            loss, accuracy = poptorch_model(batch)
            print(loss)

    # @torch.inference_mode()
    # def sample_fn(net):
    #     device = get_device(not args.no_cuda)

    #     batched_text = get_random_text((args.nb_samples, cfg.data['sequence_length'])).to(device)
    #     sample_mask = torch.zeros(args.nb_samples).bool().to(device)
    #     for n in range(cfg.sample['steps']):
    #         old_sample_mask = sample_mask.clone()
    #         logits = net(batched_text[~sample_mask])
    #         sample = Categorical(logits=logits / cfg.sample['temperature']).sample()
            
    #         mask = (torch.rand(sample.shape) > cfg.sample['sample_proportion']).to(batched_text.device)
    #         sample[mask] = batched_text[~sample_mask][mask]
 
    #         if n >= cfg.sample['min_steps']:
    #             sample_mask[~sample_mask] = torch.all((sample == batched_text[~sample_mask]).view(sample.shape[0], -1), dim=-1)

    #         if torch.all(sample_mask).item():
    #             break
    #         batched_text[~old_sample_mask] = sample
    #     debug(f"stopped sampling after {n+1} steps.")
    #     return batched_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/text8.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=4)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--nb-samples', type=int, default=4)
    parser.add_argument('--min-steps', type=int, default=10)
    args = parser.parse_args()

    main(args)

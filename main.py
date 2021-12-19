import argparse

import torch
import torch.nn.functional as F

import toml
from types import SimpleNamespace
from pathlib import Path

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.utils import set_seed, get_parameter_count, get_device

from data import get_text8

from x_transformers import TransformerWrapper, Encoder

def main(args):
    seed = set_seed(args.seed)
    info(f"random seed: {seed}")
    train_dataset, eval_dataset = get_text8('data/text8', seq_len=32)

    net = TransformerWrapper(
        num_tokens = len(train_dataset.token_id),
        max_seq_len = 32,
        attn_layers = Encoder(
            dim = 512,
            depth = 12,
            head = 8,
            use_scalenorm = True,
            ff_glu = True,
            rotary_pos_emb=True,
        )
    )
    info(f"number of parameters: {get_parameter_count(net):,}")
    
    # def loss_fn(net, batch):

    trainer_cfg = TrainerConfig(
        exp_name = 'text8-sundae',
        exp_dir = 'exp',
        batch_size = 250,
        learning_rate = 1e-5,
        nb_workers = args.nb_workers,
        use_cuda = not args.no_cuda,
        use_amp = not args.no_amp,
        save_outputs = not args.no_save,
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = eval_dataset,
        cfg = trainer_cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/cifar10.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=4)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    main(args)

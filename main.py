import argparse

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import toml
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device

from data import get_text8

from x_transformers import TransformerWrapper, Encoder

def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    info(f"random seed: {seed}")
    train_dataset, eval_dataset = get_text8(cfg.data['root'], seq_len=cfg.data['sequence_length'])
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
    info(f"number of parameters: {get_parameter_count(net):,}")
    
    def get_random_text(shape):
        return torch.randint(cfg.data['vocabulary_size'], shape)

    def corrupt_text(batched_text):
        corruption_prob_per_sequence = torch.rand((batched_text.shape[0], 1))
        rand = torch.rand(batched_text.shape)
        mask = (rand < corruption_prob_per_sequence).to(batched_text.device)

        random_text = get_random_text(batched_text.shape).to(batched_text.device)
        return mask * random_text + ~mask * batched_text

    def logits_fn(net, batched_text):
        samples = corrupt_text(batched_text)
        all_logits = []
        for _ in range(cfg.unroll_steps):
            logits = net(samples)
            samples = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        final_logits = torch.cat(all_logits, axis=0)
        return final_logits

    def loss_fn(net, batched_text):
        logits = logits_fn(net, batched_text)
        targets = batched_text.repeat(cfg.unroll_steps, 1)
        accuracy = (logits.argmax(dim=-1) == targets).sum() / targets.numel()
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
        return loss, accuracy*100.
    
    @torch.inference_mode()
    def sample_fn(net):
        device = get_device(not args.no_cuda)

        batched_text = get_random_text((args.nb_samples, cfg.data['sequence_length'])).to(device)
        sample_mask = torch.zeros(args.nb_samples).bool().to(device)
        for n in range(cfg.sample['steps']):
            old_sample_mask = sample_mask.clone()
            logits = net(batched_text[~sample_mask])
            sample = Categorical(logits=logits / cfg.sample['temperature']).sample()
            
            mask = (torch.rand(sample.shape) > cfg.sample['sample_proportion']).to(batched_text.device)
            # mask = torch.logical_or(mask, sample_mask.view(-1, 1).repeat(1, sample.shape[1]))
            sample[mask] = batched_text[~sample_mask][mask]
 
            if n >= cfg.sample['min_steps']:
                sample_mask[~sample_mask] = torch.all((sample == batched_text[~sample_mask]).view(sample.shape[0], -1), dim=-1)

            if torch.all(sample_mask).item():
                break
            batched_text[~old_sample_mask] = sample
        debug(f"stopped sampling after {n+1} steps.")
        return batched_text

    # @torch.inference_mode()
    # @torch.cuda.amp.autocast()
    # def argmax_unrolled_sample_fn(net):
        # device = get_device(not args.no_cuda)

        # batched_text = get_random_text((args.nb_samples, cfg.data['sequence_length'])).to(device)
        # sample_mask = torch.zeros(args.nb_samples).bool().to(device)
        # prev_logits = None
        # for n in tqdm(range(cfg.sample['steps'])):
            # old_sample_mask = sample_mask.clone()

            # if prev_logits == None:
                # old_sample_mask = sample_mask.clone()
                # logits = net(batched_text[~sample_mask])
                # sample = Categorical(logits=logits / cfg.sample['temperature']).sample()
            # else:
                # prev_probs = F.softmax(prev_logits, dim=-1)
                # max_prev_probs, _ = prev_probs.max(dim=-1)
                # cutoffs = torch.quantile(max_prev_probs, 0.3, dim=-1)
                # argmax_mask = max_prev_probs <= cutoffs[..., None]

                # logits = net(batched_text[~sample_mask])
                # sample = logits.argmax(dim=-1)
                # mask = (torch.rand(sample.shape) > cfg.sample['sample_proportion']).to(batched_text.device)
                # sample[mask] = batched_text[~sample_mask][mask]

                # logits = net(sample)
                # # TODO: do we want to use sample proportion masking when using this decoding method?
                # sample[argmax_mask] = logits[argmax_mask].argmax(dim=-1)
                # sample[mask] = batched_text[~sample_mask][mask]

            # if n >= cfg.sample['min_steps']:
                # sub_sample_mask = torch.all((sample == batched_text[~sample_mask]).view(sample.shape[0], -1), dim=-1)
                # sample_mask[~sample_mask] = sub_sample_mask
            # else:
                # sub_sample_mask = torch.zeros(sample.shape[0]).bool().to(sample.device)

            # if torch.all(sample_mask).item():
                # break

            # prev_logits = logits[~sub_sample_mask]
            # batched_text[~old_sample_mask] = sample
        # return batched_text

    trainer_cfg = TrainerConfig(
        **cfg.trainer,
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
    
    @torch.inference_mode()
    def callback_sample(trainer):
        trainer.net.eval()
        info("sampling from current model")
        samples = sample_fn(trainer.net)
        # samples = argmax_unrolled_sample_fn(trainer.net)

        for i, sample in enumerate(samples):
            info(f"- " + ''.join(train_dataset.id_token[i.item()] for i in sample))
        trainer.net.train()

    if args.sample:
        callback_sample(trainer)
        exit()

    trainer.register_callback(CallbackType.EvalEpoch, callback_sample, cfg.sample['sample_frequency'])
    trainer.train()

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

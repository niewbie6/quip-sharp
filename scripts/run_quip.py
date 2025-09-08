#!/usr/bin/env python
"""Run QUIP# quantization on a HuggingFace model and evaluate perplexity.

This utility loads a model, computes its perplexity on a validation dataset,
performs QUIP# quantization with a chosen codebook and then evaluates the
perplexity of the quantized model. Progress bars are shown for both the
quantization and validation phases.
"""
import argparse
import math
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lib import codebook, utils
from lib.algo.quip import quantize
from lib.utils import gptq_data_utils


def evaluate_perplexity(model, tokens, seqlen, device):
    """Compute perplexity of ``model`` on ``tokens``."""
    model.eval()
    nsamples = tokens.numel() // seqlen
    tokens = tokens[0, :nsamples * seqlen].view(nsamples, seqlen)
    loss_fct = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    progress = tqdm(range(nsamples), desc="validation", leave=False)
    for i in progress:
        inp = tokens[i:i + 1].to(device)
        with torch.no_grad():
            logits = model(inp).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inp[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        total_loss += loss.item()
        progress.set_description(
            f"validation loss {total_loss/(i+1):.4f}")
    return math.exp(total_loss / nsamples)


def collect_hessians(model, loader, device):
    """Collect input Hessians for every linear layer of ``model``."""
    hooks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks[name] = utils.register_H_hook(module, device)
    for inp, _ in tqdm(loader, desc="collecting activations", leave=False):
        model(inp.to(device))
    hessians = {}
    for name, done in hooks.items():
        H, mu, ct = done()
        mu.div_(ct)
        H.div_(ct)
        H.addmm_(-mu.unsqueeze(1), mu.unsqueeze(0))
        hessians[name] = H
    return hessians


def apply_quantization(model, hessians, cb, device, qargs):
    """Apply QUIP# quantization layer by layer."""
    for name, module in tqdm(list(model.named_modules()),
                             desc="quantizing",
                             leave=False):
        if not isinstance(module, torch.nn.Linear):
            continue
        H = hessians[name].to(device)
        n = H.shape[0]
        H = utils.regularize_H(H, n, qargs.sigma_reg)
        W = module.weight.detach().to(torch.float32)
        scale = W.square().mean().sqrt()
        if qargs.scale_override > 0:
            scale /= qargs.scale_override
        else:
            scale /= cb.opt_scale
        W = W / scale
        hatW, _ = quantize(H, W, qargs.lora_rank, cb, qargs, device)
        hatW = hatW * scale
        module.weight.data.copy_(hatW.to(module.weight.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='HF model id')
    parser.add_argument('--codebook', default='E8P12', help='QUIP# codebook')
    parser.add_argument('--dataset', default='wikitext2')
    parser.add_argument('--nsamples', type=int, default=32,
                        help='samples for Hessian estimation')
    parser.add_argument('--seqlen', type=int, default=128,
                        help='sequence length for data sampling')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.to(args.device)

    # evaluation dataset
    test_tokens = gptq_data_utils.get_test_tokens(args.dataset,
                                                 seqlen=args.seqlen,
                                                 model=tokenizer.name_or_path)
    ppl_fp = evaluate_perplexity(model, test_tokens, args.seqlen, args.device)

    # collect hessian statistics
    loader, _ = gptq_data_utils.get_loaders(args.dataset,
                                            nsamples=args.nsamples,
                                            seed=0,
                                            seqlen=args.seqlen,
                                            model=tokenizer.name_or_path)
    hessians = collect_hessians(model, loader, args.device)

    # quantization arguments
    qargs = SimpleNamespace(use_fp64=False,
                            sigma_reg=1e-2,
                            sigma_reg2=1e-2,
                            incoh_mode='had',
                            lora_rank=0,
                            rescale_WH=False,
                            full_svd=False,
                            scale_override=-1,
                            resid_scale_override=-1,
                            no_use_buffered=False,
                            lowmem_ldlq=False)
    cb = codebook.get_codebook(args.codebook)
    apply_quantization(model, hessians, cb, args.device, qargs)

    ppl_q = evaluate_perplexity(model, test_tokens, args.seqlen, args.device)

    print(f"FP16 perplexity: {ppl_fp:.4f}")
    print(f"Quantized perplexity: {ppl_q:.4f}")


if __name__ == '__main__':
    main()

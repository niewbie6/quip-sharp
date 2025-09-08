#!/usr/bin/env bash
# Run QUIP# quantization and perplexity evaluation on OLMo 1B
python scripts/run_quip.py --model allenai/OLMo-1B-hf --codebook E8P12 --dataset wikitext2 --seqlen 512 --nsamples 32

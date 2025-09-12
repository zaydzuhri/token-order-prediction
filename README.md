# token-order-prediction
From the paper "Predicting the Order of Upcoming Tokens Improves Language Modeling"

ArXiv: https://arxiv.org/abs/2508.19228

This repository contains the code for:
1. Naive pytorch implementation of ListNet loss.
2. Optimized triton implementation of fused linear ListNet loss.
3. Naive pytorch implementation of token sequence to TOP target sequence conversion function.
4. Optimized triton implementation of token sequence to TOP target sequence conversion function.

For the training code used in the paper, see: https://github.com/zaydzuhri/flame/tree/token-order-prediction

The Flame repository links to a fork of the flash-linear-attention repository where the actual architectural modifications are, see: https://github.com/zaydzuhri/flash-linear-attention/tree/token-order-prediction

All trained models and checkpoints are on my huggingface: https://huggingface.co/zaydzuhri 

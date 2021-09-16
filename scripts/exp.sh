#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m bt+hsic2 --lambda_hsic2 1e4 &
CUDA_VISIBLE_DEVICES=1 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m bt+hsic2 --lambda_hsic2 1e5 &
CUDA_VISIBLE_DEVICES=2 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m bt+hsic2 --lambda_hsic2 1e6 &
CUDA_VISIBLE_DEVICES=3 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m bt+hsic2 --lambda_hsic2 1e7 &
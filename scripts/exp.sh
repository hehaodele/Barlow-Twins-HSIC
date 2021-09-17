#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewOut --lambda_hsic 1e2 &
#CUDA_VISIBLE_DEVICES=1 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewOut --lambda_hsic 3e2 &
#CUDA_VISIBLE_DEVICES=2 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewOut --lambda_hsic 1e3 &
#CUDA_VISIBLE_DEVICES=3 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewOut --lambda_hsic 3e3 &
CUDA_VISIBLE_DEVICES=0 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewFea --lambda_hsic 1 &
CUDA_VISIBLE_DEVICES=1 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewFea --lambda_hsic 3 &
CUDA_VISIBLE_DEVICES=2 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewFea --lambda_hsic 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --lmbda 0.0078125 --batch_size 128 --feature_dim 128 --dataset cifar10 -m btHistNewFea --lambda_hsic 30 &
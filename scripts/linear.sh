#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python linear.py --dataset cifar10 --ckpt_name Mbt+hsic2_Off0.00_L1_20210913-222-12_model_450 &
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset cifar10 --ckpt_name Mbt+hsic2_Off0.00_L10_20210913-222-12_model_450 &
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset cifar10 --ckpt_name Mbt+hsic2_Off0.00_L100_20210913-222-12_model_450 &
CUDA_VISIBLE_DEVICES=3 python linear.py --dataset cifar10 --ckpt_name Mbt+hsic2_Off0.00_L1000_20210914-1414-27_model_450 &

import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils

import torchvision

import numpy as np
from main import  calcHSIC
# train or test for one epoch
def train_val(net, data_loader, phase):
    net.eval()

    data_bar = tqdm(data_loader)

    with torch.no_grad():

        total_out = []
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            _, out = net(data)
            total_out.append(out.cpu())

    out = torch.cat(total_out)
    out = (out - out.mean(dim=0, keepdim=True)) / out.std(dim=0, keepdim=True, unbiased=False)
    corr = torch.matmul(out.T, out) / len(out)
    mask = torch.eye(len(corr)).to(torch.bool).to(corr.device)
    abs_corr = torch.abs(corr)

    max_idx = [0,1]
    min_idx = [0,1]

    n = len(corr)
    for i in range(n):
        for j in range(i):
            if abs_corr[i,j] > abs_corr[max_idx[0], max_idx[1]]:
                max_idx = [i,j]
            if abs_corr[i,j] < abs_corr[min_idx[0], min_idx[1]]:
                min_idx = [i,j]

    def pair_HSIC(x, y):
        bs=512
        hsic = []
        for i in range(0, len(x), bs):
            hsic += [calcHSIC(x[i:i+bs,None], y[i:i+bs,None]).item()]
        return np.mean(hsic)

    def pair_corr(x, y):
        return (x*y).mean().item()

    def get_R2(out, mode='linear'):
        if mode == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif mode == 'svr':
            from sklearn.svm import SVR
            model = SVR()
        else:
            assert False

        R2 = []
        out_np = out.numpy()
        n = out_np.shape[-1]
        line = np.arange(n)
        out_np = out.numpy()
        for i in tqdm(range(n)):
            X, y = out_np[:, line!=i], out_np[:, i]
            print(X.shape, y.shape)
            model.fit(X,y)
            R2.append(model.score(X, y))

        return R2

    R_linear, R_svr = get_R2(out, 'linear'), get_R2(out, 'svr')

    ret = {
        'max_idx': max_idx,
        'min_idx': min_idx,
        'max_hsic': pair_HSIC(out[:,max_idx[0]],out[:,max_idx[1]]),
        'min_hsic': pair_HSIC(out[:, min_idx[0]], out[:, min_idx[1]]),
        'max_corr': pair_corr(out[:, max_idx[0]], out[:, max_idx[1]]),
        'min_corr': pair_corr(out[:, min_idx[0]], out[:, min_idx[1]]),
        'corr': corr,
        'max_out': (out[:,max_idx[0]],out[:,max_idx[1]]),
        'min_out': (out[:, min_idx[0]], out[:, min_idx[1]]),
        'R_linear': torch.FloatTensor(R_linear),
        'R_svr': torch.FloatTensor(R_svr),
    }

    msg = f'[{phase}]\n'

    for a in ['max','min']:
        for b in ['idx', 'hsic', 'corr']:
            name = a+'_'+b
            msg += f' {name}: {ret[name]}'
        msg += '\n'
    print(msg)
    return ret


if __name__ == '__main__':
    from local import *
    from header import *
    from model import Model

    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--ckpt_name', type=str, default='0.005_64_128_model',
                        help='The base string of the pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')

    args = parser.parse_args()
    model_fn, batch_size = args.ckpt_name, args.batch_size
    model_fn = f'{model_path}/{model_fn}.pth'
    model = Model(dataset=args.dataset)
    model.load_state_dict(torch.load(model_fn, map_location='cpu'), strict=False)
    model.cuda()

    dataset = args.dataset
    if dataset == 'cifar10':
        train_data = CIFAR10(root=data_path, train=True, \
                             transform=utils.CifarPairTransform(train_transform=True, pair_transform=False),
                             download=True)
        test_data = CIFAR10(root=data_path, train=False, \
                            transform=utils.CifarPairTransform(train_transform=False, pair_transform=False),
                            download=True)
    elif dataset == 'stl10':
        train_data = torchvision.datasets.STL10(root=data_path, split="train", \
                                                transform=utils.StlPairTransform(train_transform=True,
                                                                                 pair_transform=False), download=True)
        test_data = torchvision.datasets.STL10(root=data_path, split="test", \
                                               transform=utils.StlPairTransform(train_transform=False,
                                                                                pair_transform=False), download=True)
    elif dataset == 'tiny_imagenet':
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform=True,
                                                                                      pair_transform=False))
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', \
                                                     utils.TinyImageNetPairTransform(train_transform=False,
                                                                                     pair_transform=False))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    for param in model.f.parameters():
        param.requires_grad = False

    if dataset == 'cifar10':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    elif dataset == 'tiny_imagenet' or dataset == 'stl10':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    save_folder = f'{result_path}/corr_and_hsic'
    os.system(f'mkdir -p {save_folder}')
    save_path = f'{save_folder}/{args.ckpt_name}.pth'

    results = {}
    test_corr = train_val(model, test_loader, 'test_')
    torch.save(test_corr, save_path)

    fig, ax = plt.subplots(1,2, figsize=(6,3))
    out = test_corr['max_out']
    corr, hsic = test_corr['max_corr'], test_corr['max_hsic']
    ax[0].plot(out[0], out[1],'o',alpha=0.02)
    ax[0].set_title(f'corr {corr:.4f} hsic {hsic:.4f}')
    out = test_corr['min_out']
    corr, hsic = test_corr['min_corr'], test_corr['min_hsic']
    ax[1].plot(out[0], out[1],'o',alpha=0.02)
    ax[1].set_title(f'corr {corr:.4f} hsic {hsic:.4f}')
    save_path = f'{save_folder}/{args.ckpt_name}_extreme.png'
    plt.savefig(save_path)

    fig, ax = plt.subplots(1,2, figsize=(6,3))
    R = test_corr['R_linear']
    ax[0].hist(R, range=(0,1), bins=20, density=True)
    ax[0].set_title(f'R-linear mean {R.mean():.4f} std {R.std():.4f}')
    R = test_corr['R_svr']
    ax[1].hist(R, range=(0,1), bins=20, density=True)
    ax[1].set_title(f'R-svr mean {R.mean():.4f} std {R.std():.4f}')
    save_path = f'{save_folder}/{args.ckpt_name}_r2.png'
    plt.savefig(save_path)


    # results['test_corr'] = test_corr
    # train_corr = train_val(model, train_loader, 'train')
    # results['train_corr'] = train_corr.cpu()

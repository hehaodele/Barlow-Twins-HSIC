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

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, dataset):
        super(Net, self).__init__()

        # encoder
        from model import Model
        self.f = Model(dataset=dataset).f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% model: {}'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100,
                                             model_fn.split('/')[-1]))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    from local import *
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--ckpt_name', type=str, default='0.005_64_128_model',
                        help='The base string of the pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_fn, batch_size, epochs = args.ckpt_name, args.batch_size, args.epochs
    model_fn = f'{model_path}/{model_fn}.pth'
    dataset = args.dataset
    if dataset == 'cifar10':
        train_data = CIFAR10(root=data_path, train=True,\
            transform=utils.CifarPairTransform(train_transform = True, pair_transform=False), download=True)
        test_data = CIFAR10(root=data_path, train=False,\
            transform=utils.CifarPairTransform(train_transform = False, pair_transform=False), download=True)
    elif dataset == 'stl10':
        train_data =  torchvision.datasets.STL10(root=data_path, split="train", \
            transform=utils.StlPairTransform(train_transform = True, pair_transform=False), download=True)
        test_data =  torchvision.datasets.STL10(root=data_path, split="test", \
            transform=utils.StlPairTransform(train_transform = False, pair_transform=False), download=True)
    elif dataset == 'tiny_imagenet':
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                            utils.TinyImageNetPairTransform(train_transform=True, pair_transform=False))
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', \
                            utils.TinyImageNetPairTransform(train_transform = False, pair_transform=False))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_fn, dataset=dataset).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    if dataset == 'cifar10':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    elif dataset == 'tiny_imagenet' or dataset == 'stl10':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    linear_folder = f'{result_path}/linear'
    os.system(f'mkdir -p {linear_folder}')
    save_name = f'{result_path}/linear/{args.ckpt_name}_linear.csv'

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(save_name, index_label='epoch')
        #if test_acc_1 > best_acc:
        #    best_acc = test_acc_1
        #    torch.save(model.state_dict(), 'results/linear_model.pth')

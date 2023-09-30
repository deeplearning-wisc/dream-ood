# -*- coding: utf-8 -*-
import copy

import numpy as np
import os

import argparse
import time
import torch

import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F


from resnet import ResNet_Model




from utils.validation_dataset import validation_split
from utils.out_dataset import RandomImages50k

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='in100',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=160, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=8, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='energy', help='OE|energy')
parser.add_argument('--add_slope', type=int, default=0)
parser.add_argument('--add_class', type=int, default=0)
parser.add_argument('--vanilla', type=int, default=1)
parser.add_argument('--oe', type=int, default=0)
parser.add_argument('--apex', type=int, default=0)
parser.add_argument('--r50', type=int, default=0)
parser.add_argument('--augmix', type=int, default=0)
parser.add_argument('--cutmix', type=int, default=0)
parser.add_argument('--use_subset', type=int, default=0)
parser.add_argument('--randaugment', type=int, default=0)
parser.add_argument('--autoaugment', type=int, default=0)
parser.add_argument('--deepaugment', type=int, default=0)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--my_info', type=str, default='')
parser.add_argument('--additional_info', type=str, default='')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--energy_weight', type=float, default=1)  # change this to 19.2 if you are using cifar-100.
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
args = parser.parse_args()

if args.score == 'OE':
    save_info = 'oe_tune'
elif args.score == 'energy':
    save_info = 'energy_ft_sd'
if args.apex:
    # apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    amp.register_float_function(torch, 'sigmoid')

args.save = args.save + save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])



traindir = os.path.join('/nobackup/dataset/my_xfdu/IN100_new/', 'train')
valdir = os.path.join('/nobackup/dataset/my_xfdu/IN100_new/', 'val')


normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if args.augmix:
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
        )
    train_data_in_aug = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.AugMix(),
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
    )
    import random
    train_data_in_aug = torch.utils.data.Subset(train_data_in_aug,
                                            random.sample(range(len(train_data_in_aug)), 108000))
    train_data_in = torch.utils.data.ConcatDataset([train_data_in, train_data_in_aug])
    # breakpoint()
elif args.randaugment:
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
    )
    train_data_in_aug = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandAugment(),
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
    )
    import random
    train_data_in_aug = torch.utils.data.Subset(train_data_in_aug,
                                                random.sample(range(len(train_data_in_aug)), 108000))
    train_data_in = torch.utils.data.ConcatDataset([train_data_in, train_data_in_aug])

elif args.autoaugment:
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
    )
    train_data_in_aug = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.AutoAugment(trn.AutoAugmentPolicy.IMAGENET),
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ])
    )
    import random
    train_data_in_aug = torch.utils.data.Subset(train_data_in_aug,
                                                random.sample(range(len(train_data_in_aug)), 108000))
    train_data_in = torch.utils.data.ConcatDataset([train_data_in, train_data_in_aug])


else:
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
    if args.use_subset:
        import random
        # breakpoint()
        train_data_in = torch.utils.data.Subset(train_data_in,
                                          random.sample(range(len(train_data_in)), 70774))
if args.deepaugment:
    edsr_dataset = torchvision.datasets.ImageFolder(
        '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/EDSR/',
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))

    cae_dataset = torchvision.datasets.ImageFolder(
        '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/CAE/',
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
    train_data_in = torch.utils.data.ConcatDataset([train_data_in, edsr_dataset, cae_dataset])
test_data = torchvision.datasets.ImageFolder(
    valdir,
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
num_classes = 100


calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'





prefix = '/nobackup-slow/dataset/'

ood_data = dset.ImageFolder(root=prefix + "my_xfdu/sd/txt2img-samples-in100/"  + args.my_info + '/samples',
                            transform=trn.Compose([trn.RandomResizedCrop(224),
trn.RandomHorizontalFlip(),
trn.ToTensor(),
normalize,]))
map = ood_data.class_to_idx
map1 = {}
for key in list(map.keys()):
    map1[map[key]] = int(key)
# breakpoint()
if args.augmix:
    ood_data = dset.ImageFolder(
        root=prefix + "my_xfdu/sd/txt2img-samples-in100/" + args.my_info + '/samples',
        transform=trn.Compose([trn.AugMix(),
                               trn.RandomResizedCrop(224),
                               trn.RandomHorizontalFlip(),
                               trn.ToTensor(),
                               normalize, ]),
        target_transform=lambda id: map1[id])
else:
    ood_data = dset.ImageFolder(
        root=prefix + "my_xfdu/sd/txt2img-samples-in100/" + args.my_info + '/samples',
        transform=trn.Compose([trn.RandomResizedCrop(224),
                               trn.RandomHorizontalFlip(),
                               trn.ToTensor(),
                               normalize, ]),
        target_transform=lambda id: map1[id])

if args.use_subset:
    import random
    ood_data = torch.utils.data.Subset(ood_data,
                                      random.sample(range(len(ood_data)), 59086))




train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.r50:
    net = ResNet_Model(name='resnet50', num_classes=num_classes)
else:
    net = ResNet_Model(name='resnet34', num_classes=num_classes)
# net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


# Restore model
model_found = False
if args.load != '':

    pretrained_weights = torch.load(args.load)
    # breakpoint()
    for item in list(pretrained_weights.keys()):
        pretrained_weights[item[7:]] = pretrained_weights[item]
        del pretrained_weights[item]
    net.load_state_dict(pretrained_weights, strict=True)




optimizer = torch.optim.SGD(
    list(net.parameters()),
    state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

net.cuda()
if args.ngpu > 1:

    if args.apex:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1", loss_scale=1.0)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

# /////////////// Training ///////////////
criterion = torch.nn.CrossEntropyLoss()


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    loss_energy_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    # train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    batch_iterator = iter(train_loader_out)
    for _, in_set in enumerate(train_loader_in):
        # print(in_set[0])
        # print(out_set[0])

        try:
            out_set = next(batch_iterator)
        except StopIteration:
            # train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
            batch_iterator = iter(train_loader_out)
            out_set = next(batch_iterator)

        data = torch.cat((in_set[0], out_set[0]), 0)
        target = torch.cat((in_set[1], out_set[1]), 0)

        permutation_idx = torch.randperm(len(data))
        data = data[permutation_idx]
        target = target[permutation_idx]
        data, target = data.cuda(), target.cuda()
        # breakpoint()

        # forward
        x = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        # exponential moving average

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    print(scheduler.get_lr())
    state['train_loss'] = loss_avg
    state['train_energy_loss'] = loss_energy_avg



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)


save_info = save_info + "_slope_" + str(args.add_slope) + '_' + "weight_" + str(args.energy_weight)
save_info = save_info + '_' + args.my_info + '_real' + '_' + args.additional_info

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
loss_min = 100
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()


    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch) + '.pt'))

    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                             '_' + save_info + '_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results
    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))


    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )

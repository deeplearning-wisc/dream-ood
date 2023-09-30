import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn

from resnet import ResNet_Model

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=160)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--choice', type=str, default='vanilla')
args = parser.parse_args()
print(args)


# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



num_classes= 100
net = ResNet_Model(name='resnet34', num_classes=num_classes)
start_epoch = 0

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        subdir = 'energy_ft_sd'
        model_name = args.load

        if os.path.isfile(model_name):
            net.load_state_dict(remove_data_parallel(torch.load(model_name)))
            # net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume " + model_name

net.eval()


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders



def acc_print(test_loader):

    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)

    return 100 - 100 * num_wrong / (num_wrong + num_right)



# #imagenet-v2
test_data = \
    torchvision.datasets.ImageFolder(
    '/nobackup-slow/dataset/my_xfdu/imagenetv2/processed/',
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)
print(acc_print(test_loader))


#imagenet-a
test_data = \
    torchvision.datasets.ImageFolder(
    '/nobackup-slow/dataset/my_xfdu/imageneta/processed/',
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
id_mapping = test_data.class_to_idx
new_mapping = {}
for key in list(id_mapping.keys()):
    new_mapping[id_mapping[key]] = int(key)
# breakpoint()
test_data = \
    torchvision.datasets.ImageFolder(
    '/nobackup-slow/dataset/my_xfdu/imageneta/processed/',
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]), target_transform=lambda id: new_mapping[id])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)
print(acc_print(test_loader))




test_data = \
    torchvision.datasets.ImageFolder(
    os.path.join('/nobackup/dataset/my_xfdu/IN100_new/', 'val'),
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)
print(acc_print(test_loader))






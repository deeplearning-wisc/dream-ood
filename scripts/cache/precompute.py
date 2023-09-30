from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
# import models.densenet as dn
from wrn import WideResNet

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', '-d', default='imagenet', type=str, help='dataset')

args = parser.parse_args()

num_classes = 100
# from models.resnet import resnet50
from resnet import ResNet_Model
net = ResNet_Model(name='resnet34', num_classes=num_classes)
featdim = 512

net = net
from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 100
test_batch_size = 100
model_name = '/afs/cs.wisc.edu/u/x/f/xfdu/workspace/stable-diffusion/snapshots/energy_ft_sd/in100_wrn_s1_energy_ft_sd_slope_0_weight_1_keep_train_vanilla__epoch_19.pt'
# model_name = '/afs/cs.wisc.edu/u/x/f/xfdu/workspace/stable-diffusion/snapshots/energy_ft_sd/in100_wrn_s1_energy_ft_sd_slope_0_weight_1_outlier_npos_50_step_epoch_19.pt'
if os.path.isfile(model_name):
    net.load_state_dict(remove_data_parallel(torch.load(model_name)))

net = net.to(device)
transform_test_largescale = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(os.path.join('/nobackup-slow/dataset/my_xfdu/IN100_new/', 'train'), transform_test_largescale),
    batch_size=test_batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(os.path.join('/nobackup-slow/dataset/my_xfdu/IN100_new/', 'val'), transform_test_largescale),
    batch_size=test_batch_size, shuffle=True, num_workers=2, pin_memory=True)
id_train_size = 129860

feat_log = np.zeros((id_train_size, featdim))
score_log = np.zeros((id_train_size, num_classes))
label_log = np.zeros(id_train_size)

net.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        start_ind = batch_idx * batch_size
        end_ind = min((batch_idx + 1) * batch_size, len(trainloader.dataset))

        out, score = net.forward_repre(inputs)
        # outputs = net.features(inputs)
        # out = F.adaptive_avg_pool2d(outputs, 1)
        # out = out.view(out.size(0), -1)
        score = net.fc(out)
        # score = net(inputs)
        feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
        label_log[start_ind:end_ind] = targets.data.cpu().numpy()
        score_log[start_ind:end_ind] = score.data.cpu().numpy()
        if batch_idx % 10 == 0:
            print(f"{batch_idx}/{len(trainloader)}")


np.save(f"cache/{args.dataset}_resnet34_feat_stat.npy", feat_log.mean(0))
print("done")
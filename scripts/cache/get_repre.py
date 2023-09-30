import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision
from resnet import ResNet_Model
from PIL import Image as PILImage


# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import utils.svhn_loader as svhn
import utils.lsun_loader as lsun_loader
import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
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
args = parser.parse_args()
print(args)
# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar100_' in args.method_name:
    test_data = dset.CIFAR100('/nobackup/my_xfdu/cifarpy', train=False, transform=test_transform)
    train_data = dset.CIFAR100('/nobackup/my_xfdu/cifarpy', train=True, transform=test_transform)
    num_classes = 100
else:
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    test_data = \
        torchvision.datasets.ImageFolder(
            os.path.join('/nobackup/dataset/my_xfdu/IN100_new/', 'val'),
            trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                normalize,
            ]))
    train_data = \
        torchvision.datasets.ImageFolder(
            os.path.join('/nobackup/dataset/my_xfdu/IN100_nwe/', 'train'),
            trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                normalize,
            ]))
    num_classes = 100

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

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
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        elif 'energy_ft_sd' in args.method_name:
            subdir = 'energy_ft_sd'
        else:
            subdir = 'oe_scratch'

        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
        # if i == 9:
        #     breakpoint()
        if os.path.isfile(model_name):
            if 'cifar100_' not in args.method_name:
                net.load_state_dict(remove_data_parallel(torch.load(model_name)))
            else:
                net.load_state_dict(torch.load(model_name))
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

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()



def return_repre(loader, ood=False):
    repre_all = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if ood:
                if batch_idx > 21:
                    break
            data = data.cuda()
            repre, output = net.forward_repre(data)
            if repre_all is None:
                repre_all = repre
            else:
                repre_all = torch.cat([repre_all, repre], 0)
    return repre_all

def return_repre_energy(loader):
    repre_all = None
    score_all = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.cuda()
            repre, output = net.forward_repre(data)
            score = -torch.logsumexp(output, dim=1)
            if repre_all is None:
                repre_all = repre
                score_all = score
            else:
                repre_all = torch.cat([repre_all, repre], 0)
                score_all = torch.cat([score_all, score], 0)
    return repre_all, score_all

if 1:
    ood_data = dset.ImageFolder(root="/nobackup/my_xfdu/sd/txt2img-samples-cifar100/txt2img-samples_select_50_sigma_0.07/",
                                    transform=trn.Compose([trn.ToTensor(), trn.ToPILImage(),
                                                           trn.Resize(32),
                                                           trn.RandomCrop(32, padding=4),
                                                           trn.RandomHorizontalFlip(), trn.ToTensor(),
                                                           trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.test_bs, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)

    # ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/places365/",
    #                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
    #                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
    # ood_place_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
    #                                          num_workers=2, pin_memory=True)
    ood_data = dset.ImageFolder(root="/nobackup/dtd_jnakhleh/dtd/images",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_place_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=4, pin_memory=True)
    # ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/iSUN",
    #                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    # ood_place_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
    #                                          num_workers=1, pin_memory=True)
    #
    # ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/LSUN_C",
    #                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    # ood_place_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
    #                                          num_workers=1, pin_memory=True)

    # ood_data = svhn.SVHN(root='/nobackup-slow/dataset/places365/', split="test",
    #                      transform=trn.Compose(
    #                          [  # trn.Resize(32),
    #                              trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
    # ood_place_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
    #                                          num_workers=2, pin_memory=True)


    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    train_repre = return_repre(train_loader)
    test_repre = return_repre(test_loader)
    ood_repre = return_repre(ood_loader)
    ood_place_repre = return_repre(ood_place_loader, ood=True)

    train_repre = normalizer(train_repre.cpu().data.numpy())
    test_repre = normalizer(test_repre.cpu().data.numpy())
    ood_repre = normalizer(ood_repre.cpu().data.numpy())
    ood_place_repre = normalizer(ood_place_repre.cpu().data.numpy())
    import faiss

    # index = faiss.IndexFlatL2(train_repre.shape[1])
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, train_repre.shape[1])

    index.add(train_repre)

    K_in_KNN = 20

    D, _ = index.search(test_repre, K_in_KNN)
    in_score = -D[:,-1]

    D, _ = index.search(ood_repre, K_in_KNN)
    out_score = -D[:,-1]

    D, _ = index.search(ood_place_repre, K_in_KNN)
    out_place_score = -D[:, -1]

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.figure(figsize=(5.5,3))
    id_pd = pd.Series(-in_score)
    ood_pd = pd.Series(-out_score)
    place_pd = pd.Series(-out_place_score)
    # breakpoint()
    p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
    p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='Generated')
    p1 = sns.kdeplot(place_pd, shade=True, color="g", label='place')

    plt.legend()
    plt.savefig('cifar100_NPOS_select_50_sigma_0.07_w_dtd.jpg', dpi=250)
else:
    # import shutil
    # ood_data = dset.ImageFolder(root="/nobackup/my_xfdu/sd/txt2img-samples-in100/gaussian_0.6",
    #                             transform=trn.Compose([trn.Resize(256),
    #                                                    trn.CenterCrop(224),
    #                                                    trn.ToTensor(),
    #                                                    normalize]))
    # ood_loader = torch.utils.data.DataLoader(
    #     ood_data,
    #     batch_size=args.test_bs, shuffle=False,
    #     num_workers=args.prefetch, pin_memory=True)
    #
    #
    #
    # index = 0
    # filtered_mask = np.load('./filtered_mask.npy')
    # for file_name, _ in ood_loader.dataset.samples:
    #     if filtered_mask[index]:
    #         os.makedirs(file_name.replace('gaussian_0.6', 'gaussian_0.6_filtered_another'), exist_ok=True)
    #         shutil.copy(file_name, file_name.replace('gaussian_0.6', 'gaussian_0.6_filtered_another'))
    #         # breakpoint()
    #     index += 1
    # breakpoint()

    # normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    # train_repre = return_repre(train_loader)
    # test_repre = return_repre(test_loader)
    # ood_repre = return_repre(ood_loader)
    #
    # train_repre = normalizer(train_repre.cpu().data.numpy())
    # test_repre = normalizer(test_repre.cpu().data.numpy())
    # ood_repre = normalizer(ood_repre.cpu().data.numpy())
    # import faiss
    #
    # index = faiss.IndexFlatL2(train_repre.shape[1])
    # index.add(train_repre)
    #
    # K_in_KNN = 20
    #
    # D, _ = index.search(test_repre, K_in_KNN)
    # in_score = -D[:, -1]
    #
    # D, _ = index.search(ood_repre, K_in_KNN)
    # out_score = -D[:, -1]
    #
    # threshold = np.percentile(in_score, 0.95)
    # filtered_mask = out_score < threshold
    # breakpoint()
    # with torch.no_grad():
    #     for batch_idx, (data, target) in enumerate(ood_loader):
    #         data = data.cuda()
    #         breakpoint()
    #         # repre, output = net.forward_repre(data)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    #
    # plt.figure(figsize=(5.5, 3))
    # id_pd = pd.Series(-in_score)
    # ood_pd = pd.Series(-out_score)
    # breakpoint()
    # p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
    # p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='OOD')
    # plt.legend()
    # plt.savefig('in100_vs_gaussian0.5_k_20.jpg', dpi=250)


    # for real ood
    # place365 = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Places/",
    #                             transform=trn.Compose([
    #                                 trn.Resize(256),
    #                                 trn.CenterCrop(224),
    #                                 trn.ToTensor(),
    #                                 trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                                               std=[0.229, 0.224, 0.225]),
    #                             ]))
    # inat
    # place365 = torchvision.datasets.ImageFolder("/nobackup-slow/dataset/ImageNet_OOD_dataset/iNaturalist",
    #                                  transform=trn.Compose([
    #                                      trn.Resize(256),
    #                                      trn.CenterCrop(224),
    #                                      trn.ToTensor(),
    #                                      trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                    std=[0.229, 0.224, 0.225]),
    #                                  ]))
    # place365 = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/SUN/",
    #                             transform=trn.Compose([
    #     trn.Resize(256),
    #     trn.CenterCrop(224),
    #     trn.ToTensor(),
    #     trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ]))
    place365 = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Textures/",
                                transform=trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/my_xfdu/sd/txt2img-samples-in100/gaussian_0.3/",
                                transform=trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                   num_workers=2, pin_memory=True)
    ood_place_loader = torch.utils.data.DataLoader(place365, batch_size=args.test_bs, shuffle=True,
                                             num_workers=2, pin_memory=True)
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    ood_place_repre = return_repre(ood_place_loader, ood=True)
    ood_repre = normalizer(return_repre(ood_loader, ood=False).cpu().data.numpy())
    train_repre = np.load('in100_train.npy')
    test_repre = np.load('in100_test.npy')
    # ood_repre = np.load('ood_0.6.npy')
    ood_place_repre = normalizer(ood_place_repre.cpu().data.numpy())
    import faiss

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, train_repre.shape[1])

    # index = faiss.IndexFlatL2(train_repre.shape[1])
    index.add(train_repre)

    K_in_KNN = 20

    D, _ = index.search(test_repre, K_in_KNN)
    in_score = -D[:, -1]

    D, _ = index.search(ood_repre, K_in_KNN)
    out_score = -D[:, -1]

    D, _ = index.search(ood_place_repre, K_in_KNN)
    out_place_score = -D[:, -1]
    # breakpoint()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    plt.figure(figsize=(5.5, 3))
    id_pd = pd.Series(-in_score)
    ood_pd = pd.Series(-out_score)
    place_pd = pd.Series(-out_place_score)
    # breakpoint()
    p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
    p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='Generated filtered')
    p1 = sns.kdeplot(place_pd, shade=True, color="g", label='text')
    plt.legend()
    plt.savefig('in100_vs_gaussian0.3_k_20_add_text1.jpg', dpi=250)

    # for real ood
    # place365 = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Places/",
    #                             transform=trn.Compose([
    #                                 trn.Resize(256),
    #                                 trn.CenterCrop(224),
    #                                 trn.ToTensor(),
    #                                 trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                                               std=[0.229, 0.224, 0.225]),
    #                             ]))
    # inat
    # place365 = torchvision.datasets.ImageFolder("/nobackup-slow/dataset/ImageNet_OOD_dataset/iNaturalist",
    #                                  transform=trn.Compose([
    #                                      trn.Resize(256),
    #                                      trn.CenterCrop(224),
    #                                      trn.ToTensor(),
    #                                      trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                    std=[0.229, 0.224, 0.225]),
    #                                  ]))
    #     place365 = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/SUN/",
    #                             transform=trn.Compose([
    #     trn.Resize(256),
    #     trn.CenterCrop(224),
    #     trn.ToTensor(),
    #     trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ]))
    # ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/my_xfdu/sd/txt2img-samples-in100/gaussian_0.6",
    #                             transform=trn.Compose([trn.Resize(256),
    #                                                    trn.CenterCrop(224),
    #                                                    trn.ToTensor(),
    #                                                    normalize]))
    # ood_loader = torch.utils.data.DataLoader(
    #     ood_data,
    #     batch_size=args.test_bs, shuffle=False,
    #     num_workers=args.prefetch, pin_memory=True)
    #
    # id_repre, in_score = return_repre_energy(test_loader)
    # ood_repre, out_score = return_repre_energy(ood_loader)
    # breakpoint()
    # place365 = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Textures/",
    #                             transform=trn.Compose([
    #                                 trn.Resize(256),
    #                                 trn.CenterCrop(224),
    #                                 trn.ToTensor(),
    #                                 trn.Normalize(mean=[0.485, 0.456, 0.406],
    #                                               std=[0.229, 0.224, 0.225]),
    #                             ]))
    # ood_place_loader = torch.utils.data.DataLoader(place365, batch_size=args.test_bs, shuffle=True,
    #                                                num_workers=2, pin_memory=True)
    # # normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    #
    # ood_place_repre, out_place_score = return_repre(ood_place_loader)
    #
    #
    #
    #
    #
    #
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    #
    # plt.figure(figsize=(5.5, 3))
    # id_pd = pd.Series(-in_score)
    # ood_pd = pd.Series(-out_score)
    # place_pd = pd.Series(-out_place_score)
    # # breakpoint()
    # p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
    # p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='Generated')
    # p1 = sns.kdeplot(place_pd, shade=True, color="g", label='place365')
    # plt.legend()
    # plt.savefig('in100_vs_gaussian0.5_k_20_add_text_energy.jpg', dpi=250)
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
from resnet import ResNet_Model
from PIL import Image as PILImage
import torchvision

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

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)
    num_classes = 100
test_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform)
num_classes = 10
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
train_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)


# normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# test_data = \
#     torchvision.datasets.ImageFolder(
#     os.path.join('/nobackup-slow/dataset/my_xfdu/IN100_new/', 'val'),
#     trn.Compose([
#         trn.Resize(256),
#         trn.CenterCrop(224),
#         trn.ToTensor(),
#         normalize,
#     ]))
#
# train_data = \
#     torchvision.datasets.ImageFolder(
#     os.path.join('/nobackup-slow/dataset/my_xfdu/IN100_new/', 'train'),
#     trn.Compose([
#         trn.Resize(256),
#         trn.CenterCrop(224),
#         trn.ToTensor(),
#         normalize,
#     ]))
# num_classes = 100
#
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
#                                           num_workers=args.prefetch, pin_memory=True)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=True,
#                                           num_workers=args.prefetch, pin_memory=True)

# Create model
from wrn import WideResNet
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
# net = ResNet_Model(name='resnet34', num_classes=num_classes)
# from resnet_rebuttal import resnet18_cifar
# net = resnet18_cifar()
# from resnet_rebuttal import resnet34
# net = resnet34()

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

        # model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
        model_name = '/afs/cs.wisc.edu/u/x/f/xfdu/workspace/energy_ood_new/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt'
        # model_name = '/afs/cs.wisc.edu/u/x/f/xfdu/workspace/stable-diffusion/snapshots/energy_ft_sd/in100_wrn_s1_energy_ft_sd_slope_0_weight_1_keep_train_vanilla__epoch_19.pt'
        # # if i == 9:
        # #     breakpoint()
        # if os.path.isfile(model_name):
        #     net.load_state_dict(torch.load(model_name)['state_dict'])
        #     print('Model restored! Epoch:', i)
        #     start_epoch = i + 1
        #     break

        # model_name = '/nobackup-slow/dataset/my_xfdu/101_MS_epoch_100.pt'
        if os.path.isfile(model_name):
            # net.load_state_dict(remove_data_parallel(torch.load(model_name)))
            net.load_state_dict(torch.load(model_name))

            # model_name = '/nobackup-slow/dataset/my_xfdu/ImageNet-100_baseline.pt'
            # if os.path.isfile(model_name):
            #     net.load_state_dict(torch.load(model_name))
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
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
# breakpoint()
import tqdm

# for grad norm#
with torch.no_grad():
    w, b = net.fc.weight, net.fc.bias
    print('Extracting id training feature')
    feature_id_train = []
    for _, batch in enumerate(train_loader):
        data = batch[0].cuda()
        data = data.float()
        # feat = net.features(data)
        # feat = net.avgpool(feat)
        # feature = feat.view(feat.size(0), -1)

        feature = net.features(data)
        feature_id_train.append(feature.cpu().numpy())
    feature_id_train = np.concatenate(feature_id_train, axis=0)
    logit_id_train = feature_id_train @ w.T.cpu().data.numpy() + b.cpu().data.numpy()

    print('Extracting id testing feature')
    feature_id_val = []
    for _, batch in enumerate(test_loader):
        data = batch[0].cuda()
        data = data.float()
        # feat = net.features(data)
        # feat = net.avgpool(feat)
        # feature = feat.view(feat.size(0), -1)

        feature = net.features(data)
        feature_id_val.append(feature.cpu().numpy())
    feature_id_val = np.concatenate(feature_id_val, axis=0)
    logit_id_val = feature_id_val @ w.T.cpu().data.numpy() + b.cpu().data.numpy()

u = -np.matmul(pinv(w.cpu().data.numpy()), b.cpu().data.numpy())
ec = EmpiricalCovariance(assume_centered=True)
ec.fit(feature_id_train - u)
eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
NS = np.ascontiguousarray(
    (eigen_vectors.T[np.argsort(eig_vals * -1)[64:]]).T)
# breakpoint()
vlogit_id_train = norm(np.matmul(feature_id_train - u, NS),
                               axis=-1)
alpha = logit_id_train.max(
    axis=-1).mean() / vlogit_id_train.mean()
print(f'{alpha=:.4f}')

vlogit_id_val = norm(np.matmul(feature_id_val - u, NS),
                     axis=-1) * alpha
from scipy.special import logsumexp
energy_id_val = logsumexp(logit_id_val, axis=-1)
score_id = -vlogit_id_val + energy_id_val

#
# def return_repre(loader, ood=False):
#     repre_all = None
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(loader):
#             if ood:
#                 if batch_idx > 5:
#                     break
#             data = data.cuda()
#             repre = net.features(data)
#             if repre_all is None:
#                 repre_all = repre
#             else:
#                 repre_all = torch.cat([repre_all, repre], 0)
#     return repre_all
# normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
# train_repre = return_repre(train_loader)
# train_repre = normalizer(train_repre.cpu().data.numpy())
#
# import faiss
#
# # index = faiss.IndexFlatL2(train_repre.shape[1])
# res = faiss.StandardGpuResources()
# index = faiss.GpuIndexFlatL2(res, train_repre.shape[1])
#
# index.add(train_repre)
#
# K_in_KNN = 100
#
#
# def get_knn_score(index, ood_loader, ood=False):
#     ood_repre = return_repre(ood_loader,ood)
#     ood_repre = normalizer(ood_repre.cpu().data.numpy())
#     D, _ = index.search(ood_repre, K_in_KNN)
#     score = -D[:, -1]
#     return -score

# in_score = get_knn_score(index, test_loader, False)



def vim_score(features, output):
    _, pred = torch.max(output, dim=1)
    energy_ood = logsumexp(output.cpu().numpy(), axis=-1)
    vlogit_ood = norm(np.matmul(features.cpu().numpy() - u, NS),
                      axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    return -score_ood

def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            # feat = net.features(data)
            # feat = net.avgpool(feat)
            # feature = feat.view(feat.size(0), -1)

            feature = net.features(data)

            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    # breakpoint()
                    _score.append(-to_np(
                            (args.T * torch.logsumexp(output / args.T, dim=1))))
                    # _score.append(-to_np(
                    #     net.logistic_regression(
                    #         (args.T * torch.logsumexp(output / args.T, dim=1)).unsqueeze(1)).sigmoid()))
                    # _score.append(to_np(F.softmax(output,1)[:,10]))
                elif args.score == 'vim':
                    _score.append(vim_score(feature, output))
                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()



in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)
in_score = -score_id


# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg, ood_name=None):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count - 1,
                                                  args.noise, num_batches)
        else:
            # out_score = get_knn_score(index, ood_loader, True)
            out_score = get_ood_scores(ood_loader)
            # breakpoint()
        if args.out_as_pos:  # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]);
        auprs.append(measures[1]);
        fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)




# # /////////////// inat ///////////////
# ood_loader = torch.utils.data.DataLoader(
#                 torchvision.datasets.ImageFolder("/nobackup-slow/dataset/ImageNet_OOD_dataset/iNaturalist",
#                                                  transform=trn.Compose([
#     trn.Resize(256),
#     trn.CenterCrop(224),
#     trn.ToTensor(),
#     trn.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])), batch_size=args.test_bs,
#                 shuffle=True,
#                 num_workers=2)
# print('\n\ninat')
# # breakpoint()
# get_and_print_results(ood_loader, ood_name='inat')
#
#
# # /////////////// Places365 ///////////////
# ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Places/",
#                             transform=trn.Compose([
#     trn.Resize(256),
#     trn.CenterCrop(224),
#     trn.ToTensor(),
#     trn.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=2, pin_memory=True)
# print('\n\nPlaces365 Detection')
# get_and_print_results(ood_loader, ood_name='place')
#
# # /////////////// sun ///////////////
# ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/SUN/",
#                             transform=trn.Compose([
#     trn.Resize(256),
#     trn.CenterCrop(224),
#     trn.ToTensor(),
#     trn.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=1, pin_memory=True)
# print('\n\nSUN Detection')
# get_and_print_results(ood_loader, ood_name='sun')
#
#
# # /////////////// texture ///////////////
# ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/ImageNet_OOD_dataset/Textures/",
#                             transform=trn.Compose([
#     trn.Resize(256),
#     trn.CenterCrop(224),
#     trn.ToTensor(),
#     trn.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=2, pin_memory=True)
# print('\n\ntexture Detection')
# get_and_print_results(ood_loader, ood_name='text')
#
# # /////////////// Mean Results ///////////////
#
# print('\n\nMean Test Results!!!!!')
# print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
#
#
# breakpoint()
# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
get_and_print_results(ood_loader)


# /////////////// SVHN /////////////// # cropped and no sampling of the test set
ood_data = svhn.SVHN(root='/nobackup-slow/dataset/svhn/', split="test",
                     transform=trn.Compose(
                         [  # trn.Resize(32),
                             trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/places365/",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/LSUN_C",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/LSUN_resize",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_Resize Detection')
get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////
# ood_data = dset.ImageFolder(root="/nobackup-slow/dataset/iSUN",
#                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=1, pin_memory=True)
# print('\n\niSUN Detection')
# get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

# /////////////// OOD Detection of Validation Distributions ///////////////

if args.validate is False:
    exit()

auroc_list, aupr_list, fpr_list = [], [], []

# /////////////// Uniform Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(
    np.random.uniform(size=(ood_num_examples * args.num_to_avg, 3, 32, 32),
                      low=-1.0, high=1.0).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform[-1,1] Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Arithmetic Mean of Images ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('../data/vision-greg/cifarpy', train=False, transform=test_transform)
else:
    ood_data = dset.CIFAR10('../data/vision-greg/cifarpy', train=False, transform=test_transform)


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data),
                                         batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nArithmetic Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Geometric Mean of Images ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('../data/vision-greg/cifarpy', train=False, transform=trn.ToTensor())
else:
    ood_data = dset.CIFAR10('../data/vision-greg/cifarpy', train=False, transform=trn.ToTensor())


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(
    GeomMeanOfPair(ood_data), batch_size=args.test_bs, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

print('\n\nGeometric Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Jigsaw Images ///////////////

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
), 1)

ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(mean, std)])

print('\n\nJigsawed Images Detection')
get_and_print_results(ood_loader)

# /////////////// Speckled Images ///////////////

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(mean, std)])

print('\n\nSpeckle Noised Images Detection')
get_and_print_results(ood_loader)

# /////////////// Pixelated Images ///////////////

pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(mean, std)])

print('\n\nPixelate Detection')
get_and_print_results(ood_loader)

# /////////////// RGB Ghosted/Shifted Images ///////////////

rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(mean, std)])

print('\n\nRGB Ghosted/Shifted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Inverted Images ///////////////

# not done on all channels to make image ood with higher probability
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(mean, std)])

print('\n\nInverted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

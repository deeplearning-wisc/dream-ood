import numpy as np
import torch
import clip
import torch
import clip
from PIL import Image
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
import torchvision.datasets as dset
import torch.nn.functional as F
from resnet import ResNet_Model
from PIL import Image as PILImage

all_classes = ['stingray', 'hen', 'magpie', 'kite', 'vulture',
               'agama',   'tick', 'quail', 'hummingbird', 'koala',
               'jellyfish', 'snail', 'crawfish', 'flamingo', 'orca',
               'chihuahua', 'coyote', 'tabby', 'leopard', 'lion',
               'tiger','ladybug', 'fly' , 'ant', 'grasshopper',
               'monarch', 'starfish', 'hare', 'hamster', 'beaver',
               'zebra', 'pig', 'ox', 'impala',  'mink',
               'otter', 'gorilla', 'panda', 'sturgeon', 'accordion',
               'carrier', 'ambulance', 'apron', 'backpack', 'balloon',
               'banjo','barn','baseball', 'basketball', 'beacon',
               'binder', 'broom', 'candle', 'castle', 'chain',
               'chest', 'church', 'cinema', 'cradle', 'dam',
               'desk', 'dome', 'drum','envelope', 'forklift',
               'fountain', 'gown', 'hammer','jean', 'jeep',
               'knot', 'laptop', 'mower', 'library','lipstick',
               'mask', 'maze', 'microphone','microwave','missile',
                'nail', 'perfume','pillow','printer','purse',
               'rifle', 'sandal', 'screw','stage','stove',
               'swing','television','tractor','tripod','umbrella',
                'violin','whistle','wreck', 'broccoli', 'strawberry'
               ]


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


normalize = trn.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
test_data = \
    torchvision.datasets.ImageFolder(
    os.path.join('/nobackup-slow/dataset/my_xfdu/IN100_new/', 'val'),
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))


num_classes = 100

test_loader = torch.utils.data.DataLoader(test_data, batch_size=200, shuffle=False,
                                          num_workers=4, pin_memory=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(test_loader):
#         features = model.encode_image(data)
#         features /= features.norm(dim=-1, keepdim=True)
#         if batch_idx == 0:
#             labels_all = target
#             features_all = features
#         else:
#             labels_all = torch.cat([labels_all, target], 0)
#             features_all = torch.cat([features_all, features], 0)
text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_classes]).to(device)
# text = clip.tokenize(all_classes).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
breakpoint()


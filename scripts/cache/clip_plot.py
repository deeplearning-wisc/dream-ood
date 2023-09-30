import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logits', default=0, type=int)
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--pro_length', default=0, type=int)
parser.add_argument('--name', default=1., type=str)
args = parser.parse_args()

sns.set(context="paper", style="white")
outlier_number = 100
num_classes_consider = 100

data_token = np.load('./text_features.npy')
data = np.load('./data_clip.npy')
number_class = []
fixed_number = 50
for i in range(num_classes_consider):
    if i == 0:
        data_preprocess = data[i*50:i*50+50]
    else:
        data_preprocess = np.concatenate((data_preprocess, data[i*50:i*50+50]), 0)
    number_class.append(fixed_number)

data_preprocess = np.array(data_preprocess).reshape(-1, 512)

targets = []


data_preprocess = torch.from_numpy(data_preprocess)
# breakpoint()

data_preprocess = torch.cat([data_preprocess,
                             torch.from_numpy(data_token)[:num_classes_consider]],0).numpy()
# # breakpoint()
# data_preprocess = torch.cat([data_preprocess,
#                              F.normalize(torch.from_numpy(np.load('./outlier_cosine_vmf_select_100_new_version.npy').reshape(-1, 768)),
#                                          p=2, dim=-1)[:num_classes_consider*outlier_number]]).numpy()

# length = []
# for index in range(5):
#     data_preprocess = torch.cat([data_preprocess,
#                                  F.normalize(torch.from_numpy(np.load(str(index)+'.npy')),p=2, dim=-1)])
#     length.append(np.load(str(index)+'.npy').shape[0])
#
# data_preprocess = data_preprocess.numpy()


print(data_preprocess.shape)
# breakpoint()
from sklearn.manifold import TSNE
embedding1 = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30).fit_transform(data_preprocess)

# reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.4, n_components=2, metric='euclidean')#30, 0.6
# embedding1 = reducer.fit_transform(data_preprocess)

embedding = embedding1[:num_classes_consider*fixed_number]
embedding_anchor = embedding1[num_classes_consider*fixed_number:num_classes_consider*fixed_number+num_classes_consider]
# embedding_outlier = embedding1[num_classes_consider*fixed_number+num_classes_consider:num_classes_consider*fixed_number+num_classes_consider+num_classes_consider*outlier_number]
# embedding_df_outlier = embedding1[num_classes_consider*fixed_number+num_classes_consider+num_classes_consider*outlier_number:]
# breakpoint()
fig, ax = plt.subplots(figsize=(12, 12))
def get_cmap(n, name='Pastel1'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

color = get_cmap(num_classes_consider)

# color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# breakpoint()
sum = 0
sum_outlier = 0
sum_anchor = 0
sum_df = 0
# breakpoint()
for i in range(0, num_classes_consider):
    # breakpoint()
    # color1 = color[i]
    # print(color1)
    plt.scatter(embedding[:, 0][sum: sum + number_class[i]],
                embedding[:, 1][sum: sum + number_class[i]],
                cmap=color, s=5)
    # plt.scatter(embedding_outlier[:, 0][sum_outlier: sum_outlier + outlier_number],
    #             embedding_outlier[:, 1][sum_outlier: sum_outlier + outlier_number],
    #             c='b', s=60, marker='*')
    #
    if i == 99:
        plt.scatter(embedding_anchor[:, 0][sum_anchor],
                    embedding_anchor[:, 1][sum_anchor],
                    c='k', s=30, label='Token Embed.')
    else:
        plt.scatter(embedding_anchor[:, 0][sum_anchor],
                    embedding_anchor[:, 1][sum_anchor],
                    c='k', s=30)

    sum += number_class[i]
    sum_outlier += outlier_number
    sum_anchor += 1
    # sum_df += length[i]


plt.legend(fontsize=20)
# ax.legend(loc='lower left',markerscale=9)
plt.setp(ax, xticks=[], yticks=[])
plt.savefig('./feat_dis_clip.jpg', dpi=250)
# plt.show()
import numpy as np
import torch
import faiss
import umap
import time
#import matplotlib.pyplot as plt
import faiss.contrib.torch_utils

import torch.nn.functional as F

def KNN_dis_search_decrease(target, index, K=50, select=1,shift=0):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
    normed_target = target / target_norm
    #start_time = time.time()

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    #k_th_output_index = output_index[:, -1]
    if shift:
        k_th_distance, minD_idx = torch.topk(-k_th_distance, select)
    else:
        k_th_distance, minD_idx = torch.topk(k_th_distance, select)
    #k_th_index = k_th_output_index[minD_idx]
    return minD_idx, k_th_distance

def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000,depth=342,shift=0):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)
    normed_target = target / target_norm
    #start_time = time.time()

    distance, output_index = index.search(normed_target, K)
    k_th_distance = distance[:, -1]
    k_th = k_th_distance.view(length, -1)
    target_new = target.view(length, -1, depth)
    #k_th_output_index = output_index[:, -1]
    if shift:
        k_th_distance, minD_idx = torch.topk(-k_th, num_points, dim=0)
    else:
        k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)
    minD_idx = minD_idx.squeeze()
    point_list = []
    # breakpoint()
    if len(minD_idx.size()) == 1:
        minD_idx = minD_idx.reshape(-1,1)
    for i in range(minD_idx.shape[1]):
        point_list.append(i*length + minD_idx[:,i])
    #return tor+ch.cat(point_list, dim=0)

    return target[torch.cat(point_list)]

def generate_outliers(ID, input_index, negative_samples, ID_points_num=2,
                      K=20, select=1, cov_mat=0.1, sampling_ratio=1.0,
                      pic_nums=30, depth=342, shift=0):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[0], int(normed_data.shape[0] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select, shift=shift)
    boundary_data = ID[minD_idx]
    # breakpoint()
    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]
    data_point_list = torch.cat([ID[i:i+1].repeat(length,1) for i in minD_idx])
    negative_sample_cov = cov_mat * negative_samples.cuda().repeat(pic_nums,1)
    negative_sample_list = F.normalize(negative_sample_cov + data_point_list, p=2, dim=1)
    # breakpoint()
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length,depth, shift=shift)

    index.reset()

    #return ID[minD_idx]
    return point, boundary_data

def generate_outliers_OOD(ID, input_index, negative_samples, K=100, select=100, sampling_ratio=1.0):
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(negative_samples, index, K, select)

    return negative_samples[minD_idx]



def generate_outliers_rand(ID, input_index,
                           negative_samples, ID_points_num=2, K=20, select=1,
                           cov_mat=0.1, sampling_ratio=1.0, pic_nums=10,
                           repeat_times=30, depth=342):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
    ID_boundary = ID[minD_idx]
    negative_sample_list = []
    for i in range(repeat_times):
        select_idx = np.random.choice(select, int(pic_nums), replace=False)
        sample_list = ID_boundary[select_idx]
        mean = sample_list.mean(0)
        var = torch.cov(sample_list.T)
        var = torch.mm(negative_samples, var)
        trans_samples = mean + var
        negative_sample_list.append(trans_samples)
    negative_sample_list = torch.cat(negative_sample_list, dim=0)
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length,depth)

    index.reset()

    #return ID[minD_idx]
    return point


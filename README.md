# DREAM-OOD

This is the source code accompanying the paper [***Dream the Impossible: Outlier Imagination with Diffusion Models***](https://arxiv.org/pdf/2309.13415) by Xuefeng Du, Yiyou Sun, Xiaojin Zhu, and Yixuan Li


The codebase is heavily based on [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

## Ads 

Checkout our similar CVPR'22 work [STUD](https://github.com/deeplearning-wisc/stud) on object detection in video datasets if you are interested!

## Requirements
A suitable [conda](https://conda.io/) environment named `dreamood` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate dreamood
```

## Dataset Preparation

**ImageNet-100**

* Download the full ImageNet dataset from the official website [here](https://www.image-net.org/).

* Preprocess the dataset to get ImageNet-100 by running:

```
python scripts/process_in100.py --outdir xxx
```
where "--outdir" specifies the address of the dataset you want to store.

**CIFAR-100**

* The dataloader will download it automatically when first running the programs.

**OOD datasets**


* The OOD datasets with ImageNet-100 as in-distribution are 4 OOD datasets from iNaturalist, SUN, Places, and Textures, which contain the de-duplicated concepts overlapped with ImageNet.
* The OOD datasets with CIFAR-100 as in-distribution are 5 OOD datasets, i.e., SVHN, PLACES365, LSUN, ISUN, TEXTURES.
* Please refer to Part 1 and 2 of the codebase [here](https://github.com/deeplearning-wisc/knn-ood). 




## Training (Generate outliers)


**Learning the text-conditioned latent space**

Please execute the following in the command shell on different datasets:
```
python scripts/pretrain.py --dataset in100 --t 0.1
```

**Synthesizing outliers in the pixel space**

Please execute the following in the command shell on different datasets:

```
#python scripts/dream_ood_in100.py --plms --n_iter 50 --n_samples 3 --outdir /nobackup-fast/txt2img-samples-in100-demo/ --loaded_embedding /nobackup-slow/dataset/my_xfdu/diffusion/outlier_npos_embed.npy --ckpt /nobackup-slow/dataset/my_xfdu/diffusion/sd-v1-4.ckpt --skip_grid
```


## Test-time OOD detection
Please execute the following in the command shell with ImageNet-100 as in-distribution:
```
python scripts/evaluate_ood_in100.py
```
Please execute the following in the command shell with Cifar-100 as in-distribution:
```
python scripts/evaluate_ood_cifar100.py
```

**Visualization of the generated outliers**

## Training (Generate inliers)


**Learning the text-conditioned latent space**

Please execute the following in the command shell on different datasets:
```
python scripts/pretrain.py --dataset in100 --t 0.1
```

**Synthesizing inliers in the pixel space**

Please execute the following in the command shell on different datasets:

```
#python scripts/dream_id_in100.py --plms --n_iter 50 --n_samples 3 --outdir /nobackup-fast/txt2img-samples-in100-demo/ --loaded_embedding /nobackup-slow/dataset/my_xfdu/diffusion/outlier_npos_embed.npy --ckpt /nobackup-slow/dataset/my_xfdu/diffusion/sd-v1-4.ckpt --skip_grid
```

**Visualization of the generated inliers**

**Pretrained models and embeddings**

The pretrained models/embeddings for ImageNet-100 can be downloaded from [pretrained feature extractor](), [inlier embeddings](), [outlier embeddings]().

The pretrained models/embeddings for Cifar-100 can be downloaded from [pretrained feature extractor](), [inlier embeddings](), [outlier embeddings]().




## Citation ##
If you found any part of this code is useful in your research, please consider citing our paper:

```
  @inproceedings{du2023dream,
      title={Dream the Impossible: Outlier Imagination with Diffusion Models}, 
      author={Xuefeng Du and Yiyou Sun and Xiaojin Zhu and Yixuan Li },
      booktitle={Advances in Neural Information Processing Systems},
      year = {2023}
}
```






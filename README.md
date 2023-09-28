# DREAM-OOD

This is the source code accompanying the paper [***Dream the Impossible: Outlier Imagination with Diffusion Models***](https://arxiv.org/pdf/2309.13415) by Xuefeng Du, Yiyou Sun, Xiaojin Zhu, and Yixuan Li


The codebase is heavily based on [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

## Ads 

Check out our latent-based outlier synthesis papers in ICLR'22 [VOS](https://github.com/deeplearning-wisc/stud) and ICLR'23 [NPOS](https://github.com/deeplearning-wisc/npos) if you are interested!


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

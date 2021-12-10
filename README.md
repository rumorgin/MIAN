# Multi-Instance Attention Network for Few-shot Learning

The code repository for [Multi-Instance Attention Network for Few-shot Learning]

## Abstract

The attention mechanism is usually equipped with a few-shot learning framework, and plays a key role in extracting the 
semantic object(s). However, most attention networks in existing few-shot learning algorithms often work on the channel 
and/or pixel dimension, and require learning a large amount of parameters. Due to the limitation of training examples, 
these attention networks tend to be under-fitting, and may fail to find the semantic target(s). In this paper, we 
transform the few-shot learning problem into a multi-instance learning problem, and design a new attention network 
that works on image patches, which significantly decreases the number of parameters in the attention network meanwhile
classifying semantic patches and irrelevant patches directly. In each episode, we first convert the original image into 
a multi-instance bag by splitting the whole image into several blocks. Afterward, an attention-based multi-instance 
model is proposed to learn the representative prototypes of each bag. Finally, we further introduce a metric-based 
meta-classifier for prediction. Extensive experiments on typical real-world data sets have demonstrated that our proposed
algorithm achieves consistent improvement over many state-of-the-art models.

## Dataset
Please click the Google Drive [link](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing) for downloading the 
following datasets, or running the downloading bash scripts in folder `datasets/` to download.

### MiniImageNet Dataset

It contains 100 classes with 600 images in each class, which are built upon the ImageNet dataset. The 100 classes are divided into 64, 16, 20 for meta-training, meta-validation and meta-testing, respectively.

### TieredImageNet Dataset
TieredImageNet is also a subset of ImageNet, which includes 608 classes from 34 super-classes. Compared with  miniImageNet, the splits of meta-training(20), meta-validation(6) and meta-testing(8) are set according to the super-classes to enlarge the domain difference between  training and testing phase. The dataset also include more images for training and evaluation (779,165 images in total).

### CUB Dataset
CUB was originally proposed for fine-grained bird classification, which contains 11,788 images from 200 classes. We follow the splits in [FEAT](https://github.com/Sha-Lab/FEAT) that 200 classes are divided into 100, 50 and 50 for meta-training, meta-validation and meta-testing, respectively.

### FC100 Dataset
FC100 is a few-shot classification dataset built on CIFAR100. We follow the split division proposed in [TADAM](https://papers.nips.cc/paper/7352-tadam-task-dependent-adaptive-metric-for-improved-few-shot-learning.pdf), where 36 super-classes were divided into 12 (including 60 classes), 4 (including 20 classes), 4 (including 20 classes), for meta-training, meta-validation and meta-testing, respectively, and each class contains 100 images.

## Training and Testing tips

First run (`pretrain.py`) to get the pretrain model parameters. Then adjust the path of pretrain parameters in (`train.py`), run (`train.py`) to start training and testing processes. 

## Acknowledgment
Our project references the codes in the following repos.
- [DeepEMD](https://github.com/icoz69/DeepEMD)





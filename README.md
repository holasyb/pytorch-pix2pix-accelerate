# pytorch-pix2pix-accelerate

# Pytorch pix2pix with accelerate, supporting ddp and fp16  

**Note**: This code is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
Any more information about pytorch-CycleGAN-and-pix2pix can be found in the original repository.
This repository add the following features:
- Distributed Data Parallel (DDP) training
- Mixed precision training with fp16

## Prerequisites
- Python 3
- pytorch
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/holasyb/pytorch-pix2pix-accelerate
cd pytorch-pix2pix-accelerate
```

**Please prepare your own environment**

### train
#### pix2pix
- Download a pix2pix dataset (e.g.[maps](http://cmp.felk.cvut.cz/~tylecr1/maps/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh maps
```

## [Datasets](docs/datasets.md)
Download pix2pix datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.



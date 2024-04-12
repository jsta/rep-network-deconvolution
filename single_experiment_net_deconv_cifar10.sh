#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres gpu:1

enable_lmod
module load container_env pytorch-gpu/1.13.0

export CUDA_HOME=/cm/shared/applications/cuda-toolkit/11.7.1/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME


crun.pytorch-gpu -p ~/envs/cs895_project python main.py --lr .1 --optimizer SGD --arch $architecture --epochs $epochs --dataset cifar10  --batch-size 128 --msg True --deconv True --block-fc 512 --wd .001
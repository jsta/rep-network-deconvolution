#!/bin/bash

#SBATCH -c 32
#SBATCH -p high-gpu-mem
#SBATCH --gres gpu:1

enable_lmod
module load container_env pytorch-gpu/1.13.0

export CUDA_HOME=/cm/shared/applications/cuda-toolkit/11.7.1/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

crun.pytorch-gpu -p ~/envs/cs834_project python main_imagenet.py -a densenet121d -j 32 imagenet/ILSVRC/Data/CLS-LOC --deconv True
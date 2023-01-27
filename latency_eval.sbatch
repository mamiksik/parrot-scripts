#!/bin/bash
# General Requirements
#SBATCH --constraint="[a40]&[avx2|avx]"
##SBATCH --partition dmb

# CPU Requirements
#SBATCH -c 1
#SBATCH --mem=4gb

# GPU Requirements
##SBATCH --gres=gpu:ampere:2

# check if gpu are assigned, if not create empty list
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi

# load python 3.10
module load python


echo "Date                    = $(date)"
echo "Hostname                = $(hostname -s)" # log hostname
echo "Cuda Visible Devices    = '""$CUDA_VISIBLE_DEVICES""'"
echo "Working Directory       = $(pwd)"
module list

echo "T5 Cuda Latency: "
python3 src/eval/evaluate_latency_t5.py --device cuda

echo "MLM Cuda Latency: "
python3 src/eval/evaluate_latency_mlm.py --device cuda

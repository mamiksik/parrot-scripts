#!/bin/bash
# General Requirements
#SBATCH --constraint="[a40|a100]&[avx2|avx]"

# CPU Requirements
#SBATCH -c 2

# GPU Requirements
#SBATCH --gres=gpu:1

# Notification Settings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.miksik@student.utwente.nl

# check if gpu are assigned, if not create empty list
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi


# load python 3.10
module load python

echo "Date                    = $(date)"
echo "Hostname                = $(hostname -s)" # log hostname
echo "Cuda Visible Devices    = '"$CUDA_VISIBLE_DEVICES"'"
echo "Working Directory       = $(pwd)"
module list

echo "Starting: "
python3 "$@"

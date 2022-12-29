#!/bin/bash
# General Requirments
#####SBATCH --partition=ram
#SBATCH --constraint="[a40|a100]&[avx2|avx]"

# CPU Requirments
#SBATCH -c 16
#SBATCH --mem=64gb

# GPUPU Requirments
#SBATCH --gres=gpu:ampere:1
##SBATCH --sockets-per-node=1

# Notification Settings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.miksik@student.utwente.nl

# check if gpu are assigned, if not create empty list
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi


# load python 3.10
module load python

# load nvidia cuda toolkit(s)
#module load nvidia/cuda-11.8
#module load nvidia/cuda-11.3
#module load nvidia/cuda-11.3_cudnn-8.2
#module load nvidia/cuda-11.3_tensorrt-8.0

cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}

# Copy input and executable to the node
cp -r ${SLURM_SUBMIT_DIR}/* .

echo "Date                    = $(date)"
echo "Hostname                = $(hostname -s)" # log hostname
echo "Cuda Visible Devices    = '"$CUDA_VISIBLE_DEVICES"'"
echo "Working Directory       = $(pwd)"
module list

echo "Starting Model: "
python3 "$@"

cp -r output-model/* ${SLURM_SUBMIT_DIR}/output-model/

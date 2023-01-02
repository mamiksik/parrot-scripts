#!/bin/bash
# General Requirements
#SBATCH --constraint="[a40|a100]&[avx2|avx]"

# CPU Requirements
#SBATCH -c 16
#SBATCH --mem=64gb

# GPU Requirements
#SBATCH --gres=gpu:ampere:2
#SBATCH --sockets-per-node=1

# Notification Settings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.miksik@student.utwente.nl

# check if gpu are assigned, if not create empty list
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi

# load python 3.10
module load python

#cd /local
#mkdir ${SLURM_JOBID}
#cd ${SLURM_JOBID}

# Copy input and executable to the node
#cp -r ${SLURM_SUBMIT_DIR}/* .

echo "Date                    = $(date)"
echo "Hostname                = $(hostname -s)" # log hostname
echo "Cuda Visible Devices    = '"$CUDA_VISIBLE_DEVICES"'"
echo "Working Directory       = $(pwd)"
module list

echo "Starting Model: "
python3 "$@"

#cp -r output-model/* ${SLURM_SUBMIT_DIR}/output-model/

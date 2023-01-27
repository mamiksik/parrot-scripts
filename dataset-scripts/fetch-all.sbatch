#!/bin/bash
# General Requirements

# CPU Requirements
#SBATCH -c 7

# load python 3.10
module load python

echo "Date                    = $(date)"
echo "Hostname                = $(hostname -s)" # log hostname
echo "Cuda Visible Devices    = '"$CUDA_VISIBLE_DEVICES"'"
echo "Working Directory       = $(pwd)"
module list

echo "Starting: "
python3 pydrill.py --source top_repos/go.csv &
python3 pydrill.py --source top_repos/java.csv &
python3 pydrill.py --source top_repos/javascript.csv &
python3 pydrill.py --source top_repos/php.csv &
python3 pydrill.py --source top_repos/python.csv &
python3 pydrill.py --source top_repos/ruby.csv &

wait

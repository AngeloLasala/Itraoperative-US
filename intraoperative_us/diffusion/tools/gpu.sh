#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=gpu.err       # standard error file
#SBATCH --output=gpu.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python check_gpu.py
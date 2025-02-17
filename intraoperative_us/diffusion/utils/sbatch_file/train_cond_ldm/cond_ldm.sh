#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=simple_heat_cond_ldm.err       # standard error file
#SBATCH --output=simple_heat_cond_ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m intraoperative_us.diffusion.tools.train_cond_ldm --data conf\
          --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/ius_diffusion'\
          --trial trial_2
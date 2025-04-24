#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=9:50:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=vae_random_3.err       # standard error file
#SBATCH --output=vae_random_3.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m intraoperative_us.diffusion.tools.train_vae --conf cond_ldm/conf_vae_random_3\
                                                      --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                                                      --trial_name VAE_random/split_3\
                                                      --log 'info'
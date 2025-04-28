#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=9:50:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=vae_one_step.err       # standard error file
#SBATCH --output=vae_one_step.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

for split in 2 3 4 ; do
    python -m intraoperative_us.diffusion.tools.train_vae_one_step --conf one_step/conf_one_step_$split\
                                                        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                                                        --trial trial_1/split_$split\
                                                        --log 'info'
done
#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=23:50:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=vae_stack.err       # standard error file
#SBATCH --output=vae_stack.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

for split in 1 0 2 ; do
    python -m intraoperative_us.diffusion.tools.train_vae_one_step --conf one_step/conf_one_step_$split\
                                                        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                                                        --trial Stack_finetuning/split_$split\
                                                        --log 'info'
done
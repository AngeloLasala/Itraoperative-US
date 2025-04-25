#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=cond_random.err       # standard error file
#SBATCH --output=cond_random.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL    # account name

python -m intraoperative_us.diffusion.tools.train_cond_ldm_hugginface --conf cond_ldm/conf_cond_ldm_1\
          --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
          --trial VAE_finetuning\
          --experiment_name cond_ldm_random\
          --log info
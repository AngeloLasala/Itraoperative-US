#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sd_cond_3.err       # standard error file
#SBATCH --output=sd_cond_3.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m intraoperative_us.diffusion.tools.train_stable_diffusion --conf conf_stable_diffusion_3\
          --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/ius_diffusion'\
          --trial trial_SD_finetuning\
          --type_image ius\
          --experiment SD_init_random_cond_empty_text\
          --log warning\
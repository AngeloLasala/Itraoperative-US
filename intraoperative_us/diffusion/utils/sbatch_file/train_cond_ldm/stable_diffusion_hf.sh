#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sd_uncond_2.err       # standard error file
#SBATCH --output=sd_uncond_2.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m intraoperative_us.diffusion.tools.train_stable_diffusion --conf conf_stable_diffusion\
          --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/ius_diffusion'\
          --trial trial_SD_random\
          --type_image ius\
          --experiment SD_init_sd1.5_cond_empty_text\
          --log warning\
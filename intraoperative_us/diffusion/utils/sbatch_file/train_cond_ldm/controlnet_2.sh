#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=controlnet_2.err       # standard error file
#SBATCH --output=controlnet_2.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m intraoperative_us.diffusion.tools.train_controlnet --conf conf_controlnet_2\
          --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
          --trial trial_SD_finetuning\
          --type_image ius\
          --experiment Controlnet_sd1.5_empty_text\
          --log log\
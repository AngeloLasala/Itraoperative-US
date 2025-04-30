#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sd_random.err       # standard error file
#SBATCH --output=sd_random.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m intraoperative_us.diffusion.tools.train_one_step --conf one_step/conf_one_step_1\
          --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
          --trial trial_1\
          --type_image one_step\
          --experiment ldm_finetuning\
          --log info\
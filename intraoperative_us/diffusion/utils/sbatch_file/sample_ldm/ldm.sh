#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=ldm.err       # standard error file
#SBATCH --output=ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for trial in small_vae; do
    for epoch in 3000 ; do
            python -m intraoperative_us.diffusion.tools.sample_ldm\
                    --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/ius_diffusion'\
                    --trial $trial\
                    --experiment uncond_ldm_1\
                    --type_image mask\
                    --epoch $epoch\

        done
done
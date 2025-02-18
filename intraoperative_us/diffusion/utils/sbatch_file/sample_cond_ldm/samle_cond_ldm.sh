#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sample_cond_ldm.err       # standard error file
#SBATCH --output=sample_cond_ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for trial in trial_3; do
    for w in 0.0 0.5 1.0 2.0 3.0 4.0; do
        for epoch in 80 100 120 140 150; do
                python -m intraoperative_usy.diffusion.tools.sample_cond_ldm\
                        --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/ius_diffusion'\
                        --trial $trial\
                        --experiment cond_ldm_1\
                        --epoch $epoch\
                        --guide_w $w\

         done
    done
done
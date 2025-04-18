#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=small_fine.err       # standard error file
#SBATCH --output=small_fine.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

for trial in trial_ldm; do
    for w in  3.0 5.0 7.0; do
        for epoch in 5000 7000 8000 9000 ; do
                python -m intraoperative_us.diffusion.tools.sample_cond_ldm_hugginface\
                        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/small_vae/uncond_ldm_1/w_-1.0/ddpm/samples_ep_3000"\
                        --trial $trial\
                        --experiment big_finetuning\
                        --epoch $epoch\
                        --guide_w $w\

         done
    done
done
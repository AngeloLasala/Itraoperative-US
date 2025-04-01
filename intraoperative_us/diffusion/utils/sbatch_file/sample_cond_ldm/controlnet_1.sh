#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=controlnet_1.err       # standard error file
#SBATCH --output=controlnet_1.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name 

for trial in trial_SD_finetuning; do
    for w in 7.0; do
        for epoch in 1200 1800 2400 3000 ; do
                python -m intraoperative_us.diffusion.tools.sample_controlnet\
                        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/small_vae/uncond_ldm_1/w_-1.0/ddpm/samples_ep_3000"\
                        --trial $trial\
                        --experiment Controlnet_sd1.5_empty_text\
                        --epoch $epoch\
                        --guide_w $w\
                        --log info\

         done
    done
done
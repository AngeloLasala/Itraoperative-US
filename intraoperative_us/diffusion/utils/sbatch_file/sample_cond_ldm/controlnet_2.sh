#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=controlnet_2.err       # standard error file
#SBATCH --output=controlnet_2.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name 

for trial in VAE_finetuning VAE_random; do
    for w in 3.0; do
        for epoch in 5000 6000 7000 8000 9000 ; do
                python -m intraoperative_us.diffusion.tools.sample_controlnet\
                        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_1/uncond_ldm/w_-1.0/ddpm/samples_ep_3000"\
                        --trial $trial\
                        --split 'split_1'\
                        --experiment Controlnet_finetuning_empty_text\
                        --epoch $epoch\
                        --guide_w $w\
                        --scheduler 'ddpm'\
                        --num_sample_timesteps 1000\

         done
    done
done
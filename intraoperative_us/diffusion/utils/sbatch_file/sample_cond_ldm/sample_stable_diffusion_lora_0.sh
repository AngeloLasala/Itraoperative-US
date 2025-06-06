#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=3:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=split_0.err       # standard error file
#SBATCH --output=split_0.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name 

for trial in VAE_finetuning; do
    for w in -1.0; do
        for epoch in 9000 ; do
                python -m intraoperative_us.diffusion.tools.sample_stable_diffusion\
                        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
                        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_0/uncond_ldm/w_-1.0/ddpm/samples_ep_3000"\
                        --trial $trial\
                        --split 'split_0'\
                        --experiment SD_lora_empty_text\
                        --epoch $epoch\
                        --guide_w $w\
                        --scheduler 'ddpm'\
                        --num_sample_timesteps 1000\

         done
    done
done
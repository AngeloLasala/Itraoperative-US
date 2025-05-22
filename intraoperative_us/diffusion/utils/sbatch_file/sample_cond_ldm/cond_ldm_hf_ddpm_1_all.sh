#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sampling_all.err       # standard error file
#SBATCH --output=sampling_all.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python -m intraoperative_us.diffusion.tools.sample_cond_ldm_hugginface\
        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_0/uncond_ldm/w_-1.1/ddpm/samples_ep_3000"\
        --trial VAE_finetuning\
        --split 'split_0'\
        --experiment cond_ldm_finetuning\
        --epoch 5000\
        --guide_w 3.0\
        --scheduler 'ddpm'\
        --num_sample_timesteps 1000\

python -m intraoperative_us.diffusion.tools.sample_cond_ldm_hugginface\
        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_1/uncond_ldm/w_-1.1/ddpm/samples_ep_3000"\
        --trial VAE_finetuning\
        --split 'split_1'\
        --experiment cond_ldm_finetuning\
        --epoch 6000\
        --guide_w 3.0\
        --scheduler 'ddpm'\
        --num_sample_timesteps 1000\

python -m intraoperative_us.diffusion.tools.sample_cond_ldm_hugginface\
        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_2/uncond_ldm/w_-1.1/ddpm/samples_ep_3000"\
        --trial VAE_finetuning\
        --split 'split_2'\
        --experiment cond_ldm_finetuning\
        --epoch 5000\
        --guide_w 3.0\
        --scheduler 'ddpm'\
        --num_sample_timesteps 1000\

python -m intraoperative_us.diffusion.tools.sample_cond_ldm_hugginface\
        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_3/uncond_ldm/w_-1.1/ddpm/samples_ep_1000"\
        --trial VAE_finetuning\
        --split 'split_3'\
        --experiment cond_ldm_finetuning\
        --epoch 5000\
        --guide_w 3.0\
        --scheduler 'ddpm'\
        --num_sample_timesteps 1000\        

python -m intraoperative_us.diffusion.tools.sample_cond_ldm_hugginface\
        --save_folder '/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion'\
        --generated_mask_dir "/leonardo_work/IscrC_AIM-ORAL/Angelo/trained_model/ius_diffusion/mask/split_4/uncond_ldm/w_-1.1/ddpm/samples_ep_1500"\
        --trial VAE_finetuning\
        --split 'split_4'\
        --experiment cond_ldm_finetuning\
        --epoch 5000\
        --guide_w 3.0\
        --scheduler 'ddpm'\
        --num_sample_timesteps 1000\

         
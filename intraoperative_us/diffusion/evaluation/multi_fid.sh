#!/bin/bash
for trial in VAE_random ; do
    for split in split_1; do
        for exp in Controlnet_lora_empty_text ; do
            for w in 3.0; do
                for scheduler in ddpm ddim dpm_solver ; do
                    python fid.py --trial $trial\
                                  --split $split\
                                  --experiment $exp\
                                  --guide_w $w\
                                  --scheduler $scheduler\
                                  --log info
                done
            done
        done
    done
done

    
                
        
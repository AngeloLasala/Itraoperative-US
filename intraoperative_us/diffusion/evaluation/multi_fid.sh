#!/bin/bash
for trial in VAE_finetuning ; do
    for split in split_4; do
        for exp in cond_ldm_finetuning ; do
            for w in 3.0; do
                for scheduler in ddpm ; do
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

    
                
        
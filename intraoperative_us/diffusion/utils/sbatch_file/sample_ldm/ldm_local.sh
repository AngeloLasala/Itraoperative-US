for trial in split_4; do
    for epoch in 500 1000 1500 2000 2500 3000 ; do
            python -m intraoperative_us.diffusion.tools.sample_ldm\
                    --save_folder "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model"\
                    --trial $trial\
                    --experiment uncond_ldm\
                    --type_image mask\
                    --epoch $epoch\

        done
done
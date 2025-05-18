for split in 4; do
    for epoch in 100 200 300 400 500 600 ; do
            python -m intraoperative_us.gan.tools.test\
                    --dataset_path "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/mask/split_$split/uncond_ldm/w_-1.0/ddpm/samples_ep_1500"\
                    --checkpoints_dir "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/gan"\
                    --epoch $epoch\
                    --name split_$split   
        done
done
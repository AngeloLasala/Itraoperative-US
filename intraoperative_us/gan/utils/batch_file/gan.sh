for split in 0 1 2; do
    for epoch in 600 ; do
            python -m intraoperative_us.gan.tools.train\
                    --dataset_path "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/dataset"\
                    --checkpoints_dir "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model/gan/"\
                    --n_epochs $epoch\
                    --name split_$split\
                    --n_epochs_decay 0\
                    --splitting_json splitting_$split.json\
                    --log info\
                     
        done
done
#!/bin/bash
for trial in bce_loss ; do
    for split in split_0 split_2 split_3 split_4; do
        for exp in real_and_gen ; do
            for data in val; do
                    python test_seg.py --trial $trial --split $split --experiment $exp --dataset_split $data
            done
        done
    done
done

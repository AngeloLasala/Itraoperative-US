"""
Robust data splitting for train and validation of diffussio models.

Test: 3 subjects = [24, 25, 27]
Train + val: 20 subject

The idea is to have a 18:2 splitting of developend set that i will use in this way
1) Single split 18:2: evaluating the best stategy within each model (initialization and fine-tuning strategy)
2) 5-fold cross-validation: evaluating the best model (Pix2Pix, condLDM, StableDiffusion, ControlNet, One-step diffusion)
"""
import os
import random
import numpy as np
import argparse
import re
import json

def get_number_case(str):
    """
    Get the number of case from the string
    """
    match = re.search(r'\d+', str)
    if match:
        return int(match.group())
    else:
        return None
   

def main(args):
    """
    Robust split
    """
    dataset_path = args.dataset
    subject_list = [i for i in os.listdir(dataset_path)]

    test_list = ["Case24-US-before", "Case25-US-before", "Case27-US-before"]
    development_list = [i for i in subject_list if i not in test_list]

    # Shuffle and select 10 unique validation items
    np.random.seed(42)
    random.seed(42)
    shuffled = development_list.copy()
    random.shuffle(shuffled)
    all_val_items = shuffled[:10]

    # Remaining items to be used in training
    remaining_items = shuffled[10:]

    splits = []
    for i in range(5):
        splitting_dict = {"train": None, "val": None, "test": None} 
        val = all_val_items[i*2:i*2+2]
        train = [item for item in development_list if item not in val]

        splitting_dict["train"] = train
        splitting_dict["val"] = val
        splitting_dict["test"] = test_list
        print(f"Split {i}:")
        print("Train:", len(train), ":", "Val:", len(val), ":", "Test:", len(test_list))
        print("Train:", [get_number_case(i) for i in train])
        print("Val:", [get_number_case(i) for i in val])
        print("Test:", [get_number_case(i) for i in test_list])

        
        # save the dict with json
        # get the parent directory of the dataset
        save_dir = os.path.dirname(dataset_path)
        with open(os.path.join(save_dir, f"splitting_{i}.json"), 'w') as f:
            json.dump(splitting_dict, f, indent=4)
    
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust data splitting for diffusion models")
    parser.add_argument('--dataset', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset/dataset", help='Dataset')
    args = parser.parse_args()

    main(args)


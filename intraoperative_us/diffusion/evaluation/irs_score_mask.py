"""
Compute the image retrieval score (IRS) for a set of images based on their masks.

Adapted from original code: 
https://github.com/MischaD/BeyondFID/blob/main/beyondfid/metrics/irs.py
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.optimize import fsolve
import numpy as np
from functools import lru_cache
import numpy as np
from tqdm import tqdm
import torch
import os
import argparse
import logging
import yaml
from torch.utils.data import DataLoader
from intraoperative_us.diffusion.evaluation.investigate_vae import get_config_value
from intraoperative_us.diffusion.dataset.dataset import IntraoperativeUS_mask, GeneratedMaskDataset


class LogFactorial:
    def __init__(self, max_value):
        """Initialize the LogFactorial class with a precomputed table up to max_value."""
        self.max_value = max_value
        self.logfactorial_lookup = self.compute_logfactorial_lookup(max_value)

    def compute_logfactorial_lookup(self, max_value):
        """Compute the logfactorial values up to max_value and store them in a list."""
        logfactorial_lookup = [0] * (max_value + 1)  # Initialize the list with zeros
        for i in range(2, max_value + 1):
            logfactorial_lookup[i] = logfactorial_lookup[i - 1] + np.log(i)
        return logfactorial_lookup

    def extend_lookup(self, new_max_value):
        """Extend the lookup table if the requested value exceeds the current max_value."""
        # Only extend if new_max_value is larger than current max_value
        if new_max_value > self.max_value:
            logger.info(f"Extending LogFactorial lookup table to {new_max_value}")
            # Extend the current table to new_max_value
            for i in range(self.max_value + 1, new_max_value + 1):
                self.logfactorial_lookup.append(self.logfactorial_lookup[i - 1] + np.log(i))
            self.max_value = new_max_value

    def __call__(self, x):
        """Make the class callable to return log(x!)."""
        if x > self.max_value:
            # If x exceeds the current max_value, extend the lookup table
            self.extend_lookup(x)
        return self.logfactorial_lookup[x]


def log_binom(n, k): 
    return logfactorial(n) - logfactorial(k) - logfactorial(n-k)


# Stirling number approximation function
def log_stirling_second_kind_approx(n, k):
    # Compute v and solve for G
    if n == k: 
        return 0 
    assert k > 0  and k < n 

    v = n / k

    # Define the function for G
    def G_func(G):
        return G - v * np.exp(G - v)

    # Solve for G
    G_initial_guess = 0.5  # Starting guess for G
    G = fsolve(G_func, G_initial_guess)[0]

    #print(f"n {n} -- k {k} -- v {v} -- G {G}")

    # Compute the other parts of the approximation formula
    part1 = 0.5 * np.log((v - 1) / (v * (1 - G)))
    part2 = (n-k) * np.log(((v - 1) / (v - G)))
    part3 = n*np.log(k) -  k * np.log(n) + k * (1 - G)

    # Combine parts with binomial coefficient
    approximation = part1 + part2 + part3 + log_binom(n, k)

    return approximation


def log_compute_formula(s, k, n):
    """
    Compute the log formu
    """
    logstir = log_stirling_second_kind_approx(n, k)
    return logstir + logfactorial(s) - logfactorial(s-k) - n * np.log(s)


logfactorial = LogFactorial(int(5e4)) # will be recomputed

class IRSMetric(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha_e = config["alpha_e"]
        self.confidence = True
        self.prob_tolerance = config["prob_tolerance"]
        self.naive = config["naive"]
        self.batch_size = config["batch_size"] # for computation of closes neighbour
        self.verbose = config["verbose"]


    def compute_irs_inf(self, n_train_max, n_sampled, k_measured): 
        # n_train_max = len(train_dataset)
        # n_sampled = python sample.py --N=n_sampled --outpath=samples
        # k_measured = beyondfid irs --trainpath=train_dataset --synthpath=samples

        n_train = n_train_max
        alpha_e = self.alpha_e
        confidence= self.confidence 
        prob_tolerance= self.prob_tolerance
        naive= self.naive

        alpha_of_IRS_alpha = n_sampled / n_train
        if self.verbose:
            print(f"Maximum Possible Number of Train Images: {n_train_max}\nSampled images: {n_sampled}\nLearned images: {k_measured}")
            print(f"IRS (alpha={alpha_of_IRS_alpha:.2f}): {k_measured / n_train_max}")
        
        if naive == True: 
            probs = []
            n_train_ests = [*range(k_measured, n_train_max)]
            for n_train_est in n_train_ests: 
                alpha_of_IRS_alpha = n_sampled / n_train_est 
                irs_alpha = np.exp(log_compute_formula(s=n_train_est, k=k_measured, n=n_sampled))
                probs.append(irs_alpha)
                if len(probs) > 2 and probs[-2] > probs[-1] and irs_alpha < prob_tolerance: 
                    break
            probs = np.array(probs)

            irs_inf = np.argmax(probs)
            n_learned_pred = n_train_ests[irs_inf] # most likely 

        else: 
            # do binary search instead of naive search. Uses the fact that ther is a single mode in the function over different s (note- not a distribution)
            low = k_measured
            high = n_train_max
            while low <= high:
                mid = (low + high) // 2

                # has to be either mid-1, mid, or mid+1
                if high - low == 2: 
                    break

                prob_mid_m1 = log_compute_formula(s=mid-1, k=k_measured, n=n_sampled)
                prob_mid = log_compute_formula(s=mid, k=k_measured, n=n_sampled)
                prob_mid_p1 = log_compute_formula(s=mid+1, k=k_measured, n=n_sampled)
                #print(f"Low: {low} -- mid: {mid} -- high: {high} ")
                #print(f"m1: {prob_mid_m1} -- mid: {prob_mid} -- mid+1: {prob_mid_p1}")

                if prob_mid > max(prob_mid_m1, prob_mid_p1): 
                    break # mid is highest  
                if prob_mid >= prob_mid_p1: 
                    high = mid
                else: 
                    low = mid

            prob_mid_l1 = log_compute_formula(s=mid-1, k=k_measured, n=n_sampled)
            prob_mid = log_compute_formula(s=mid, k=k_measured, n=n_sampled) 
            prob_mid_u1 = log_compute_formula(s=mid+1, k=k_measured, n=n_sampled)
            n_learned_pred = mid - 1 + np.argmax(np.array([prob_mid_l1, prob_mid, prob_mid_u1]))


        # cannot have more than kmax different images
        kmax = min(n_train, n_sampled)

        if confidence == True: 
            # Binary search for n_learned_high
            low, high = n_learned_pred + 1, n_train_max
            while low <= high:
                mid = (low + high) // 2
                prob = 0 
                prob_k = 1  # Just a dummy value

                for k in range(min(k_measured, n_sampled), 0, -1):  # reversed
                    if prob_k > prob_tolerance:
                        prob_k = np.exp(log_compute_formula(s=mid, k=k, n=n_sampled))
                        prob += prob_k
                
                if prob < alpha_e:
                    high = mid - 1  # Search the lower half
                else:
                    low = mid + 1  # Search the upper half
            n_learned_high = high  # The largest value that satisfies the condition

            # Binary search for n_learned_low
            low, high = k_measured, n_learned_pred - 1
            while low <= high:
                mid = (low + high) // 2
                prob = 0
                prob_k = 1  # Just a dummy value

                for k in range(k_measured, kmax + 1):
                    if prob_k > prob_tolerance:
                        prob_k = np.exp(log_compute_formula(s=mid, k=k, n=n_sampled))
                        prob += prob_k

                if prob < alpha_e:
                    low = mid + 1  # Search the upper half
                else:
                    high = mid - 1  # Search the lower half

            n_learned_low = low  # The smallest value that satisfies the condition

        irs_pred = n_learned_pred / n_train_max
        if self.verbose: 
            print(f"IRS (inf): {irs_pred}")
            print(f"Predicted number of images for IRS_infinity: {n_learned_pred} -- IRS: {irs_pred}")

        if confidence == True: 
            irs_prep_higher = n_learned_high / n_train_max
            irs_prep_lower = n_learned_low / n_train_max
            if self.verbose: 
                print(f"Predicted number of images for IRS_infinity,H: {n_learned_high} -- IRS: {irs_prep_higher}")
                print(f"Predicted number of images for IRS_infinity,L: {n_learned_low} -- IRS: {irs_prep_lower}\n")
            return (irs_prep_lower, irs_pred, irs_prep_higher)

        return (None, irs_pred, None)

    
    def compute_support(self, features_a, features_b):
        features_train = features_a.to("cuda") # real / training data
        features_test = features_b # test or synthetic

        closest = []
        for i in tqdm(range(0, features_test.size(0), self.batch_size), desc="Processing Batches"):
            # 512 new 'generated' images each batch 
            batch_features = features_test[i:i+self.batch_size].to("cuda")

            dist = torch.cdist(features_train, batch_features, p=2)
            dist = dist.argmin(dim=0).cpu()
            batch_features.cpu()
            closest.extend(dist.tolist())

        features_train.cpu()
        dist.size()
        perc = len(set(closest)) / len(features_train)
        return closest, perc


    def compute_train_only(self, train): 
        results = {}
        for alpha in self.alphas: 
            percs = []
            for fold in range(self.folds):
                closest, perc = self.compute_closest_for_alpha(train, alpha, fold=fold)
                percs.append(perc)
            results[alpha] = {"mean": float(np.array(percs).mean()), "std": float(np.array(percs).std())}
        return {"diversity_train_only": results}


    def compute(self, train, test, snth): 
        # test is ignored for now 
        n_train = len(train)
        n_ref_test = len(test)
        n_ref_snth = len(snth)
        n_ref = min(n_ref_test, n_ref_snth)

        # Define a generator with a fixed seed
        generator = torch.Generator()
        generator.manual_seed(42) 

        if n_ref_snth > n_ref: 
            logger.info(f"Randomly sampling {n_ref} synthetic images to make reference estimate IRS_a more accurate")
            rnd_idx = torch.randperm(n_ref_snth, generator=generator)[:n_ref]
            snth = snth[rnd_idx]
        elif n_ref_test > n_ref: 
            logger.info(f"Randomly sampling {n_ref} test images to make reference estimate IRS_a more accurate. Consider using multiple folds")
            rnd_idx = torch.randperm(n_ref_test, generator=generator)[:n_ref]
            test = test[rnd_idx]

        results = {}
        for name, ref_data in zip(["test", "snth"], [test, snth]): 
            closest, perc = self.compute_support(train, ref_data)

            k_measured = len(set(closest))
            alpha = k_measured / n_train

            if self.verbose: 
                logger.info(f"Computing IRS results")

            irs_pred_lower, irs_pred, irs_prep_higher = self.compute_irs_inf(n_train, len(ref_data), k_measured)
            k_learned_pred = int(irs_pred * n_train)

            results[name] = {"n_train": n_train, 
                "n_sampled": len(ref_data), 
                "k_measured": k_measured,
                "alpha": alpha,
                "irs_alpha": perc, 
                "irs_inf_u": irs_prep_higher,
                "irs_inf_l": irs_pred_lower,
                "irs_inf": irs_pred,
                "k_pred_inf": k_learned_pred,
            }
        
        results["irs_adjusted"] = results["snth"]["k_pred_inf"] / results["test"]["k_pred_inf"]
        logger.info(f"IRS_adjusted_inf = {results['irs_adjusted']}")
        return results


    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        results = {}
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            
            if hashtrain == hashtest:
                logger.info("Train and test path are equal. Subsampling train dataset and using them as test dataset")
                # Both are the same, so irs_real will be 1.0
                # Manual subsampling: split train in half with random samples in train and the other half in test
                permuted_indices = torch.randperm(train.size(0))
                split_index = train.size(0) // 2
                test = train[permuted_indices[split_index:]]
                train = train[permuted_indices[:split_index]]

            if len(test) < 50_000: 
                n_subsets = int(len(snth) // len(test))
                indices = torch.randperm(snth.size(0))

                if n_subsets == 0:
                    logger.info("Fewer snth samples than test images. Manually reducing size of test dataset.")
                    test_indices = torch.randperm(test.size(0))[:snth.size(0)]
                    test = test[test_indices]
                    n_subsets = 1

                immed_results = []
                for i in range(n_subsets): 
                    metrics = self.compute(train, test, snth[indices[i * len(test):(i+1)*len(test)]])
                    immed_results.append(metrics)

                mean_results = {}
                for key in immed_results[0].keys():
                    # Ensure that nested dictionaries (e.g., results under "test" and "snth") are handled
                    if isinstance(immed_results[0][key], dict):
                        mean_results[key] = {}
                        for sub_key in immed_results[0][key].keys():
                            mean_results[key][sub_key] = sum(result[key][sub_key] for result in immed_results) / len(immed_results)
                    else:
                        mean_results[key] = sum(result[key] for result in immed_results) / len(immed_results)
            else: 
                metrics = self.compute(train, test, snth)

            # test data is unused
            results[model] = metrics
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)
        return results


def main(par_dir, conf, trial, experiment, epoch, guide_w, scheduler_type, n_points, show_gen_mask):
    """
    """
     ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    condition_config = get_config_value(autoencoder_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']


    # REAL TUMOR MASK
    data_img = IntraoperativeUS_mask(size= [dataset_config['im_size_h'], dataset_config['im_size_w']],
                            dataset_path= dataset_config['dataset_path'],
                            im_channels= dataset_config['im_channels'],
                            splitting_json=dataset_config['splitting_json'],
                            split='train',
                            splitting_seed=dataset_config['splitting_seed'],
                            train_percentage=dataset_config['train_percentage'],
                            val_percentage=dataset_config['val_percentage'],
                            test_percentage=dataset_config['test_percentage'],
                            data_augmentation=False)
    logging.info(f'len train data {len(data_img)}')
    data_loader = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)

    # GENERATIVE MASK
    generated_mask_dir = os.path.join(par_dir, trial, experiment, f'w_{guide_w}', scheduler_type, f"samples_ep_{epoch}")
    prefix_mask = 'mask' if args.type_image != 'mask' else 'x0'
    if args.type_image != 'mask':
        generated_mask_dir = os.path.join(generated_mask_dir, "masks")

    data_gen = GeneratedMaskDataset(par_dir = generated_mask_dir, size=[dataset_config['im_size_h'], dataset_config['im_size_w']], input_channels=dataset_config['im_channels'], prefix_mask=prefix_mask)
    data_loader_gen = DataLoader(data_gen, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f'len generated data {len(data_gen)}')

    # for loop to find the closest real mask 
    logging.info(f"Start comparing generated masks with real masks for compiting k_measured...")
    dsc_list, idx_list = [], []
    for idx_gen, gen_mask in enumerate(tqdm(data_loader_gen, desc="Processing Generated Masks")): 
        gen_mask = gen_mask
        
        dsc_best = 0.0
        idx_best = 0
        for idx_real, real_mask in enumerate(data_loader):
            real_mask = real_mask
            ## compute the dsc score
            dsc_score = torch.sum(torch.logical_and(gen_mask, real_mask)) / torch.sum(torch.logical_or(gen_mask, real_mask))
            
            if dsc_score > dsc_best:
                dsc_best = dsc_score
                idx_best = idx_real
        dsc_list.append(float(dsc_best))
        idx_list.append(idx_best)
    
    print(len(dsc_list), len(idx_list))
    ## printthe unique values of idx_list
    unique_idx = set(idx_list)
    print(f"Number of unique real masks: {len(unique_idx)}")

    ## measure the IRS
    conf_irs = {"alpha_e" : 0.05,
                "prob_tolerance" : 1e-6,
                "naive":False, 
                "batch_size": 512*(2**2),
                "verbose":True}
    
    irs_metric = IRSMetric(conf_irs)

    ## save the dsc_list and idx_list
    dsc_list_path = os.path.join(generated_mask_dir, 'dsc_list.npy')
    idx_list_path = os.path.join(generated_mask_dir, 'idx_list.npy')

    irs_low, irs_pred, irs_up = irs_metric.compute_irs_inf(len(data_img), len(data_gen), len(unique_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/trained_model",
                                                   help='folder to save the model')
    parser.add_argument('--type_image', type=str, default='mask', help='type of image to evaluate, ius or mask')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--scheduler_type', type=str, default='ddpm', help='scheduler for the diffusion model')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--n_points', type=int, default=100, help='number of points to sample the mask')
    parser.add_argument('--show_gen_mask', action='store_true', help="show the generative and mask images, default=False")
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    experiment_dir = os.path.join(args.save_folder, args.type_image, args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')

    main(par_dir = os.path.join(args.save_folder, args.type_image), conf=config, trial=args.trial,
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, scheduler_type=args.scheduler_type, n_points=args.n_points,
         show_gen_mask=args.show_gen_mask)



    # N_train = 350
    # N_sample = 400
    # k = 50  # Number of images that are measured

    # # Example usage
    # conf_irs = {"alpha_e" : 0.05,
    #             "prob_tolerance" : 1e-6,
    #             "naive":False, 
    #             "batch_size": 512*(2**2),
    #             "verbose":True}
    
    # irs_metric = IRSMetric(conf_irs)

    # irs_list = []   
    # for k in range(1, N_train-10,10):
    #     irs_low, irs_pred, irs_up = irs_metric.compute_irs_inf(N_train, N_sample, k)
    #     irs_list.append((irs_low, irs_pred, irs_up))

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, N_train-10,10), [x[1] for x in irs_list], marker='o', linestyle='-', color='blue')
    # plt.fill_between(range(1, N_train-10,10), 
    #                  [x[0] for x in irs_list], 
    #                  [x[2] for x in irs_list], 
    #                  color='blue', alpha=0.2, label='Confidence Interval')
    # plt.title('IRS vs k_measured')
    # plt.xlabel('k_measured (Number of Images Measured)')
    # plt.ylabel('IRS')
    # plt.grid()
    # plt.show()

    
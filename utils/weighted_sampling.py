import pandas as pd
import numpy as np
import os

import torch

def weighted_sampler_generator_v0(data_txt_dir, dataset, data_txt_dir2=None, args=None):

    # Weighted sampling
    if dataset == 'RAF-DB':
        print("Iterating label file ...")
        df = pd.read_csv(data_txt_dir, delim_whitespace=True, header=None, engine='python')
        label_frame = df[df[0].str.startswith('train')] # RAF-DB train data
        target = []
        num_class = 7
        class_sample_count = list(0. for _ in range(num_class))
        for idx in range(len(label_frame)):
            labels = label_frame.iloc[idx, -1]
            labels = np.array(labels)
            labels = labels.astype('float')
            labels = torch.from_numpy(labels).long()
            labels = torch.squeeze(labels)
            labels -= 1
            class_sample_count[labels] += 1
            target.append(labels)

        print("Iterating label file done!")
        target = np.array(target)
        target_train = target
        # # class_sample_count = [len(target_train[target_train==i]) for i in range(8)]
        # class_sample_count = [1290, 281, 717, 4772, 1982, 705,
        #                       2524]  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
        # weights = 1. / torch.Tensor(class_sample_count)
        weights = 1. / torch.Tensor(class_sample_count)
        weights = weights / max(weights)
        print(f'Class Sample Count: {class_sample_count}')
        print(f'Weight: {weights}')

        samples_weight = np.array([weights[t] for t in target_train])
        samples_weight = torch.from_numpy(samples_weight)
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                          len(samples_weight))

    elif dataset == 'FERPlus':
        print("Iterating label file ...")
        df = pd.read_csv(data_txt_dir, sep=',', header=None, engine='python')
        label_frame = df.iloc[:, 2:-2].idxmax(axis='columns').values - 2
        target = []
        num_class = 8
        class_sample_count = list(0. for _ in range(num_class))
        for idx in range(len(label_frame)):
            labels = label_frame[idx]
            labels = labels.astype('float')
            labels = torch.tensor(labels).long()
            class_sample_count[labels] += 1
            target.append(labels)

        print("Iterating label file done!")
        target = np.array(target)
        target_train = target

        weights = 1. / torch.Tensor(class_sample_count)
        weights = weights / max(weights)
        print(f'Class Sample Count: {class_sample_count}')
        print(f'Weight: {weights}')

        samples_weight = np.array([weights[t] for t in target_train])
        samples_weight = torch.from_numpy(samples_weight)
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                          len(samples_weight))

    elif dataset == 'AffectNet':
        print("Iterating label file ...")
        df = pd.read_csv(data_txt_dir, sep=',', header=0, engine='python')
        label_frame_all = df.iloc[:, 6].values
        num_class = 8
        label_frame_used_idx = (label_frame_all < num_class).nonzero()
        label_frame = label_frame_all[label_frame_used_idx]

        target = []
        class_sample_count = list(0. for _ in range(num_class))
        for idx in range(len(label_frame)):
            labels = label_frame[idx]
            labels = labels.astype('float')
            labels = torch.tensor(labels).long()
            class_sample_count[labels] += 1
            target.append(labels)

        print("Iterating label file done!")
        target = np.array(target)
        target_train = target

        weights = 1. / torch.Tensor(class_sample_count)
        weights = weights / max(weights)
        print(f'Class Sample Count: {class_sample_count}')
        print(f'Weight: {weights}')

        samples_weight = np.array([weights[t] for t in target_train])
        samples_weight = torch.from_numpy(samples_weight)
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                          len(samples_weight))

    elif dataset == 'Diamond_v0':
        print("Iterating label file ...")
        classes = [n for n in os.listdir(data_txt_dir) if 'txt' not in n]
        classes.sort()
        class_sample_count = []
        target_train = []
        for idx, cls in enumerate(classes):
            cc = len(os.listdir(os.path.join(data_txt_dir,cls)))
            class_sample_count.append(cc)
            target_temp = [idx] * cc
            target_train = target_train + target_temp


        weights = 1. / torch.Tensor(class_sample_count)
        weights = weights / max(weights)
        print(f'Class Sample Count: {class_sample_count}')
        print(f'Weight: {weights}')

        # samples_weight = np.array([weights[t] for t in target_train])
        # # samples_weight = np.array(weights)
        # samples_weight = torch.from_numpy(samples_weight) #.cuda(args.gpu, non_blocking=True)
        samples_weight = np.array([weights[t] for t in target_train])
        samples_weight = torch.from_numpy(samples_weight)
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                          len(samples_weight))

    elif dataset == 'Diamond':
        print("Iterating label file ...")
        classes = [n for n in os.listdir(data_txt_dir) if 'txt' not in n]
        classes.sort()
        class_sample_count = []
        target_train = []
        for idx, cls in enumerate(classes):
            cc = len(os.listdir(os.path.join(data_txt_dir,cls)))
            class_sample_count.append(cc)
            target_temp = [idx] * cc
            target_train = target_train + target_temp


        weights = 1. / torch.Tensor(class_sample_count)
        weights = weights / max(weights)
        print(f'Class Sample Count: {class_sample_count}')
        print(f'Weight: {weights}')

        # samples_weight = np.array([weights[t] for t in target_train])
        # # samples_weight = np.array(weights)
        # samples_weight = torch.from_numpy(samples_weight) #.cuda(args.gpu, non_blocking=True)
        samples_weight = np.array([weights[t] for t in target_train])
        samples_weight = torch.from_numpy(samples_weight)
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                          len(samples_weight))

    return weighted_sampler


def weighted_sampler_generator(dataset, args=None):

    print("Iterating label file ...")
    classes = dataset.classes
    classes.sort()
    class_sample_count = []
    target_train = []

    class_sample_count = list(torch.unique(torch.tensor(dataset.targets), return_counts=True)[1])
    for idx,cls in enumerate(classes):
        target_train = target_train + [idx]*class_sample_count[idx]

    weights = 1. / torch.Tensor(class_sample_count)
    weights = weights / max(weights)
    print(f'Class Sample Count: {class_sample_count}')
    print(f'Weight: {weights}')

    samples_weight = np.array([weights[t] for t in target_train])
    samples_weight = torch.from_numpy(samples_weight)
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                        len(samples_weight))

    return weighted_sampler
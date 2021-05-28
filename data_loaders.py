# import argparse
import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
from loss import *
import pandas as pd

from colorama import init
from colorama import Fore, Back, Style
import numpy as np

def get_train_data_loaders(args):
    if args.modality == 'Audio' or 'RGB' or args.modality == 'ALL':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = len(pd.read_pickle(args.train_source_list).index)
    num_target = len(pd.read_pickle(args.train_target_list).index)
    num_val = len(pd.read_pickle(args.val_list).index)

    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target
    num_source_val = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_val = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    source_set = TSNDataSet(args.train_source_data+".pkl", args.train_source_list, num_dataload=num_source_train, num_segments=args.num_segments,
                    new_length=data_length, modality=args.modality,
                    image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
                    random_shift=False,
                    test_mode=True,
                )

    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

    target_set = TSNDataSet(args.train_target_data+".pkl", args.train_target_list, num_dataload=num_target_train, num_segments=args.num_segments,
                    new_length=data_length, modality=args.modality,
                    image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                    random_shift=False,
                    test_mode=True,
                )

    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)

    return (source_loader, target_loader)


def get_val_data_loaders(args):
    if args.modality == 'Audio' or 'RGB' or args.modality == 'ALL':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = len(pd.read_pickle(args.train_source_list).index)
    num_target = len(pd.read_pickle(args.train_target_list).index)
    num_val = len(pd.read_pickle(args.val_list).index)

    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target
    num_source_val = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_val = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    source_set_val = TSNDataSet(args.val_source_data+".pkl", args.val_source_list, num_dataload=num_source_val, num_segments=args.val_segments,
                    new_length=data_length, modality=args.modality,
                    image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
                    random_shift=False,
                    test_mode=True,
                )

    source_sampler_val = torch.utils.data.sampler.RandomSampler(source_set_val)
    source_loader_val = torch.utils.data.DataLoader(source_set_val, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler_val, num_workers=args.workers, pin_memory=True)

    target_set_val = TSNDataSet(args.val_target_data+".pkl", args.val_target_list, num_dataload=num_target_val, num_segments=args.val_segments,
                    new_length=data_length, modality=args.modality,
                    image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                    random_shift=False,
                    test_mode=True,
                )

    target_sampler_val = torch.utils.data.sampler.RandomSampler(target_set_val)
    target_loader_val = torch.utils.data.DataLoader(target_set_val, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler_val, num_workers=args.workers, pin_memory=True)

    return (source_loader_val, target_loader_val)

def get_test_data_loaders(args):
    data_length = 1 if args.modality == "RGB" else 1
    num_test = len(pd.read_pickle(args.test_list).index)
    if args.noun_target_data is not None:
        data_set = TSNDataSet(args.test_target_data+".pkl", args.test_list, num_dataload=num_test, num_segments=args.test_segments,
            new_length=data_length, modality=args.modality,
            image_tmpl="img_{:05d}.t7" if args.modality in ['RGB', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'] else args.flow_prefix+"{}_{:05d}.t7",
            test_mode=True, noun_data_path=args.noun_target_data+".pkl"
            )
    else:
        data_set = TSNDataSet(args.test_target_data+".pkl", args.test_list, num_dataload=num_test, num_segments=args.test_segments,
            new_length=data_length, modality=args.modality,
            image_tmpl="img_{:05d}.t7" if args.modality in ['RGB', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'] else args.flow_prefix+"{}_{:05d}.t7",
            test_mode=True
            )
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.bS, shuffle=False, num_workers=args.workers, pin_memory=True)

    data_gen = tqdm(data_loader)

    output = []
    attn_values = torch.Tensor()

    return data_loader

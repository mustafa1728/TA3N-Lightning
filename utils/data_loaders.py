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
from utils.loss import *
import pandas as pd

from pathlib import Path
from colorama import init
from colorama import Fore, Back, Style
import numpy as np


def get_train_data_loaders(cfg):
    if cfg.DATASET.MODALITY == 'Audio' or 'RGB' or cfg.DATASET.MODALITY == 'ALL':
        data_length = 1
    elif cfg.DATASET.MODALITY in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = len(pd.read_pickle(cfg.PATHS.TRAIN_SOURCE_LIST).index)
    num_target = len(pd.read_pickle(cfg.PATHS.TRAIN_TARGET_LIST).index)
    num_val = len(pd.read_pickle(cfg.PATHS.VAL_LIST).index)

    num_iter_source = num_source / cfg.TRAINER.BATCH_SIZE[0]
    num_iter_target = num_target / cfg.TRAINER.BATCH_SIZE[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[
                                                                              0] == 'Y' else num_source
    num_target_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[
                                                                              1] == 'Y' else num_target
    num_source_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[0] == 'Y' else num_source
    num_target_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[1] == 'Y' else num_target

    train_source_data = Path(cfg.PATHS.PATH_DATA_SOURCE + ".pkl")
    train_source_list = Path(cfg.PATHS.TRAIN_SOURCE_LIST)
    source_set = TSNDataSet(train_source_data, train_source_list,
                            num_dataload=num_source_train,
                            num_segments=cfg.DATASET.NUM_SEGMENTS,
                            new_length=data_length, modality=cfg.DATASET.MODALITY,
                            image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                   "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            )

    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=cfg.TRAINER.BATCH_SIZE[0], shuffle=False,
                                                sampler=source_sampler, num_workers=cfg.TRAINER.WORKERS, pin_memory=True)

    target_set = TSNDataSet(cfg.PATHS.PATH_DATA_TARGET + ".pkl", cfg.PATHS.TRAIN_TARGET_LIST,
                            num_dataload=num_target_train, num_segments=cfg.DATASET.NUM_SEGMENTS,
                            new_length=data_length, modality=cfg.DATASET.MODALITY,
                            image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                   "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            )

    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=cfg.TRAINER.BATCH_SIZE[1], shuffle=False,
                                                sampler=target_sampler, num_workers=cfg.TRAINER.WORKERS, pin_memory=True)

    return source_loader, target_loader


def get_val_data_loaders(cfg):
    if cfg.DATASET.MODALITY == 'Audio' or 'RGB' or cfg.DATASET.MODALITY == 'ALL':
        data_length = 1
    elif cfg.DATASET.MODALITY in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = len(pd.read_pickle(cfg.PATHS.TRAIN_SOURCE_LIST).index)
    num_target = len(pd.read_pickle(cfg.PATHS.TRAIN_TARGET_LIST).index)
    num_val = len(pd.read_pickle(cfg.PATHS.VAL_LIST).index)

    num_iter_source = num_source / cfg.TRAINER.BATCH_SIZE[0]
    num_iter_target = num_target / cfg.TRAINER.BATCH_SIZE[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[
                                                                              0] == 'Y' else num_source
    num_target_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[
                                                                              1] == 'Y' else num_target
    num_source_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[0] == 'Y' else num_source
    num_target_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[1] == 'Y' else num_target

    source_set_val = TSNDataSet(cfg.PATHS.PATH_VAL_DATA_SOURCE + ".pkl", cfg.PATHS.VAL_SOURCE_LIST,
                                num_dataload=num_source_val, num_segments=cfg.DATASET.VAL_SEGMENTS,
                                new_length=data_length, modality=cfg.DATASET.MODALITY,
                                image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                       "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                random_shift=False,
                                test_mode=True,
                                )

    source_sampler_val = torch.utils.data.sampler.RandomSampler(source_set_val)
    source_loader_val = torch.utils.data.DataLoader(source_set_val, batch_size=cfg.TRAINER.BATCH_SIZE[0], shuffle=False,
                                                    sampler=source_sampler_val, num_workers=cfg.TRAINER.WORKERS,
                                                    pin_memory=True)

    target_set_val = TSNDataSet(cfg.PATHS.PATH_VAL_DATA_TARGET + ".pkl", cfg.PATHS.VAL_TARGET_LIST,
                                num_dataload=num_target_val, num_segments=cfg.DATASET.VAL_SEGMENTS,
                                new_length=data_length, modality=cfg.DATASET.MODALITY,
                                image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                       "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                random_shift=False,
                                test_mode=True,
                                )

    target_sampler_val = torch.utils.data.sampler.RandomSampler(target_set_val)
    target_loader_val = torch.utils.data.DataLoader(target_set_val, batch_size=cfg.TRAINER.BATCH_SIZE[1], shuffle=False,
                                                    sampler=target_sampler_val, num_workers=cfg.TRAINER.WORKERS,
                                                    pin_memory=True)

    return source_loader_val, target_loader_val


def get_test_data_loaders(cfg):
    data_length = 1 if cfg.DATASET.MODALITY == "RGB" else 1
    num_test = len(pd.read_pickle(cfg.PATHS.VAL_LIST).index)
    if cfg.TESTER.NOUN_TARGET_DATA is not None:
        data_set = TSNDataSet(cfg.TESTER.TEST_TARGET_DATA + ".pkl", cfg.PATHS.VAL_LIST, num_dataload=num_test,
                              num_segments=cfg.TESTER.TEST_SEGMENTS,
                              new_length=data_length, modality=cfg.DATASET.MODALITY,
                              image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ['RGB', 'RGBDiff', 'RGBDiff2',
                                                                                     'RGBDiffplus'] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                              test_mode=True, noun_data_path=cfg.TESTER.NOUN_TARGET_DATA + ".pkl"
                              )
    else:
        data_set = TSNDataSet(cfg.TESTER.TEST_TARGET_DATA + ".pkl", cfg.PATHS.VAL_LIST, num_dataload=num_test,
                              num_segments=cfg.TESTER.TEST_SEGMENTS,
                              new_length=data_length, modality=cfg.DATASET.MODALITY,
                              image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ['RGB', 'RGBDiff', 'RGBDiff2',
                                                                                     'RGBDiffplus'] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                              test_mode=True
                              )
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=cfg.TESTER.BATCH_SIZE, shuffle=False,
                                              num_workers=cfg.TRAINER.WORKERS, pin_memory=True)

    return data_loader

import os

from model import TA3NTrainer

import torch
import torch.backends.cudnn as cudnn

from utils.logging import *


def set_hyperparameters(model, cfg):
    model.optimizerName = cfg.TRAINER.OPTIMIZER_NAME
    model.loss_type = cfg.TRAINER.LOSS_TYPE
    model.lr = cfg.TRAINER.LR
    model.momentum = cfg.TRAINER.MOMENTUM
    model.weight_decay = cfg.TRAINER.WEIGHT_DECAY
    model.epochs = cfg.TRAINER.MAX_EPOCHS
    model.batch_size = cfg.TRAINER.BATCH_SIZE

    model.lr_adaptive = cfg.TRAINER.LR_ADAPTIVE
    model.lr_decay = cfg.TRAINER.LR_DECAY
    model.lr_steps = cfg.TRAINER.LR_STEPS

    model.alpha = cfg.HYPERPARAMETERS.ALPHA
    model.beta = cfg.HYPERPARAMETERS.BETA
    model.gamma = cfg.HYPERPARAMETERS.GAMMA
    model.mu = cfg.HYPERPARAMETERS.MU

    model.train_metric = cfg.TRAINER.TRAIN_METRIC
    model.dann_warmup = cfg.TRAINER.DANN_WARMUP

    model.tensorboard = True
    model.path_exp = cfg.PATHS.EXP_PATH
    if not os.path.isdir(model.path_exp):
        os.makedirs(model.path_exp)

    model.pretrain_source = cfg.TRAINER.PRETRAIN_SOURCE
    model.clip_gradient = cfg.TRAINER.CLIP_GRADIENT

    model.dis_DA = cfg.MODEL.DIS_DA
    model.use_target = cfg.MODEL.USE_TARGET
    model.add_fc = cfg.MODEL.ADD_FC
    model.place_dis = cfg.MODEL.PLACE_DIS
    model.place_adv = cfg.MODEL.PLACE_ADV
    model.pred_normalize = cfg.MODEL.PRED_NORMALIZE
    model.add_loss_DA = cfg.MODEL.ADD_LOSS_DA
    model.ens_DA = cfg.MODEL.ENS_DA

    model.arch = cfg.MODEL.ARCH
    model.save_model = cfg.TRAINER.SAVE_MODEL
    model.labels_available = True
    model.adv_DA = cfg.MODEL.ADV_DA

    if model.loss_type == 'nll':
        model.criterion = torch.nn.CrossEntropyLoss()
        model.criterion_domain = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss type")


def initialise_trainer(cfg):
    log_debug('Baseline:' + cfg.DATASET.BASELINE_TYPE)
    log_debug('Frame aggregation method:' + cfg.DATASET.FRAME_AGGREGATION)

    log_debug('target data usage:' + cfg.MODEL.USE_TARGET)
    if cfg.MODEL.USE_TARGET is None:
        log_debug('no Domain Adaptation')
    else:
        if cfg.MODEL.DIS_DA is not None:
            log_debug('Apply the discrepancy-based Domain Adaptation approach:' + cfg.MODEL.DIS_DA)
            if len(cfg.MODEL.PLACE_DIS) != cfg.MODEL.ADD_FC + 2:
                log_error('len(place_dis) should be equal to add_fc + 2')
                raise ValueError('len(place_dis) should be equal to add_fc + 2')

        if cfg.MODEL.ADV_DA is not None:
            log_debug('Apply the adversarial-based Domain Adaptation approach:' + cfg.MODEL.ADV_DA)

        if cfg.MODEL.USE_BN is not None:
            log_debug('Apply the adaptive normalization approach:' + cfg.MODEL.USE_BN)

    # determine the categories
    # want to allow multi-label classes.

    # Original way to compute number of classes
    ####class_names = [line.strip().split(' ', 1)[1] for line in open(cfg.class_file)]
    ####num_class = len(class_names)

    # New approach
    num_class_str = cfg.DATASET.NUM_CLASSES.split(",")
    # single class
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
        for num in num_class_str:
            num_class.append(int(num))

    # === check the folder existence ===#
    path_exp = cfg.PATHS.EXP_PATH
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)

    # === initialize the model ===#
    log_info('preparing the model......')
    model = TA3NTrainer(num_class, cfg.DATASET.BASELINE_TYPE, cfg.DATASET.FRAME_AGGREGATION, cfg.DATASET.MODALITY,
                        train_segments=cfg.DATASET.NUM_SEGMENTS, val_segments=cfg.DATASET.NUM_SEGMENTS,
                        base_model=cfg.MODEL.ARCH, path_pretrained=cfg.TRAINER.PRETRAINED,
                        add_fc=cfg.MODEL.ADD_FC, fc_dim=cfg.MODEL.FC_DIM,
                        dropout_i=cfg.MODEL.DROPOUT_I, dropout_v=cfg.MODEL.DROPOUT_V,
                        partial_bn=not cfg.MODEL.NO_PARTIALBN,
                        use_bn=cfg.MODEL.USE_BN if cfg.MODEL.USE_TARGET is not None else None,
                        ens_DA=cfg.MODEL.ENS_DA if cfg.MODEL.USE_TARGET is not None else None,
                        n_rnn=cfg.MODEL.N_RNN, rnn_cell=cfg.MODEL.RNN_CELL, n_directions=cfg.MODEL.N_DIRECTIONS,
                        n_ts=cfg.MODEL.N_TS,
                        use_attn=cfg.MODEL.USE_ATTN, n_attn=cfg.MODEL.N_ATTN, use_attn_frame=cfg.MODEL.USE_ATTN_FRAME,
                        verbose=cfg.TRAINER.VERBOSE, share_params=cfg.MODEL.SHARE_PARAMS)

    if cfg.TRAINER.OPTIMIZER_NAME == 'SGD':
        log_debug('using SGD')
        model.optimizerName = 'SGD'
    elif cfg.TRAINER.OPTIMIZER_NAME == 'Adam':
        log_debug('using Adam')
        model.optimizerName = 'Adam'
    else:
        log_error('optimizer not support or specified!!!')
        exit()

    # === check point ===#
    log_debug('checking the checkpoint......')
    if cfg.TRAINER.RESUME != "":
        if os.path.isfile(cfg.TRAINER.RESUME):
            checkpoint = torch.load(cfg.TRAINER.RESUME)
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            log_debug("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAINER.RESUME, checkpoint['epoch']))
            if cfg.TRAINER.RESUME_HP:
                log_debug("=> loaded checkpoint hyper-parameters")
                model.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            log_error("=> no checkpoint found at '{}'".format(cfg.TRAINER.RESUME))

    cudnn.benchmark = True

    # --- Optimizer ---#
    # define loss function (criterion) and optimizer
    if cfg.TRAINER.LOSS_TYPE == 'nll':
        model.loss_type = 'nll'
    else:
        raise ValueError("Unknown loss type")

    set_hyperparameters(model, cfg)

    return model


def set_hyperparameters_test(model, cfg):
    model.batch_size = cfg.TRAINER.BATCH_SIZE
    model.alpha = cfg.HYPERPARAMETERS.ALPHA
    model.beta = cfg.HYPERPARAMETERS.BETA
    model.gamma = cfg.HYPERPARAMETERS.GAMMA
    model.mu = cfg.HYPERPARAMETERS.MU

    model.criterion = torch.nn.CrossEntropyLoss()
    model.criterion_domain = torch.nn.CrossEntropyLoss()


def initialise_tester(cfg):
    # New approach
    num_class_str = cfg.DATASET.NUM_CLASSES.split(",")
    # single class
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
    for num in num_class_str:
        num_class.append(int(num))

    verb_net = TA3NTrainer(num_class, cfg.DATASET.BASELINE_TYPE, cfg.DATASET.FRAME_AGGREGATION, cfg.DATASET.MODALITY,
                           train_segments=cfg.TESTER.TEST_SEGMENTS if cfg.DATASET.BASELINE_TYPE == 'video' else 1,
                           val_segments=cfg.TESTER.TEST_SEGMENTS if cfg.DATASET.BASELINE_TYPE == 'video' else 1,
                           base_model=cfg.MODEL.ARCH, add_fc=cfg.MODEL.ADD_FC, fc_dim=cfg.MODEL.FC_DIM,
                           share_params=cfg.MODEL.SHARE_PARAMS,
                           dropout_i=cfg.TESTER.DROPOUT_I, dropout_v=cfg.TESTER.DROPOUT_V, use_bn=cfg.MODEL.USE_BN,
                           partial_bn=False,
                           n_rnn=cfg.MODEL.N_RNN, rnn_cell=cfg.MODEL.RNN_CELL, n_directions=cfg.MODEL.N_DIRECTIONS,
                           n_ts=cfg.MODEL.N_TS,
                           use_attn=cfg.MODEL.USE_ATTN, n_attn=cfg.MODEL.N_ATTN,
                           use_attn_frame=cfg.MODEL.USE_ATTN_FRAME,
                           verbose=cfg.TESTER.VERBOSE, before_softmax=False)

    verb_checkpoint = torch.load(cfg.TESTER.WEIGHTS)
    # print(verb_checkpoint)
    verb_base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(verb_checkpoint['state_dict'].items())}
    verb_net.load_state_dict(verb_base_dict)
    # verb_net = torch.nn.DataParallel(verb_net)
    set_hyperparameters_test(verb_net, cfg)
    verb_net.eval()

    if cfg.TESTER.NOUN_WEIGHTS is not None:
        noun_net = TA3NTrainer(num_class, cfg.DATASET.BASELINE_TYPE, cfg.DATASET.FRAME_AGGREGATION,
                               cfg.DATASET.MODALITY,
                               train_segments=cfg.TESTER.TEST_SEGMENTS if cfg.DATASET.BASELINE_TYPE == 'video' else 1,
                               val_segments=cfg.TESTER.TEST_SEGMENTS if cfg.DATASET.BASELINE_TYPE == 'video' else 1,
                               base_model=cfg.MODEL.ARCH, add_fc=cfg.MODEL.ADD_FC, fc_dim=cfg.MODEL.FC_DIM,
                               share_params=cfg.MODEL.SHARE_PARAMS,
                               dropout_i=cfg.TESTER.DROPOUT_I, dropout_v=cfg.TESTER.DROPOUT_V, use_bn=cfg.MODEL.USE_BN,
                               partial_bn=False,
                               n_rnn=cfg.MODEL.N_RNN, rnn_cell=cfg.MODEL.RNN_CELL, n_directions=cfg.MODEL.N_DIRECTIONS,
                               n_ts=cfg.MODEL.N_TS,
                               use_attn=cfg.MODEL.USE_ATTN, n_attn=cfg.MODEL.N_ATTN,
                               use_attn_frame=cfg.MODEL.USE_ATTN_FRAME,
                               verbose=cfg.TESTER.VERBOSE, before_softmax=False)
        noun_checkpoint = torch.load(cfg.TESTER.NOUN_WEIGHTS)

        noun_base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(noun_checkpoint['state_dict'].items())}
        noun_net.load_state_dict(noun_base_dict)
        # noun_net = torch.nn.DataParallel(noun_net.cuda())
        set_hyperparameters_test(noun_net, cfg)
        noun_net.eval()
    else:
        noun_net = None

    return (verb_net, noun_net)

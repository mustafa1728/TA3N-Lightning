"""
Default configurations for action recognition TA3N lightning
"""

import os
from pickle import NONE

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.TO_VALIDATE = False # choices = [True, False]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

# dataset paths
_C.DATASET.PATH_DATA_ROOT = "data/" # directory where the feature pickles are stored. Depends on users
_C.DATASET.PATH_LABELS_ROOT = "data/" # directory where the annotations are stored. Depends on users
_C.DATASET.PATH_EXP_ROOT="model/action-model/" # directory where the checkpoints are to be stored. Depends on users


_C.DATASET.DATASET_SOURCE="source_train" # depends on users
_C.DATASET.DATASET_TARGET="target_train" # depends on users
if _C.TO_VALIDATE:
    _C.DATASET.VAL_DATASET_SOURCE="source_val" # depends on users
    _C.DATASET.VAL_DATASET_TARGET="target_val" # depends on users
else:
    _C.DATASET.VAL_DATASET_SOURCE= None 
    _C.DATASET.VAL_DATASET_TARGET= None 
_C.DATASET.NUM_SOURCE= 16115 # number of training data (source)
_C.DATASET.NUM_TARGET= 26115 # number of training data (target)

_C.DATASET.PATH_DATA_SOURCE=os.path.join(_C.DATASET.PATH_DATA_ROOT, _C.DATASET.DATASET_SOURCE)
_C.DATASET.PATH_DATA_TARGET=os.path.join(_C.DATASET.PATH_DATA_ROOT, _C.DATASET.DATASET_TARGET)
if _C.TO_VALIDATE:
    _C.DATASET.PATH_VAL_DATA_SOURCE=os.path.join(_C.DATASET.PATH_DATA_ROOT, _C.DATASET.VAL_DATASET_SOURCE)
    _C.DATASET.PATH_VAL_DATA_TARGET=os.path.join(_C.DATASET.PATH_DATA_ROOT, _C.DATASET.VAL_DATASET_TARGET)
else:
    _C.DATASET.PATH_VAL_DATA_SOURCE= None 
    _C.DATASET.PATH_VAL_DATA_SOURCE= None 

_C.DATASET.TRAIN_SOURCE_LIST=os.path.join(_C.DATASET.PATH_LABELS_ROOT, 'EPIC_100_uda_source_train.pkl') # '/domain_adaptation_source_train_pre-release_v3.pkl'
_C.DATASET.TRAIN_TARGET_LIST=os.path.join(_C.DATASET.PATH_LABELS_ROOT, 'EPIC_100_uda_target_train_timestamps.pkl') # '/domain_adaptation_target_train_pre-release_v6.pkl'
if _C.TO_VALIDATE:
    _C.DATASET.VAL_SOURCE_LIST=os.path.join(_C.DATASET.PATH_LABELS_ROOT, "EPIC_100_uda_source_val.pkl")
    _C.DATASET.VAL_TARGET_LIST=os.path.join(_C.DATASET.PATH_LABELS_ROOT, "EPIC_100_uda_target_val.pkl")
else:
    _C.DATASET.VAL_SOURCE_LIST= None 
    _C.DATASET.VAL_TARGET_LIST= None 
_C.DATASET.VAL_LIST=os.path.join(_C.DATASET.PATH_LABELS_ROOT, "EPIC_100_uda_target_test_timestamps.pkl")
_C.DATASET.PATH_EXP=os.path.join(_C.DATASET.PATH_EXP_ROOT, "Testexp")

# dataset parameters
_C.DATASET = CN()
_C.DATASET.DATASET = "epic" # dataset choices = [hmdb_ucf, hmdb_ucf_small, ucf_olympic]
_C.DATASET.NUM_CLASSES = 97300
_C.DATASET.MODALITY = "ALL" # choices = [RGB ]
_C.DATASET.FRAME_TYPE = "feature" # choices = [frame]
_C.DATASET.NUM_SEGMENTS = 5 # sample frame # of each video for training
_C.DATASET.BASELINE_TYPE = "video" # choices = ['frame', 'tsn']
_C.DATASET.FRAME_AGGREGATION = "trn-m" # method to integrate the frame-level features. choices = [avgpool, trn, trn-m, rnn, temconv]

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

_C.MODEL = CN()
_C.MODEL.ADD_FC = 1 # number of shared features
_C.MODEL.FC_DIM = 512 # dimension of shared features
_C.MODEL.ARCH = "TBN" # choices  = [resnet50]
_C.MODEL.USE_TARGET = "uSv" # choices  = [uSv, Sv, none]
_C.MODEL.SHARE_PARAMS = "Y" # choices  = [Y, N]
_C.MODEL.PRED_NORMALIZE = "N" # choices  = [Y, N]
_C.MODEL.WEIGHTED_CLASS_LOSS_DA = "N" # choices  = [Y, N]
_C.MODEL.WEIGHTED_CLASS_LOSS = "N" # choices  = [Y, N]


# DA configs
if _C.MODEL.USE_TARGET == "none":
	_C.MODEL.EXP_DA_NAME="baseline"
else:
	_C.MODEL.EXP_DA_NAME="DA"
_C.MODEL.DIS_DA = None # choices  = [DAN, JAN]
_C.MODEL.ADV_POS_0 = "Y" # discriminator for relation features. choices  = [Y, N]
_C.MODEL.ADV_DA = "RevGrad" # choices  = [None]
_C.MODEL.ADD_LOSS_DA = "attentive_entropy" # choices  = [None, target_entropy, attentive_entropy]
_C.MODEL.ENS_DA = None # choices  = [None, MCD]

# Attention configs
_C.MODEL.USE_ATTN = "TransAttn" # choices  = [None, TransAttn, general]
_C.MODEL.USE_ATTN_FRAME = None # choices  = [None, TransAttn, general]
_C.MODEL.USE_BN = None # choices  = [None, AdaBN, AutoDIAL]

# ---------------------------------------------------------------------------- #
# Hyperparameters
# ---------------------------------------------------------------------------- #
_C.HYPERPARAMETERS = CN()
_C.HYPERPARAMETERS.ALPHA = 0
_C.HYPERPARAMETERS.BETA = [0.75, 0.75, 0.5]
_C.HYPERPARAMETERS.N_ATTN = 1
_C.HYPERPARAMETERS.GAMMA = 0.003 # U->H: 0.003 | H->U: 0.3
_C.HYPERPARAMETERS.MU = 0

# ---------------------------------------------------------------------------- #
# Trainer
# ---------------------------------------------------------------------------- #

_C.TRAINER = CN()
_C.TRAINER.TRAIN_METRIC = "all" # choices  = [noun, verb]
_C.TRAINER.FC_DIM = 512 # dimension of shared features
_C.TRAINER.ARCH = "TBN" # choices  = [resnet50]
_C.TRAINER.USE_TARGET = "uSv" # choices  = [uSv, Sv, none]
_C.TRAINER.SHARE_PARAMS = "Y" # choices  = [Y, N]

# Learning configs
_C.TRAINER.LR = 0.003
_C.TRAINER.LR_DECAY = 10
_C.TRAINER.LR_ADAPTIVE = None # choices = [None, loss, dann]
_C.TRAINER.LR_STEPS = [10, 20]
_C.TRAINER.BATCH_SIZE = [128, 128*(_C.DATASET.NUM_TARGET/_C.DATASET.NUM_SOURCE), 128]
_C.TRAINER.OPTIMIZER_NAME = "SGD" # choices = [SGD, Adam]
_C.TRAINER.GD = 20

_C.TRAINER.PRETRAINED = None

_C.TRAINER.MIN_EPOCHS = 25
_C.TRAINER.MAX_EPOCHS = 30

# ---------------------------------------------------------------------------- #
# Miscellaneous configs
# ---------------------------------------------------------------------------- #

_C.MODEL.N_RNN = 1
_C.MODEL.RNN_CELL = "LSTM"
_C.MODEL.N_DIRECTIONS = 1
_C.MODEL.N_TS = 5
_C.MODEL.TENSORBOARD = True
_C.TRAINER.JOBS = 2
_C.TRAINER.EF = 1
_C.TRAINER.PF = 50
_C.TRAINER.SF = 50
_C.TRAINER.COPY_LIST = ["N", "N"]



_C.DATASET.EXP_PATH = os.path.join(_C.DATASET.PATH_EXP + '_' + _C.TRAINER.OPTIMIZER_NAME + '-share_params_' + _C.MODEL.SHARE_PARAMS + '-lr_' + str(_C.TRAINER.LR) + '-bS_' + str(_C.TRAINER.BATCH_SIZE[0]), _C.DATASET.DATASET + '-'+ str(_C.DATASET.NUM_SEGMENTS) + '-seg-disDA_' + _C.MODEL.DIS_DA + '-alpha_' + str(_C.HYPERPARAMETERS.ALPHA) + '-advDA_' + _C.MODEL.ADV_DA + '-beta_' + str(_C.HYPERPARAMETERS.BETA[0])+ '_'+ str(_C.HYPERPARAMETERS.BETA[1])+'_'+ str(_C.HYPERPARAMETERS.BETA[2])+"_gamma_" + str(_C.HYPERPARAMETERS.GAMMA) + "_mu_" + str(_C.HYPERPARAMETERS.MU))



def get_cfg_defaults():
    return _C.clone()
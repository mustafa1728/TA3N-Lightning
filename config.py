"""
Default configurations for action recognition TA3N lightning
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.TO_VALIDATE = False # choices = [True, False]


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
_C.PATHS = CN()
_C.PATHS.PATH_DATA_ROOT = "data/" # directory where the feature pickles are stored. Depends on users
_C.PATHS.PATH_LABELS_ROOT = "annotations/" # directory where the annotations are stored. Depends on users
_C.PATHS.PATH_EXP_ROOT="model/action-model/" # directory where the checkpoints are to be stored. Depends on users

_C.PATHS.DATASET_SOURCE="source_train" # depends on users
_C.PATHS.DATASET_TARGET="target_train" # depends on users


# training   
_C.PATHS.PATH_DATA_SOURCE=os.path.join(_C.PATHS.PATH_DATA_ROOT, _C.PATHS.DATASET_SOURCE)
_C.PATHS.PATH_DATA_TARGET=os.path.join(_C.PATHS.PATH_DATA_ROOT, _C.PATHS.DATASET_TARGET)

_C.PATHS.TRAIN_SOURCE_LIST=os.path.join(_C.PATHS.PATH_LABELS_ROOT, 'EPIC_100_uda_source_train.pkl') # '/domain_adaptation_source_train_pre-release_v3.pkl'
_C.PATHS.TRAIN_TARGET_LIST=os.path.join(_C.PATHS.PATH_LABELS_ROOT, 'EPIC_100_uda_target_train_timestamps.pkl') # '/domain_adaptation_target_train_pre-release_v6.pkl'

_C.PATHS.VAL_LIST=os.path.join(_C.PATHS.PATH_LABELS_ROOT, "EPIC_100_uda_target_test_timestamps.pkl")
_C.PATHS.PATH_EXP=os.path.join(_C.PATHS.PATH_EXP_ROOT, "Testexp")


# validation
_C.PATHS.VAL_DATASET_SOURCE="source_val" # depends on users
_C.PATHS.VAL_DATASET_TARGET="target_val" # depends on users

_C.PATHS.PATH_VAL_DATA_SOURCE=os.path.join(_C.PATHS.PATH_DATA_ROOT, _C.PATHS.VAL_DATASET_SOURCE)
_C.PATHS.PATH_VAL_DATA_TARGET=os.path.join(_C.PATHS.PATH_DATA_ROOT, _C.PATHS.VAL_DATASET_TARGET)

_C.PATHS.VAL_SOURCE_LIST=os.path.join(_C.PATHS.PATH_LABELS_ROOT, "EPIC_100_uda_source_val.pkl")
_C.PATHS.VAL_TARGET_LIST=os.path.join(_C.PATHS.PATH_LABELS_ROOT, "EPIC_100_uda_target_val.pkl")

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET = "epic" # dataset choices = [hmdb_ucf, hmdb_ucf_small, ucf_olympic]
_C.DATASET.NUM_CLASSES = "97,300"
_C.DATASET.NUM_SOURCE= 16115 # number of training data (source)
_C.DATASET.NUM_TARGET= 26115 # number of training data (target)

_C.DATASET.MODALITY = "RGB" # choices = [ALL, RGB, Audio, Flow]
_C.DATASET.FRAME_TYPE = "feature" # choices = [frame]
_C.DATASET.NUM_SEGMENTS = 5 # sample frame # of each video for training
_C.DATASET.VAL_SEGMENTS = 5 # sample frame # of each video for training
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

_C.MODEL.DROPOUT_I = 0.5
_C.MODEL.DROPOUT_V = 0.5
_C.MODEL.NO_PARTIALBN = True


# DA configs
if _C.MODEL.USE_TARGET == "none":
	_C.MODEL.EXP_DA_NAME="baseline"
else:
	_C.MODEL.EXP_DA_NAME="DA"
_C.MODEL.DIS_DA = "DAN" # choices  = [DAN, CORAL, JAN]
_C.MODEL.ADV_POS_0 = "Y" # discriminator for relation features. choices  = [Y, N]
_C.MODEL.ADV_DA = "RevGrad" # choices  = [None]
_C.MODEL.ADD_LOSS_DA = "attentive_entropy" # choices  = [None, target_entropy, attentive_entropy]
_C.MODEL.ENS_DA = None # choices  = [None, MCD]

# Attention configs
_C.MODEL.USE_ATTN = "TransAttn" # choices  = [None, TransAttn, general]
_C.MODEL.USE_ATTN_FRAME = None # choices  = [None, TransAttn, general]
_C.MODEL.USE_BN = None # choices  = [None, AdaBN, AutoDIAL]
_C.MODEL.N_ATTN = 1
_C.MODEL.PLACE_DIS = ["Y", "Y", "N"]
_C.MODEL.PLACE_ADV = ["Y", "Y", "Y"]


# ---------------------------------------------------------------------------- #
# Hyperparameters
# ---------------------------------------------------------------------------- #
_C.HYPERPARAMETERS = CN()
_C.HYPERPARAMETERS.ALPHA = 0
_C.HYPERPARAMETERS.BETA = [0.75, 0.75, 0.5]
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
_C.TRAINER.PRETRAIN_SOURCE = False
_C.TRAINER.VERBOSE = True
_C.TRAINER.DANN_WARMUP = True

# Learning configs
_C.TRAINER.LOSS_TYPE = 'nll'
_C.TRAINER.LR = 0.003
_C.TRAINER.LR_DECAY = 10
_C.TRAINER.LR_ADAPTIVE = None # choices = [None, loss, dann]
_C.TRAINER.LR_STEPS = [10, 20]
_C.TRAINER.MOMENTUM = 0.9
_C.TRAINER.WEIGHT_DECAY = 0.0001
_C.TRAINER.BATCH_SIZE = [128, int(128*_C.DATASET.NUM_TARGET/_C.DATASET.NUM_SOURCE), 128]
_C.TRAINER.OPTIMIZER_NAME = "SGD" # choices = [SGD, Adam]
_C.TRAINER.CLIP_GRADIENT = 20

_C.TRAINER.PRETRAINED = None
_C.TRAINER.RESUME = ""
_C.TRAINER.RESUME_HP = ""

_C.TRAINER.MIN_EPOCHS = 25
_C.TRAINER.MAX_EPOCHS = 30

_C.TRAINER.ACCELERATOR = "ddp"



_C.PATHS.EXP_PATH = os.path.join(_C.PATHS.PATH_EXP + '_' + _C.TRAINER.OPTIMIZER_NAME + '-share_params_' + _C.MODEL.SHARE_PARAMS + '-lr_' + str(_C.TRAINER.LR) + '-bS_' + str(_C.TRAINER.BATCH_SIZE[0]), _C.DATASET.DATASET + '-'+ str(_C.DATASET.NUM_SEGMENTS) + '-alpha_' + str(_C.HYPERPARAMETERS.ALPHA) + '-beta_' + str(_C.HYPERPARAMETERS.BETA[0])+ '_'+ str(_C.HYPERPARAMETERS.BETA[1])+'_'+ str(_C.HYPERPARAMETERS.BETA[2])+"_gamma_" + str(_C.HYPERPARAMETERS.GAMMA) + "_mu_" + str(_C.HYPERPARAMETERS.MU))


# ---------------------------------------------------------------------------- #
# Tester
# ---------------------------------------------------------------------------- #
_C.TESTER = CN()

_C.TESTER.TEST_TARGET_DATA = os.path.join(_C.PATHS.PATH_DATA_ROOT, "target_test")

_C.TESTER.WEIGHTS = os.path.join(_C.PATHS.EXP_PATH , "checkpoint.pth.tar")
_C.TESTER.NOUN_WEIGHTS = None
_C.TESTER.BATCH_SIZE = 512
_C.TESTER.NOUN_TARGET_DATA = None
_C.TESTER.RESULT_JSON = "test.json"
_C.TESTER.TEST_SEGMENTS = 5 # sample frame # of each video for testing
_C.TESTER.SAVE_SCORES = os.path.join(_C.PATHS.EXP_PATH , "scores")
_C.TESTER.SAVE_CONFUSION = os.path.join(_C.PATHS.EXP_PATH , "confusion_matrix")

_C.TESTER.VERBOSE = True

# ---------------------------------------------------------------------------- #
# Miscellaneous configs
# ---------------------------------------------------------------------------- #

_C.MODEL.N_RNN = 1
_C.MODEL.RNN_CELL = "LSTM"
_C.MODEL.N_DIRECTIONS = 1
_C.MODEL.N_TS = 5
_C.MODEL.TENSORBOARD = True
_C.MODEL.FLOW_PREFIX = ""
_C.TRAINER.JOBS = 2
_C.TRAINER.EF = 1
_C.TRAINER.PF = 50
_C.TRAINER.SF = 50
_C.TRAINER.COPY_LIST = ["N", "N"]
_C.TRAINER.SAVE_MODEL = True






def get_cfg_defaults():
    return _C.clone()
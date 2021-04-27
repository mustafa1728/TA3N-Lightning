import argparse
import time
import sys

import json
from json import encoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from pytorch_lightning import Trainer

from dataset import TSNDataSet
from models_lightning import VideoModel
from utils.utils import plot_confusion_matrix

from colorama import init
from colorama import Fore, Back, Style
from tqdm import tqdm
from time import sleep
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
init(autoreset=True)

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('num_class', type=str, default="classInd.txt")
parser.add_argument('modality', type=str, choices=['ALL', 'Audio','RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('test_target_data', type=str)
parser.add_argument('result_json', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--noun_target_data', type=str, default=None)
parser.add_argument('--noun_weights', type=str, default=None)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--test_segments', type=int, default=5)
parser.add_argument('--add_fc', default=1, type=int, metavar='M', help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)')
parser.add_argument('--fc_dim', type=int, default=512, help='dimension of added fc')
parser.add_argument('--baseline_type', type=str, default='frame', choices=['frame', 'video', 'tsn'])
parser.add_argument('--frame_aggregation', type=str, default='avgpool', choices=['avgpool', 'rnn', 'temconv', 'trn-m', 'none'], help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--dropout_i', type=float, default=0)
parser.add_argument('--dropout_v', type=float, default=0)

#------ RNN ------
parser.add_argument('--n_rnn', default=1, type=int, metavar='M',
                    help='number of RNN layers (e.g. 0, 1, 2, ...)')
parser.add_argument('--rnn_cell', type=str, default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('--n_directions', type=int, default=1, choices=[1, 2],
                    help='(bi-) direction RNN')
parser.add_argument('--n_ts', type=int, default=5, help='number of temporal segments')

# ========================= DA Configs ==========================
parser.add_argument('--share_params', type=str, default='Y', choices=['Y', 'N'])
parser.add_argument('--use_bn', type=str, default='none', choices=['none', 'AdaBN', 'AutoDIAL'])
parser.add_argument('--use_attn_frame', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism for frames only')
parser.add_argument('--use_attn', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism')
parser.add_argument('--n_attn', type=int, default=1, help='number of discriminators for transferable attention')

# ========================= Monitor Configs ==========================
parser.add_argument('--top', default=[1, 3, 5], nargs='+', type=int, help='show top-N categories')
parser.add_argument('--verbose', default=False, action="store_true")

# ========================= Runtime Configs ==========================
parser.add_argument('--save_confusion', type=str, default=None)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--save_attention', type=str, default=None)
parser.add_argument('--max_num', type=int, default=-1, help='number of videos to test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--bS', default=2, help='batch size', type=int, required=False)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()
gpu_count = torch.cuda.device_count()
# New approach
num_class_str = args.num_class.split(",")
# single class
if len(num_class_str) < 1:
	raise Exception("Must specify a number of classes to train")
else:
	num_class = []
	for num in num_class_str:
		num_class.append(int(num))

criterion = torch.nn.CrossEntropyLoss().cuda()

#=== Load the network ===#
print(Fore.CYAN + 'preparing the model......')
verb_net = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
		train_segments=args.test_segments if args.baseline_type == 'video' else 1, val_segments=args.test_segments if args.baseline_type == 'video' else 1,
		base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
		dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
		n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
		use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
		verbose=args.verbose, before_softmax=False)

verb_checkpoint = torch.load(args.weights)

verb_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(verb_checkpoint['state_dict'].items())}
verb_net.load_state_dict(verb_base_dict)
# verb_net = torch.nn.DataParallel(verb_net)
verb_net.eval()

if args.noun_weights is not None:
	noun_net = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
						  train_segments=args.test_segments if args.baseline_type == 'video' else 1,
						  val_segments=args.test_segments if args.baseline_type == 'video' else 1,
						  base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
						  dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
						  n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
						  use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
						  verbose=args.verbose, before_softmax=False)
	noun_checkpoint = torch.load(args.noun_weights)

	noun_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(noun_checkpoint['state_dict'].items())}
	noun_net.load_state_dict(noun_base_dict)
	# noun_net = torch.nn.DataParallel(noun_net.cuda())
	noun_net.eval()
else:
	noun_net = None


#=== Data loading ===#
print(Fore.CYAN + 'loading data......')

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
print('\n', Fore.CYAN + 'data loaded from: ', args.test_target_data+".pkl")



print(Fore.CYAN + 'starting validation......')
trainer = Trainer()

trainer.test(model = verb_net, test_dataloaders=data_loader, ckpt_path=args.weights, verbose = True)
print(Fore.CYAN + 'validation complete')
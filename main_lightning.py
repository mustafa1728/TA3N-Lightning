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
from models_lightning import VideoModel
from loss import *
from opts import parser
from utils.utils import randSelectBatch
import math
import pandas as pd

from colorama import init
from colorama import Fore, Back, Style
import numpy as np


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()
print(Fore.YELLOW + "Number of GPUS available: ", gpu_count)

def main():
	global args
	args = parser.parse_args()

	print(Fore.GREEN + 'Baseline:', args.baseline_type)
	print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)

	print(Fore.GREEN + 'target data usage:', args.use_target)
	if args.use_target == 'none':
		print(Fore.GREEN + 'no Domain Adaptation')
	else:
		if args.dis_DA != 'none':
			print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
			if len(args.place_dis) != args.add_fc + 2:
				raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

		if args.adv_DA != 'none':
			print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

		if args.use_bn != 'none':
			print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

	# determine the categories
	#want to allow multi-label classes.

	#Original way to compute number of classes
	####class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
	####num_class = len(class_names)

	#New approach
	num_class_str = args.num_class.split(",")
	#single class
	if len(num_class_str) < 1:
		raise Exception("Must specify a number of classes to train")
	else:
		num_class = []
		for num in num_class_str:
			num_class.append(int(num))

	#=== check the folder existence ===#
	path_exp = args.exp_path + args.modality + '/'
	if not os.path.isdir(path_exp):
		os.makedirs(path_exp)

	if args.tensorboard:
		writer_train = SummaryWriter(path_exp + '/tensorboard_train')  # for tensorboardX
		writer_val = SummaryWriter(path_exp + '/tensorboard_val')  # for tensorboardX
	#=== initialize the model ===#
	print(Fore.CYAN + 'preparing the model......')
	model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
				train_segments=args.num_segments, val_segments=args.val_segments, 
				base_model=args.arch, path_pretrained=args.pretrained,
				add_fc=args.add_fc, fc_dim = args.fc_dim,
				dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
				use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
				n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
				use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
				verbose=args.verbose, share_params=args.share_params)

	if args.optimizer == 'SGD':
		print(Fore.YELLOW + 'using SGD')
		model.optimizerName = 'SGD'
	elif args.optimizer == 'Adam':
		print(Fore.YELLOW + 'using Adam')
		model.optimizerName = 'Adam'
	else:
		print(Back.RED + 'optimizer not support or specified!!!')
		exit()

	#=== check point ===#
	start_epoch = 1
	print(Fore.CYAN + 'checking the checkpoint......')
	if args.resume:
		if os.path.isfile(args.resume):
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch'] + 1
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch'])))
			if args.resume_hp:
				print("=> loaded checkpoint hyper-parameters")
				optimizer.load_state_dict(checkpoint['optimizer'])
		else:
			print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	#--- open log files ---#
	if args.resume:
		train_file = open(path_exp + 'train.log', 'a')
		train_short_file = open(path_exp + 'train_short.log', 'a')
		val_file = open(path_exp + 'val.log', 'a')
		val_short_file = open(path_exp + 'val_short.log', 'a')
		train_file.write('========== start: ' + str(start_epoch) + '\n')  # separation line
		train_short_file.write('========== start: ' + str(start_epoch) + '\n')
		val_file.write('========== start: ' + str(start_epoch) + '\n')
		val_short_file.write('========== start: ' + str(start_epoch) + '\n')
	else:
		train_short_file = open(path_exp + 'train_short.log', 'w')
		val_short_file = open(path_exp + 'val_short.log', 'w')
		train_file = open(path_exp + 'train.log', 'w')
		val_file = open(path_exp + 'val.log', 'w')
	val_best_file = open(path_exp + 'best_val.log', 'a')

    # --- Optimizer ---#
	# define loss function (criterion) and optimizer
	if args.loss_type == 'nll':
		model.loss_type = 'nll'
	else:
		raise ValueError("Unknown loss type")

    # --- Parameters ---#
	model.beta = args.beta
	model.gamma = args.gamma
	model.mu = args.mu

	#=== Data loading ===#
	print(Fore.CYAN + 'loading data......')

	if args.use_opencv:
		print("use opencv functions")

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

	to_validate  = args.val_source_data != "none" and args.val_target_data != "none" 
	if(to_validate):
		print(Fore.CYAN + 'Loading validation data......')
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



	if args.train_metric == "all":
		monitor = "Prec@1 Action"
	elif args.train_metric == "noun":
		monitor = "Prec@1 Noun"
	elif args.train_metric == "verb":
		monitor = "Prec@1 Verb" 
	else:
		raise Exception("invalid metric to train")

	checkpoint_callback = ModelCheckpoint(
		monitor = monitor,
		dirpath=path_exp,
		filename='checkpoint',
		mode = 'max'
	)
	checkpoint_callback.FILE_EXTENSION = ".pth.tar"

	#=== Actual Training ===#
	
	trainer = Trainer(min_epochs=20, max_epochs=30, callbacks=[checkpoint_callback], gpus = gpu_count, accelerator='ddp')

	print(Fore.CYAN + 'start training......')	
	start_train = time.time()
	if(to_validate):
		trainer.fit(model, (source_loader, target_loader), (source_loader_val, target_loader_val))
	else:
		trainer.fit(model, (source_loader, target_loader))

	
	end_train = time.time()
	print(Fore.CYAN + 'total training time:', end_train - start_train)


	# --- write the total time to log files ---#
	line_time = 'total time: {:.3f} '.format(end_train - start_train)

	train_file.write(line_time)
	train_short_file.write(line_time)

	#--- close log files ---#
	train_file.close()
	train_short_file.close()

	if target_set.labels_available:
		val_best_file.write('%.3f\n' % best_prec1)
		val_file.write(line_time)
		val_short_file.write(line_time)
		val_file.close()
		val_short_file.close()

	if args.tensorboard:
		writer_train.close()
		writer_val.close()

	print(Fore.CYAN + 'Training complete')
if __name__ == '__main__':
	main()

import os
import numpy as np
import time
import argparse

import torch
import torch.nn.parallel
import torch.optim

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tensorboardX import SummaryWriter

from utils.loss import *
from utils.model_init import initialise_trainer
from config import get_cfg_defaults
from utils.data_loaders import get_train_data_loaders, get_val_data_loaders
from utils.logging import *



np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

gpu_count = torch.cuda.device_count()
log_info( "Number of GPUS available: " + str(gpu_count))

def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="TA3N Domain Adaptation")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--gpus", default="0", help="gpu id(s) to use", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args

def main():
	args = arg_parse()
	cfg = get_cfg_defaults()
	cfg.merge_from_file(args.cfg)
	cfg.freeze()

	# log_info(str(cfg))

	path_exp = os.path.join(cfg.PATHS.EXP_PATH, cfg.DATASET.MODALITY)

	#========== model init ========#

	log_info('Initialising model......')
	model = initialise_trainer(cfg)

	#========== log files init ========#

	open_log_files(cfg)
	
	#========== Data loading ========#
	
	log_info('Loading data......')
	source_loader, target_loader = get_train_data_loaders(cfg)
	
	if(cfg.TO_VALIDATE):
		log_info('Loading validation data......')
		source_loader_val, target_loader_val = get_val_data_loaders(cfg)

	#========== Callbacks and checkpoints ========#

	if cfg.TRAINER.TRAIN_METRIC == "all":
		monitor = "Prec@1 Action"
	elif cfg.TRAINER.TRAIN_METRIC == "noun":
		monitor = "Prec@1 Noun"
	elif cfg.TRAINER.TRAIN_METRIC == "verb":
		monitor = "Prec@1 Verb" 
	else:
		log_error("invalid metric to train")
		raise Exception("invalid metric to train")

	checkpoint_callback = ModelCheckpoint(
		monitor = monitor,
		dirpath=path_exp,
		filename='checkpoint',
		mode = 'max'
	)
	checkpoint_callback.FILE_EXTENSION = ".pth.tar"

	#========== Actual Training ========#
	
	trainer = Trainer(
		min_epochs=cfg.TRAINER.MIN_EPOCHS, 
		max_epochs=cfg.TRAINER.MAX_EPOCHS, 
		callbacks=[checkpoint_callback], 
		gpus = args.gpus, 
		accelerator=cfg.TRAINER.ACCELERATOR
	)

	log_info('Starting training......')	
	start_train = time.time()

	if(cfg.TO_VALIDATE):
		trainer.fit(model, (source_loader, target_loader), (source_loader_val, target_loader_val))
	else:
		trainer.fit(model, (source_loader, target_loader))
	
	end_train = time.time()

	#========== Logging ========#

	write_log_files('total time: {:.3f} '.format(end_train - start_train), model.best_prec1)
	model.writer_train.close()
	model.writer_val.close()
	
	log_info('Training complete')
	log_info('Total training time:' + str(end_train - start_train))
	
	if(cfg.TO_VALIDATE):
		log_info('Validation scores:\n |  Prec@1 Verb: ' + str(model.prec1_verb_val) + "\n |  Prec@1 Noun: " + str(model.prec1_noun_val)+ "\n |  Prec@1 Action: " + str(model.prec1_val) + "\n |  Prec@5 Verb: " + str(model.prec5_verb_val) + "\n |  Prec@5 Noun: " + str(model.prec5_noun_val) + "\n |  Prec@5 Action: " + str(model.prec5_val) + "\n |  Loss total: " + str(model.losses_val))


if __name__ == '__main__':
	main()

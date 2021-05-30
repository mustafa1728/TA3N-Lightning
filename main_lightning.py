import numpy as np
import time

import torch
import torch.nn.parallel
import torch.optim

from colorama import init
from colorama import Fore, Back, Style


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tensorboardX import SummaryWriter

from utils.loss import *
from utils.opts import parser
from utils.model_init import initialise_trainer
from utils.data_loaders import get_train_data_loaders, get_val_data_loaders
from utils.logging import open_log_files, write_log_files

import logging
logging.basicConfig(format='%(asctime)s  |  %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()
logging.info( "Number of GPUS available: " + str(gpu_count))

def main():
	args = parser.parse_args()

	path_exp = args.exp_path + args.modality + '/'

	#========== model init ========#

	logging.info('Initialising model......')
	model = initialise_trainer(args)

	#========== log files init ========#

	open_log_files(args)
	
	#========== Data loading ========#
	
	logging.info('Loading data......')

	if args.use_opencv:
		logging.debug("use opencv functions")

	source_loader, target_loader = get_train_data_loaders(args)
	
	to_validate  = args.val_source_data != "none" and args.val_target_data != "none" 
	if(to_validate):
		logging.info('Loading validation data......')
		source_loader_val, target_loader_val = get_val_data_loaders(args)

	#========== Callbacks and checkpoints ========#

	if args.train_metric == "all":
		monitor = "Prec@1 Action"
	elif args.train_metric == "noun":
		monitor = "Prec@1 Noun"
	elif args.train_metric == "verb":
		monitor = "Prec@1 Verb" 
	else:
		logging.error("invalid metric to train")
		raise Exception("invalid metric to train")

	checkpoint_callback = ModelCheckpoint(
		monitor = monitor,
		dirpath=path_exp,
		filename='checkpoint',
		mode = 'max'
	)
	checkpoint_callback.FILE_EXTENSION = ".pth.tar"

	#========== Actual Training ========#
	
	trainer = Trainer(min_epochs=20, max_epochs=30, callbacks=[checkpoint_callback], gpus = gpu_count, accelerator='ddp')

	logging.info('Starting training......')	
	start_train = time.time()

	if(to_validate):
		trainer.fit(model, (source_loader, target_loader), (source_loader_val, target_loader_val))
	else:
		trainer.fit(model, (source_loader, target_loader))
	
	end_train = time.time()

	#========== Logging ========#

	write_log_files('total time: {:.3f} '.format(end_train - start_train), best_prec1)
	model.writer_train.close()
	model.writer_val.close()
	
	logging.info('Training complete')
	logging.info('Total training time:' + str(end_train - start_train))


if __name__ == '__main__':
	main()

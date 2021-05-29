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




np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()
print(Fore.YELLOW + "Number of GPUS available: ", gpu_count)

def main():
	args = parser.parse_args()

	path_exp = args.exp_path + args.modality + '/'

	#========== model init ========#

	print(Fore.CYAN + 'Initialising model......')
	model = initialise_trainer(args)

	#========== log files init ========#

	open_log_files(args)

	#========== tensorboard init ========#

	if args.tensorboard:
		writer_train = SummaryWriter(path_exp + '/tensorboard_train')  # for tensorboardX
		writer_val = SummaryWriter(path_exp + '/tensorboard_val')  # for tensorboardX
	
	#========== Data loading ========#
	
	print(Fore.CYAN + 'loading data......')

	if args.use_opencv:
		print("use opencv functions")

	source_loader, target_loader = get_train_data_loaders(args)
	
	to_validate  = args.val_source_data != "none" and args.val_target_data != "none" 
	if(to_validate):
		print(Fore.CYAN + 'Loading validation data......')
		source_loader_val, target_loader_val = get_val_data_loaders(args)

	#========== Callbacks and checkpoints ========#

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

	#========== Actual Training ========#
	
	trainer = Trainer(min_epochs=20, max_epochs=30, callbacks=[checkpoint_callback], gpus = gpu_count, accelerator='ddp')

	print(Fore.CYAN + 'start training......')	
	start_train = time.time()
	if(to_validate):
		trainer.fit(model, (source_loader, target_loader), (source_loader_val, target_loader_val))
	else:
		trainer.fit(model, (source_loader, target_loader))

	#========== Logging ========#
	end_train = time.time()
	print(Fore.CYAN + 'total training time:', end_train - start_train)

	write_log_files('total time: {:.3f} '.format(end_train - start_train), best_prec1)
	
	if args.tensorboard:
		writer_train.close()
		writer_val.close()

	print(Fore.CYAN + 'Training complete')


if __name__ == '__main__':
	main()

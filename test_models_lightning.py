import numpy as np
import json
from json import encoder

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from pytorch_lightning import Trainer

from colorama import init
from colorama import Fore, Back, Style

from utils.opts_test import parser
from utils.model_init import initialise_tester
from utils.data_loaders import get_test_data_loaders

from utils.logging import *

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
init(autoreset=True)

def main():
	args = parser.parse_args()

	#========== model init ========#

	log_info('Preparing the model......')
	verb_net, noun_net = initialise_tester(args)
	
	#========== Data loading ========#

	log_info('Loading data......')
	data_loader = get_test_data_loaders(args)
	log_info('Data loaded from: ' + args.test_target_data+".pkl")

	#========== Actual Testing ========#
	log_info('starting validation......')
	trainer = Trainer(gpus = torch.cuda.device_count())

	trainer.test(model = verb_net, test_dataloaders=data_loader, ckpt_path=args.weights, verbose = True)

	log_info('validation complete')



if __name__ == '__main__':
	main()

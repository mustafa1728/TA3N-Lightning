import argparse
import json
from json import encoder

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from pytorch_lightning import Trainer

from utils.model_init import initialise_tester
from utils.data_loaders import get_test_data_loaders
from config import get_cfg_defaults
from utils.logging import *

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def arg_parse():
	"""Parsing arguments"""
	parser = argparse.ArgumentParser(description="TA3N Domain Adaptation Testing")
	parser.add_argument("--cfg", required=True, help="path to config file", type=str)
	parser.add_argument("--gpus", default="0", help="gpu id(s) to use", type=str)
	parser.add_argument("--ckpt", default=None, help="pre-trained parameters for the model (ckpt files)", type=str)
	args = parser.parse_args()
	return args

def main():
	args = arg_parse()
	cfg = get_cfg_defaults()
	cfg.merge_from_file(args.cfg)
	cfg.freeze()

	#========== model init ========#

	log_info('Preparing the model......')
	verb_net, noun_net = initialise_tester(cfg)
	
	#========== Data loading ========#

	log_info('Loading data......')
	data_loader = get_test_data_loaders(cfg)
	log_info('Data loaded from: ' + cfg.TESTER.TEST_TARGET_DATA+".pkl")

	#========== Actual Testing ========#
	log_info('starting validation......')
	trainer = Trainer(gpus = args.gpus)

	if args.ckpt is None:
		ckpt_path = cfg.TESTER.WEIGHTS
	else:
		ckpt_path = args.ckpt
	trainer.test(model = verb_net, test_dataloaders=data_loader, ckpt_path=ckpt_path, verbose = cfg.TESTER.VERBOSE)
	if noun_net is not None:
		trainer.test(model = noun_net, test_dataloaders=data_loader, ckpt_path=ckpt_path, verbose = cfg.TESTER.VERBOSE)

	log_info('validation complete')



if __name__ == '__main__':
	main()

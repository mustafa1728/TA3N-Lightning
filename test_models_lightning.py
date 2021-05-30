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

import logging
logging.basicConfig(format='%(asctime)s  |  %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
init(autoreset=True)

def main():
	args = parser.parse_args()

	#========== model init ========#

	logging.info('Preparing the model......')
	verb_net, noun_net = initialise_tester(args)
	
	#========== Data loading ========#

	logging.info('Loading data......')
	data_loader = get_test_data_loaders(args)
	logging.info('Data loaded from: ' + args.test_target_data+".pkl")

	#========== Actual Testing ========#
	logging.info('starting validation......')
	trainer = Trainer()

	trainer.test(model = verb_net, test_dataloaders=data_loader, ckpt_path=args.weights, verbose = True)

	logging.info(('Testing Results: Prec@1 verb {top1_verb.avg:.3f}  Prec@1 noun {top1_noun.avg:.3f} Prec@1 action {top1_action.avg:.3f} Prec@5 verb {top5_verb.avg:.3f} Prec@5 noun {top5_noun.avg:.3f} Prec@5 action {top5_action.avg:.3f} Loss {loss.avg:.5f}'
			.format(top1_verb=verb_net.top1_verb_val, top1_noun=verb_net.top1_noun_val, top1_action=verb_net.top1_action_val, top5_verb=verb_net.top5_verb_val, top5_noun=verb_net.top5_noun_val, top5_action=verb_net.top5_action_val, loss=verb_net.losses_val)))

	logging.info('validation complete')



if __name__ == '__main__':
	main()

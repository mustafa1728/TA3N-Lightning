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


from opts import parser
from model_init import initialise_tester
from data_loaders import get_test_data_loaders



encoder.FLOAT_REPR = lambda o: format(o, '.3f')
init(autoreset=True)

def main():
	args = parser.parse_args()
	gpu_count = torch.cuda.device_count()
	

	criterion = torch.nn.CrossEntropyLoss()

	#========== model init ========#

	print(Fore.CYAN + 'preparing the model......')
	verb_net, noun_net = initialise_tester(args)
	
	#========== Data loading ========#

	print(Fore.CYAN + 'loading data......')
	data_loader = get_test_data_loaders(args)
	print('\n', Fore.CYAN + 'data loaded from: ', args.test_target_data+".pkl")

	#========== Actual Testing ========#
	print(Fore.CYAN + 'starting validation......')
	trainer = Trainer()

	trainer.test(model = verb_net, test_dataloaders=data_loader, ckpt_path=args.weights, verbose = True)

	print(('Testing Results: Prec@1 verb {top1_verb.avg:.3f}  Prec@1 noun {top1_noun.avg:.3f} Prec@1 action {top1_action.avg:.3f} Prec@5 verb {top5_verb.avg:.3f} Prec@5 noun {top5_noun.avg:.3f} Prec@5 action {top5_action.avg:.3f} Loss {loss.avg:.5f}'
			.format(top1_verb=verb_net.top1_verb_val, top1_noun=verb_net.top1_noun_val, top1_action=verb_net.top1_action_val, top5_verb=verb_net.top5_verb_val, top5_noun=verb_net.top5_noun_val, top5_action=verb_net.top5_action_val, loss=verb_net.losses_val)))

	print(Fore.CYAN + 'validation complete')



if __name__ == '__main__':
	main()

import torch
from model import TA3NTrainer
from torch import nn
import logging
from utils.model_init import set_hyperparameters
from config import get_cfg_defaults

logging.basicConfig(filename="test_trainer.txt",
                    filemode='a',
                    format='%(levelname)-6s | %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

def toString(x):
    return "mean: {} | std: {} | size: {}".format(torch.mean(x), torch.std(x), x.size())

class SmallTA3NTrainer(TA3NTrainer):
    def __init__(self, num_class=[97,300], baseline_type="video", frame_aggregation="trn-m", modality="All"):
        super(SmallTA3NTrainer, self).__init__(num_class, baseline_type, frame_aggregation, modality)
        cfg = get_cfg_defaults()
        set_hyperparameters(self, cfg)
        self.batch_size = [128, 128, 128]
        self.train_metric = "all"
        self.dis_DA = "DAN"
        self.use_target = "uSv"
        self.place_dis = ["Y", "Y", "N"]
        self.alpha = 0

        torch.manual_seed(100)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.softmax = nn.LogSoftmax()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_domain = torch.nn.CrossEntropyLoss()

    def forward_single(self, x):
        feats = []
        logging.debug("Input: "+toString(x))
        x = self.fc1(x)
        feats.append(x)
        logging.debug("fc1: "+toString(x))
        x = self.fc2(x)
        feats.append(x)
        logging.debug("fc2: "+toString(x))
        x = self.fc3(x)
        feats.append(x)
        logging.debug("fc3: "+toString(x))
        class_out = self.softmax(x)
        class_out = torch.sum(class_out, 1)
        adv_out = class_out
        logging.debug("softmax: "+toString(class_out))
        return feats, class_out, [adv_out, adv_out, adv_out]
    
    def forward(self, input_source, input_target, is_train=True, reverse=False):
        x_s, source_class, adv_out_s = self.forward_single(input_source)
        source_class = [source_class, source_class]
        x_t, target_class, adv_out_t = self.forward_single(input_target)
        target_class = [target_class, target_class]
        return x_s, source_class, source_class, adv_out_s, x_s, x_t, target_class, target_class, adv_out_t, x_t


    def test(self):
        input = torch.ones(128, 5, 1024)
        ones = torch.ones(128, dtype=torch.long)
        batch = ((input, [ones, ones], None), (input, [ones, ones], None))	
        task_loss, adv_loss, log_metrics = self.compute_loss(batch)
        logging.debug("task_loss: {}".format(task_loss))
        logging.debug("adv_loss: {}".format(adv_loss))
        for metric in log_metrics:
            logging.debug("metric: {} - {}".format(metric, log_metrics[metric]))


model = SmallTA3NTrainer()
model.test()

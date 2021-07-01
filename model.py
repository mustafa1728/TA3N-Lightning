import time
import numpy as np
from math import ceil

from torch import nn
from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from torch.nn.utils import clip_grad_norm_

from utils.logging import *
from utils.loss import *

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class TRNRelationModule(pl.LightningModule):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(TRNRelationModule, self).__init__()
        self.num_frames = num_frames
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = num_bottleneck
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck),
                nn.ReLU(),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

class TRNRelationModuleMultiScale(pl.LightningModule):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(TRNRelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        # self.num_class = num_class
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        )

            self.fc_fusion_scales += [fc_fusion]

        self.log('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_scale_1 = input[:, self.relations_scales[0][0] , :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1) # add one dimension for the later concatenation
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

            #for idx in idx_relations_randomsample:
            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = act_relation.unsqueeze(1)  # add one dimension for the later concatenation
                act_relation_all += act_relation

            act_all = torch.cat((act_all, act_relation_all), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


# definition of Gradient Reversal Layer
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None

# definition of Gradient Scaling Layer
class GradScale(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output * ctx.beta
		return grad_input, None

# definition of Temporal-ConvNet Layer
class TCL(pl.LightningModule):
	def __init__(self, conv_size, dim):
		super(TCL, self).__init__()

		self.conv2d = nn.Conv2d(dim, dim, kernel_size=(conv_size,1), padding=(conv_size//2,0))

		# initialization
		kaiming_normal_(self.conv2d.weight)

	def	forward(self, x):
		x = self.conv2d(x)

		return x

class TA3NTrainer(pl.LightningModule):
	def __init__(self, num_class, baseline_type, frame_aggregation, modality,
				train_segments=5, val_segments=25,
				base_model='resnet101', path_pretrained='', new_length=None,
				before_softmax=True,
				dropout_i=0.5, dropout_v=0.5, use_bn=None, ens_DA=None,
				crop_num=1, partial_bn=True, verbose=True, add_fc=1, fc_dim=1024,
				n_rnn=1, rnn_cell='LSTM', n_directions=1, n_ts=5,
				use_attn='TransAttn', n_attn=1, use_attn_frame=None,
				share_params='Y'):
		super().__init__()
		super(TA3NTrainer, self).__init__()
		self.num_class = num_class
		self.modality = modality
		self.train_segments = train_segments
		self.val_segments = val_segments
		self.baseline_type = baseline_type
		self.frame_aggregation = frame_aggregation
		self.reshape = True
		self.before_softmax = before_softmax
		self.dropout_rate_i = dropout_i
		self.dropout_rate_v = dropout_v
		self.use_bn = use_bn
		self.ens_DA = ens_DA
		self.crop_num = crop_num
		self.add_fc = add_fc
		self.fc_dim = fc_dim
		self.share_params = share_params
		self.verbose = verbose
		self.epochs = 30

		# RNN
		self.n_layers = n_rnn
		self.rnn_cell = rnn_cell
		self.n_directions = n_directions
		self.n_ts = n_ts # temporal segment

		# Attention
		self.use_attn = use_attn 
		self.n_attn = n_attn
		self.use_attn_frame = use_attn_frame



		if new_length is None:
			self.new_length = 1 if modality == "RGB" else 5
		else:
			self.new_length = new_length

		if verbose:
			log_info(("""
				Initializing TSN with base model: {}.
				TSN Configurations:
				input_modality:     {}
				num_segments:       {}
				new_length:         {}
				""".format(base_model, self.modality, self.train_segments, self.new_length)))

		self._prepare_DA(num_class, base_model, modality)

		if not self.before_softmax:
			self.softmax = nn.Softmax()

		self._enable_pbn = partial_bn
		if partial_bn:
			self.partialBN(True)
		self.no_partialbn = not partial_bn

		self.best_prec1 = 0


		#======= Setting up losses and the criterion ======#

		
		self.loss_c_current = 0
		self.loss_c_previous = 0
		self.save_attention = -1

		self.end = time.time()
		self.end_val = time.time()

		#======= For lightning to use custom optimizer ======#
		self.automatic_optimization  = True

		self.tensorboard = False

	def _prepare_DA(self, num_class, base_model, modality): # convert the model to DA framework
		if base_model == "TBN" and modality=="ALL":
			self.feature_dim = 3072
		elif base_model == "TBN":
			self.feature_dim = 1024
		else:
			model_test = getattr(torchvision.models, base_model)(True) # model_test is only used for getting the dim #
			self.feature_dim = model_test.fc.in_features

		std = 0.001
		feat_shared_dim = min(self.fc_dim, self.feature_dim) if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
		feat_frame_dim = feat_shared_dim

		self.relu = nn.ReLU(inplace=True)
		self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
		self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

		#------ frame-level layers (shared layers + source layers + domain layers) ------#
		if self.add_fc < 1:
			raise ValueError('add at least one fc layer')

		# 1. shared feature layers
		self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
		normal_(self.fc_feature_shared_source.weight, 0, std)
		constant_(self.fc_feature_shared_source.bias, 0)

		if self.add_fc > 1:
			self.fc_feature_shared_2_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_2_source.weight, 0, std)
			constant_(self.fc_feature_shared_2_source.bias, 0)

		if self.add_fc > 2:
			self.fc_feature_shared_3_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_3_source.weight, 0, std)
			constant_(self.fc_feature_shared_3_source.bias, 0)

		# 2. frame-level feature layers
		self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
		normal_(self.fc_feature_source.weight, 0, std)
		constant_(self.fc_feature_source.bias, 0)

		# 3. domain feature layers (frame-level)
		self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim)
		normal_(self.fc_feature_domain.weight, 0, std)
		constant_(self.fc_feature_domain.bias, 0)

		# 4. classifiers (frame-level)
		self.fc_classifier_source_verb = nn.Linear(feat_frame_dim, num_class[0])
		self.fc_classifier_source_noun = nn.Linear(feat_frame_dim, num_class[1])
		normal_(self.fc_classifier_source_verb.weight, 0, std)
		constant_(self.fc_classifier_source_verb.bias, 0)
		normal_(self.fc_classifier_source_noun.weight, 0, std)
		constant_(self.fc_classifier_source_noun.bias, 0)


		self.fc_classifier_domain = nn.Linear(feat_frame_dim, 2)
		normal_(self.fc_classifier_domain.weight, 0, std)
		constant_(self.fc_classifier_domain.bias, 0)

		if self.share_params == 'N':
			self.fc_feature_shared_target = nn.Linear(self.feature_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_target.weight, 0, std)
			constant_(self.fc_feature_shared_target.bias, 0)
			if self.add_fc > 1:
				self.fc_feature_shared_2_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_2_target.weight, 0, std)
				constant_(self.fc_feature_shared_2_target.bias, 0)
			if self.add_fc > 2:
				self.fc_feature_shared_3_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_3_target.weight, 0, std)
				constant_(self.fc_feature_shared_3_target.bias, 0)

			self.fc_feature_target = nn.Linear(feat_shared_dim, feat_frame_dim)
			normal_(self.fc_feature_target.weight, 0, std)
			constant_(self.fc_feature_target.bias, 0)

			self.fc_classifier_target_verb = nn.Linear(feat_frame_dim, num_class[0])
			normal_(self.fc_classifier_target_verb.weight, 0, std)
			constant_(self.fc_classifier_target_verb.bias, 0)
			self.fc_classifier_target_noun = nn.Linear(feat_frame_dim, num_class[1])
			normal_(self.fc_classifier_target_noun.weight, 0, std)
			constant_(self.fc_classifier_target_noun.bias, 0)


		# BN for the above layers
		if self.use_bn is not None:  # S & T: use AdaBN (ICLRW 2017) approach
			self.bn_shared_S = nn.BatchNorm1d(feat_shared_dim)  # BN for the shared layers
			self.bn_shared_T = nn.BatchNorm1d(feat_shared_dim)
			self.bn_source_S = nn.BatchNorm1d(feat_frame_dim)  # BN for the source feature layers
			self.bn_source_T = nn.BatchNorm1d(feat_frame_dim)

		#------ aggregate frame-based features (frame feature --> video feature) ------#
		if self.frame_aggregation == 'rnn': # 2. rnn
			self.hidden_dim = feat_frame_dim
			if self.rnn_cell == 'LSTM':
				self.rnn = nn.LSTM(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))
			elif self.rnn_cell == 'GRU':
				self.rnn = nn.GRU(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))

			# initialization
			for p in range(self.n_layers):
				kaiming_normal_(self.rnn.all_weights[p][0])
				kaiming_normal_(self.rnn.all_weights[p][1])

			self.bn_before_rnn = nn.BatchNorm2d(1)
			self.bn_after_rnn = nn.BatchNorm2d(1)

		elif self.frame_aggregation == 'trn': # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 512
			self.TRN = TRNRelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
		elif self.frame_aggregation == 'trn-m':  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 256
			self.TRN = TRNRelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)

		elif self.frame_aggregation == 'temconv': # 3. temconv

			self.tcl_3_1 = TCL(3, 1)
			self.tcl_5_1 = TCL(5, 1)
			self.bn_1_S = nn.BatchNorm1d(feat_frame_dim)
			self.bn_1_T = nn.BatchNorm1d(feat_frame_dim)

			self.tcl_3_2 = TCL(3, 1)
			self.tcl_5_2 = TCL(5, 2)
			self.bn_2_S = nn.BatchNorm1d(feat_frame_dim)
			self.bn_2_T = nn.BatchNorm1d(feat_frame_dim)

			self.conv_fusion = nn.Sequential(
				nn.Conv2d(2, 1, kernel_size=(1, 1), padding=(0, 0)),
				nn.ReLU(inplace=True),
			)

		# ------ video-level layers (source layers + domain layers) ------#
		if self.frame_aggregation == 'avgpool': # 1. avgpool
			feat_aggregated_dim = feat_shared_dim
		if 'trn' in self.frame_aggregation : # 4. trn
			feat_aggregated_dim = self.num_bottleneck
		elif self.frame_aggregation == 'rnn': # 2. rnn
			feat_aggregated_dim = self.hidden_dim
		elif self.frame_aggregation == 'temconv': # 3. temconv
			feat_aggregated_dim = feat_shared_dim

		feat_video_dim = feat_aggregated_dim

	
		# 2. domain feature layers (video-level)
		self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_domain_video.weight, 0, std)
		constant_(self.fc_feature_domain_video.bias, 0)

		# 3. classifiers (video-level)
		self.fc_classifier_video_verb_source = nn.Linear(feat_video_dim, num_class[0])
		normal_(self.fc_classifier_video_verb_source.weight, 0, std)
		constant_(self.fc_classifier_video_verb_source.bias, 0)

		self.fc_classifier_video_noun_source = nn.Linear(feat_video_dim, num_class[1])
		normal_(self.fc_classifier_video_noun_source.weight, 0, std)
		constant_(self.fc_classifier_video_noun_source.bias, 0)

		if self.ens_DA == 'MCD':
			self.fc_classifier_video_source_2 = nn.Linear(feat_video_dim, num_class) # second classifier for self-ensembling
			normal_(self.fc_classifier_video_source_2.weight, 0, std)
			constant_(self.fc_classifier_video_source_2.bias, 0)

		self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
		normal_(self.fc_classifier_domain_video.weight, 0, std)
		constant_(self.fc_classifier_domain_video.bias, 0)

		# domain classifier for TRN-M
		if self.frame_aggregation == 'trn-m':
			self.relation_domain_classifier_all = nn.ModuleList()
			for i in range(self.train_segments-1):
				relation_domain_classifier = nn.Sequential(
					nn.Linear(feat_aggregated_dim, feat_video_dim),
					nn.ReLU(),
					nn.Linear(feat_video_dim, 2)
				)
				self.relation_domain_classifier_all += [relation_domain_classifier]

		if self.share_params == 'N':
			self.fc_feature_video_target = nn.Linear(feat_aggregated_dim, feat_video_dim)
			normal_(self.fc_feature_video_target.weight, 0, std)
			constant_(self.fc_feature_video_target.bias, 0)
			self.fc_feature_video_target_2 = nn.Linear(feat_video_dim, feat_video_dim)
			normal_(self.fc_feature_video_target_2.weight, 0, std)
			constant_(self.fc_feature_video_target_2.bias, 0)

			self.fc_classifier_video_verb_target = nn.Linear(feat_video_dim, num_class)
			normal_(self.fc_classifier_video_verb_target.weight, 0, std)
			constant_(self.fc_classifier_video_verb_target.bias, 0)

			self.fc_classifier_video_noun_target = nn.Linear(feat_video_dim, num_class)
			normal_(self.fc_classifier_video_noun_target.weight, 0, std)
			constant_(self.fc_classifier_video_noun_target.bias, 0)

		# BN for the above layers
		if self.use_bn is not None:  # S & T: use AdaBN (ICLRW 2017) approach
			self.bn_source_video_S = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_T = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_2_S = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_2_T = nn.BatchNorm1d(feat_video_dim)

		self.alpha = torch.ones(1)
		
		if self.use_bn == 'AutoDIAL':
			self.alpha = nn.Parameter(self.alpha)

		# ------ attention mechanism ------#
		# conventional attention
		if self.use_attn == 'general':
			self.attn_layer = nn.Sequential(
				nn.Linear(feat_aggregated_dim, feat_aggregated_dim),
				nn.Tanh(),
				nn.Linear(feat_aggregated_dim, 1)
			)


	def train(self, mode=True):
		# not necessary in our setting
		"""
		Override the default train() to freeze the BN parameters
		:return:
		"""
		super(TA3NTrainer, self).train(mode)
		count = 0
		if self._enable_pbn:
			log_debug("Freezing BatchNorm2D except the first one.")
			for m in self.base_model.modules():
				if isinstance(m, nn.BatchNorm2d):
					count += 1
					if count >= (2 if self._enable_pbn else 1):
						m.eval()

						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False

	def partialBN(self, enable):
		self._enable_pbn = enable

	def get_trans_attn(self, pred_domain):
		softmax = nn.Softmax(dim=1)
		logsoftmax = nn.LogSoftmax(dim=1)
		entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
		weights = 1 - entropy

		return weights

	def get_general_attn(self, feat):
		num_segments = feat.size()[1]
		feat = feat.view(-1, feat.size()[-1]) # reshape features: 128x4x256 --> (128x4)x256
		weights = self.attn_layer(feat) # e.g. (128x4)x1
		weights = weights.view(-1, num_segments, weights.size()[-1]) # reshape attention weights: (128x4)x1 --> 128x4x1
		weights = F.softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

		return weights

	def get_attn_feat_frame(self, feat_fc, pred_domain): # not used for now
		if self.use_attn == 'TransAttn':
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, 1).repeat(1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 512)
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn

	def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
		if self.use_attn == 'TransAttn':
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn, weights_attn[:,:,0]

	def aggregate_frames(self, feat_fc, num_segments, pred_domain):
		feat_fc_video = None
		if self.frame_aggregation == 'rnn':
			# 2. RNN
			feat_fc_video = feat_fc.view((-1, num_segments) + feat_fc.size()[-1:])  # reshape for RNN

			# temporal segments and pooling
			len_ts = round(num_segments/self.n_ts)
			num_extra_f = len_ts*self.n_ts-num_segments
			if num_extra_f < 0: # can remove last frame-level features
				feat_fc_video = feat_fc_video[:, :len_ts * self.n_ts, :]  # make the temporal length can be divided by n_ts (16 x 25 x 512 --> 16 x 24 x 512)
			elif num_extra_f > 0: # need to repeat last frame-level features
				feat_fc_video = torch.cat((feat_fc_video, feat_fc_video[:,-1:,:].repeat(1,num_extra_f,1)), 1) # make the temporal length can be divided by n_ts (16 x 5 x 512 --> 16 x 6 x 512)

			feat_fc_video = feat_fc_video.view(
				(-1, self.n_ts, len_ts) + feat_fc_video.size()[2:])  # 16 x 6 x 512 --> 16 x 3 x 2 x 512
			feat_fc_video = nn.MaxPool2d(kernel_size=(len_ts, 1))(
				feat_fc_video)  # 16 x 3 x 2 x 512 --> 16 x 3 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(2)  # 16 x 3 x 1 x 512 --> 16 x 3 x 512

			hidden_temp = torch.zeros(self.n_layers * self.n_directions, feat_fc_video.size(0),
									  self.hidden_dim // self.n_directions)
			hidden_temp = hidden_temp.type_as(feat_fc)

			if self.rnn_cell == 'LSTM':
				hidden_init = (hidden_temp, hidden_temp)
			elif self.rnn_cell == 'GRU':
				hidden_init = hidden_temp

			self.rnn.flatten_parameters()
			feat_fc_video, hidden_final = self.rnn(feat_fc_video, hidden_init)  # e.g. 16 x 25 x 512

			# get the last feature vector
			feat_fc_video = feat_fc_video[:, -1, :]

		else:
			# 1. averaging
			feat_fc_video = feat_fc.view((-1, 1, num_segments) + feat_fc.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
			if self.use_attn == 'TransAttn': # get the attention weighting
				weights_attn = self.get_trans_attn(pred_domain)
				weights_attn = weights_attn.view(-1, 1, num_segments,1).repeat(1,1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
				feat_fc_video = (weights_attn+1) * feat_fc_video

			feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512

		return feat_fc_video

	def final_output(self, pred, pred_video, num_segments):
		if self.baseline_type == 'video':
			base_out = pred_video
		else:
			base_out = pred

		if not self.before_softmax:
			base_out = (self.softmax(base_out[0]), self.softmax(base_out[1]))
		output = base_out

		if self.baseline_type == 'tsn':
			if self.reshape:
				base_out = (base_out[0].view((-1, num_segments) + base_out[0].size()[1:]),
							base_out[1].view((-1, num_segments) + base_out[1].size()[1:])) # e.g. 16 x 3 x 12 (3 segments)
			output = (base_out[0].mean(1), base_out[1].mean(1)) # e.g. 16 x 12

		return output

	def domain_classifier_frame(self, feat, beta):
		feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
		feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
		feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
		pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)

		return pred_fc_domain_frame

	def domain_classifier_video(self, feat_video, beta):
		feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
		feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
		feat_fc_domain_video = self.relu(feat_fc_domain_video)
		pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

		return pred_fc_domain_video

	def domain_classifier_relation(self, feat_relation, beta):
		# 128x4x256 --> (128x4)x2
		pred_fc_domain_relation_video = None
		for i in range(len(self.relation_domain_classifier_all)):
			feat_relation_single = feat_relation[:,i,:].squeeze(1) # 128x1x256 --> 128x256
			feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta[0]) # the same beta for all relations (for now)

			pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
	
			if pred_fc_domain_relation_video is None:
				pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
			else:
				pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)
		
		pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)

		return pred_fc_domain_relation_video

	def domainAlign(self, input_S, input_T, is_train, name_layer, alpha, num_segments, dim):
		input_S = input_S.view((-1, dim, num_segments) + input_S.size()[-1:])  # reshape based on the segments (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
		input_T = input_T.view((-1, dim, num_segments) + input_T.size()[-1:])  # reshape based on the segments

		# clamp alpha
		alpha = max(alpha,0.5)

		# rearange source and target data
		num_S_1 = int(round(input_S.size(0) * alpha))
		num_S_2 = input_S.size(0) - num_S_1
		num_T_1 = int(round(input_T.size(0) * alpha))
		num_T_2 = input_T.size(0) - num_T_1

		if is_train and num_S_2 > 0 and num_T_2 > 0:
			input_source = torch.cat((input_S[:num_S_1], input_T[-num_T_2:]), 0)
			input_target = torch.cat((input_T[:num_T_1], input_S[-num_S_2:]), 0)
		else:
			input_source = input_S
			input_target = input_T

		# adaptive BN
		input_source = input_source.view((-1, ) + input_source.size()[-1:]) # reshape to feed BN (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
		input_target = input_target.view((-1, ) + input_target.size()[-1:])

		if name_layer == 'shared':
			input_source_bn = self.bn_shared_S(input_source)
			input_target_bn = self.bn_shared_T(input_target)
		elif 'trn' in name_layer:
			input_source_bn = self.bn_trn_S(input_source)
			input_target_bn = self.bn_trn_T(input_target)
		elif name_layer == 'temconv_1':
			input_source_bn = self.bn_1_S(input_source)
			input_target_bn = self.bn_1_T(input_target)
		elif name_layer == 'temconv_2':
			input_source_bn = self.bn_2_S(input_source)
			input_target_bn = self.bn_2_T(input_target)

		input_source_bn = input_source_bn.view((-1, dim, num_segments) + input_source_bn.size()[-1:])  # reshape back (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
		input_target_bn = input_target_bn.view((-1, dim, num_segments) + input_target_bn.size()[-1:])  #

		# rearange back to the original order of source and target data (since target may be unlabeled)
		if is_train and num_S_2 > 0 and num_T_2 > 0:
			input_source_bn = torch.cat((input_source_bn[:num_S_1], input_target_bn[-num_S_2:]), 0)
			input_target_bn = torch.cat((input_target_bn[:num_T_1], input_source_bn[-num_T_2:]), 0)

		# reshape for frame-level features
		if name_layer == 'shared' or name_layer == 'trn_sum':
			input_source_bn = input_source_bn.view((-1,) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
			input_target_bn = input_target_bn.view((-1,) + input_target_bn.size()[-1:])
		elif name_layer == 'trn':
			input_source_bn = input_source_bn.view((-1, num_segments) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
			input_target_bn = input_target_bn.view((-1, num_segments) + input_target_bn.size()[-1:])

		return input_source_bn, input_target_bn

	def forward(self, input_source, input_target, is_train=True, reverse=True):
		batch_source = input_source.size()[0]
		batch_target = input_target.size()[0]
		num_segments = self.train_segments if is_train else self.val_segments
		# sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
		sample_len = self.new_length
		feat_all_source = []
		feat_all_target = []
		pred_domain_all_source = []
		pred_domain_all_target = []

		# input_data is a list of tensors --> need to do pre-processing
		feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 256 x 25 x 2048 --> 6400 x 2048
		feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048

		#=== shared layers ===#
		# need to separate BN for source & target ==> otherwise easy to overfit to source data
		if self.add_fc < 1:
			raise ValueError( 'not enough fc layer')

		feat_fc_source = self.fc_feature_shared_source(feat_base_source)
		feat_fc_target = self.fc_feature_shared_target(feat_base_target) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target)

		# adaptive BN
		if self.use_bn is not None:
			feat_fc_source, feat_fc_target = self.domainAlign(feat_fc_source, feat_fc_target, is_train, 'shared', self.alpha, num_segments, 1)

		feat_fc_source = self.relu(feat_fc_source)
		feat_fc_target = self.relu(feat_fc_target)
		feat_fc_source = self.dropout_i(feat_fc_source)
		feat_fc_target = self.dropout_i(feat_fc_target)

		# feat_fc = self.dropout_i(feat_fc)
		feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
		feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

		if self.add_fc > 1: 
			feat_fc_source = self.fc_feature_shared_2_source(feat_fc_source)
			feat_fc_target = self.fc_feature_shared_2_target(feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_2_source(feat_fc_target)

			feat_fc_source = self.relu(feat_fc_source)
			feat_fc_target = self.relu(feat_fc_target)
			feat_fc_source = self.dropout_i(feat_fc_source)
			feat_fc_target = self.dropout_i(feat_fc_target)

			feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

		if self.add_fc > 2: 
			feat_fc_source = self.fc_feature_shared_3_source(feat_fc_source)
			feat_fc_target = self.fc_feature_shared_3_target(feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_3_source(feat_fc_target)

			feat_fc_source = self.relu(feat_fc_source)
			feat_fc_target = self.relu(feat_fc_target)
			feat_fc_source = self.dropout_i(feat_fc_source)
			feat_fc_target = self.dropout_i(feat_fc_target)

			feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

		# === adversarial branch (frame-level) ===#
		pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, self.beta)
		pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, self.beta)

		pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

		if self.use_attn_frame is not None: # attend the frame-level features only
			feat_fc_source = self.get_attn_feat_frame(feat_fc_source, pred_fc_domain_frame_source)
			feat_fc_target = self.get_attn_feat_frame(feat_fc_target, pred_fc_domain_frame_target)

		#=== source layers (frame-level) ===#

		pred_fc_source = (self.fc_classifier_source_verb(feat_fc_source), self.fc_classifier_source_noun(feat_fc_source))
		pred_fc_target = (self.fc_classifier_target_verb(feat_fc_target) if self.share_params == 'N' else self.fc_classifier_source_verb(feat_fc_target),
						  self.fc_classifier_target_noun(feat_fc_target) if self.share_params == 'N' else self.fc_classifier_source_noun(feat_fc_target))
		if self.baseline_type == 'frame':
			feat_all_source.append(pred_fc_source[0].view((batch_source, num_segments) + pred_fc_source[0].size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(pred_fc_target[0].view((batch_target, num_segments) + pred_fc_target[0].size()[-1:]))

		### aggregate the frame-based features to video-based features ###
		if self.frame_aggregation == 'avgpool' or self.frame_aggregation == 'rnn':
			feat_fc_video_source = self.aggregate_frames(feat_fc_source, num_segments, pred_fc_domain_frame_source)
			feat_fc_video_target = self.aggregate_frames(feat_fc_target, num_segments, pred_fc_domain_frame_target)

			attn_relation_source = feat_fc_video_source[:,0] # assign random tensors to attention values to avoid runtime error
			attn_relation_target = feat_fc_video_target[:,0] # assign random tensors to attention values to avoid runtime error

		elif 'trn' in self.frame_aggregation:
			feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
			feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

			feat_fc_video_relation_source = self.TRN(feat_fc_video_source) # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
			feat_fc_video_relation_target = self.TRN(feat_fc_video_target)

			# adversarial branch
			pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, self.beta)
			pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, self.beta)

			# transferable attention
			if self.use_attn is not None: # get the attention weighting
				feat_fc_video_relation_source, attn_relation_source = self.get_attn_feat_relation(feat_fc_video_relation_source, pred_fc_domain_video_relation_source, num_segments)
				feat_fc_video_relation_target, attn_relation_target = self.get_attn_feat_relation(feat_fc_video_relation_target, pred_fc_domain_video_relation_target, num_segments)
			else:
				attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
				attn_relation_target = feat_fc_video_relation_target[:,:,0] # assign random tensors to attention values to avoid runtime error

			# sum up relation features (ignore 1-relation)
			feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
			feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)

		elif self.frame_aggregation == 'temconv': # DA operation inside temconv
			feat_fc_video_source = feat_fc_source.view((-1, 1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments
			feat_fc_video_target = feat_fc_target.view((-1, 1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments

			# 1st TCL
			feat_fc_video_source_3_1 = self.tcl_3_1(feat_fc_video_source)
			feat_fc_video_target_3_1 = self.tcl_3_1(feat_fc_video_target)

			if self.use_bn is not None:
				feat_fc_video_source_3_1, feat_fc_video_target_3_1 = self.domainAlign(feat_fc_video_source_3_1, feat_fc_video_target_3_1, is_train, 'temconv_1', self.alpha, num_segments, 1)

			feat_fc_video_source = self.relu(feat_fc_video_source_3_1)  # 16 x 1 x 5 x 512
			feat_fc_video_target = self.relu(feat_fc_video_target_3_1)  # 16 x 1 x 5 x 512

			feat_fc_video_source = nn.AvgPool2d(kernel_size=(num_segments, 1))(feat_fc_video_source)  # 16 x 4 x 1 x 512
			feat_fc_video_target = nn.AvgPool2d(kernel_size=(num_segments, 1))(feat_fc_video_target)  # 16 x 4 x 1 x 512

			feat_fc_video_source = feat_fc_video_source.squeeze(1).squeeze(1)  # e.g. 16 x 512
			feat_fc_video_target = feat_fc_video_target.squeeze(1).squeeze(1)  # e.g. 16 x 512

		if self.baseline_type == 'video':
			feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
			feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))

		#=== source layers (video-level) ===#
		feat_fc_video_source = self.dropout_v(feat_fc_video_source)
		feat_fc_video_target = self.dropout_v(feat_fc_video_target)

		if reverse:
			feat_fc_video_source = GradReverse.apply(feat_fc_video_source, self.mu)
			feat_fc_video_target = GradReverse.apply(feat_fc_video_target, self.mu)

		pred_fc_video_source = (self.fc_classifier_video_verb_source(feat_fc_video_source), self.fc_classifier_video_noun_source(feat_fc_video_source))
		pred_fc_video_target = (self.fc_classifier_video_verb_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_verb_source(feat_fc_video_target),
									 self.fc_classifier_video_noun_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_noun_source(feat_fc_video_target))

		if self.baseline_type == 'video': # only store the prediction from classifier 1 (for now)
			feat_all_source.append(pred_fc_video_source[0].view((batch_source,) + pred_fc_video_source[0].size()[-1:]))
			feat_all_target.append(pred_fc_video_target[0].view((batch_target,) + pred_fc_video_target[0].size()[-1:]))
			feat_all_source.append(pred_fc_video_source[1].view((batch_source,) + pred_fc_video_source[1].size()[-1:]))
			feat_all_target.append(pred_fc_video_target[1].view((batch_target,) + pred_fc_video_target[1].size()[-1:]))

		#=== adversarial branch (video-level) ===#
		pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, self.beta)
		pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, self.beta)

		pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

		# video relation-based discriminator
		if self.frame_aggregation == 'trn-m':
			num_relation = feat_fc_video_relation_source.size()[1]
			pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
			pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))
		else:
			pred_domain_all_source.append(pred_fc_domain_video_source) # if not trn-m, add dummy tensors for relation features
			pred_domain_all_target.append(pred_fc_domain_video_target)

		#=== final output ===#
		output_source = self.final_output(pred_fc_source, pred_fc_video_source, num_segments) # select output from frame or video prediction
		output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

		output_source_2 = output_source
		output_target_2 = output_target

		if self.ens_DA == 'MCD':
			pred_fc_video_source_2 = self.fc_classifier_video_source_2(feat_fc_video_source)
			pred_fc_video_target_2 = self.fc_classifier_video_target_2(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source_2(feat_fc_video_target)
			output_source_2 = self.final_output(pred_fc_source, pred_fc_video_source_2, num_segments)
			output_target_2 = self.final_output(pred_fc_target, pred_fc_video_target_2, num_segments)

		return attn_relation_source, output_source, output_source_2, pred_domain_all_source[::-1], feat_all_source[::-1], attn_relation_target, output_target, output_target_2, pred_domain_all_target[::-1], feat_all_target[::-1] # reverse the order of feature list due to some multi-gpu issues


	
	def configure_optimizers(self):
		"""
		Automatically called by lightning to get the optimizer for training
		Returns:
			optimizer: a pytorch optimizer 
		"""

		if self.optimizerName == 'SGD':
			log_info('using SGD')
			optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
		elif self.optimizerName == 'Adam':
			log_info('using Adam')
			optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
		else:
			log_error('optimizer not support or specified!!!')
		
		return optimizer
		

	def adjust_learning_rate_loss(self, optimizer, decay, stat_current, stat_previous, op):
		ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y), '>=': (lambda x, y: x >= y), '<=': (lambda x, y: x <= y)}
		if ops[op](stat_current, stat_previous):
			for param_group in optimizer.param_groups:
				param_group['lr'] /= decay
	
	def get_parameters_watch_list(self):
		"""
		Update this list for parameters to watch while training (ie log with MLFlow)
		"""
		return {
			"alpha": self.alpha,
			"beta": self.beta[0],
			"mu": self.mu,
			"last_epoch": self.current_epoch,
		}

	def _update_batch_epoch_factors(self, batch_id):
		self._init_epochs = 0
		if self.current_epoch >= self._init_epochs:
			delta_epoch = self.current_epoch - self._init_epochs
			p = (batch_id + delta_epoch * self.batch_size[0]) / (
				self.epochs * self.batch_size[0]
			)
			beta_dann = 2. / (1. + np.exp(-1.0 * p)) - 1
			self._grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1

			self.beta = [beta_dann if self.beta[i] < 0 else self.beta[i] for i in range(len(self.beta))] # replace the default beta if value < 0
			if self.dann_warmup:
				beta_new = [beta_dann*self.beta[i] for i in range(len(self.beta))]
			else:
				beta_new = self.beta
			self.beta = beta_new

			if self.lr_adaptive == 'dann':
				for param_group in self.optimizers().param_groups:
					param_group['lr'] /= p 

		self._adapt_lambda = False
		self.lamb_da = 1
		if self._adapt_lambda:
			self.lamb_da = self._init_lambda * self._grow_fact

	def compute_loss(self, batch, split_name = "V"):
		((source_data, source_label, _), (target_data, target_label, _)) = batch	

		source_size_ori = source_data.size()  # original shape
		target_size_ori = target_data.size()  # original shape
		batch_source_ori = source_size_ori[0]
		batch_target_ori = target_size_ori[0]
		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_source_ori < self.batch_size[0]:
			source_data_dummy = torch.zeros(self.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
			source_data_dummy = source_data_dummy.type_as(source_data)
			source_data = torch.cat((source_data, source_data_dummy))
		if batch_target_ori < self.batch_size[1]:
			target_data_dummy = torch.zeros(self.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
			target_data_dummy = target_data_dummy.type_as(target_data)
			target_data = torch.cat((target_data, target_data_dummy))


		source_label_verb = source_label[0] # pytorch 0.4.X
		source_label_noun = source_label[1]  # pytorch 0.4.X

		target_label_verb = target_label[0] # pytorch 0.4.X
		target_label_noun = target_label[1] # pytorch 0.4.X

		if self.baseline_type == 'frame':
			source_label_verb_frame = source_label_verb.unsqueeze(1).repeat(1, self.train_segments).view(-1) # expand the size for all the frames
			source_label_noun_frame = source_label_noun.unsqueeze(1).repeat(1, self.train_segments).view(-1) # expand the size for all the frames
			target_label_verb_frame = target_label_verb.unsqueeze(1).repeat(1, self.train_segments).view(-1)
			target_label_noun_frame = target_label_noun.unsqueeze(1).repeat(1, self.train_segments).view(-1)

		label_source_verb = source_label_verb_frame if self.baseline_type == 'frame' else source_label_verb  # determine the label for calculating the loss function
		label_target_verb = target_label_verb_frame if self.baseline_type == 'frame' else target_label_verb

		label_source_noun = source_label_noun_frame if self.baseline_type == 'frame' else source_label_noun  # determine the label for calculating the loss function
		label_target_noun = target_label_noun_frame if self.baseline_type == 'frame' else target_label_noun


		attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = self(source_data, target_data, is_train=True, reverse=False)

		attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_source_ori)
		attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)


		out_verb = out_source[0]
		out_noun = out_source[1]
		label_verb = label_source_verb
		label_noun = label_source_noun

		loss_verb = self.criterion(out_verb, label_verb)
		loss_noun = self.criterion(out_noun, label_noun)
		if self.train_metric == "all":
			loss_classification = 0.5*(loss_verb+loss_noun)
		elif self.train_metric == "noun":
			loss_classification = loss_noun# 0.5*(loss_verb+loss_noun)
		elif self.train_metric == "verb":
			loss_classification = loss_verb  # 0.5*(loss_verb+loss_noun)
		else:
			raise Exception("invalid metric to train")

		loss = loss_classification

		# 2. calculate the loss for DA
		# (I) discrepancy-based approach: discrepancy loss
		if self.dis_DA is not None and self.use_target is not None:
			loss_discrepancy = 0

			kernel_muls = [2.0]*2
			kernel_nums = [2, 5]
			fix_sigma_list = [None]*2

			if self.dis_DA == 'JAN':
				# ignore the features from shared layers
				feat_source_sel = feat_source[:-self.add_fc]
				feat_target_sel = feat_target[:-self.add_fc]

				size_loss = min(feat_source_sel[0].size(0), feat_target_sel[0].size(0))  # choose the smaller number
				feat_source_sel = [feat[:size_loss] for feat in feat_source_sel]
				feat_target_sel = [feat[:size_loss] for feat in feat_target_sel]

				loss_discrepancy += JAN(feat_source_sel, feat_target_sel, kernel_muls=kernel_muls, kernel_nums=kernel_nums, fix_sigma_list=fix_sigma_list, ver=2)

			else:
				# extend the parameter list for shared layers
				kernel_muls.extend([kernel_muls[-1]]*self.add_fc)
				kernel_nums.extend([kernel_nums[-1]]*self.add_fc)
				fix_sigma_list.extend([fix_sigma_list[-1]]*self.add_fc)

				for l in range(0, self.add_fc + 2):  # loss from all the features (+2 because of frame-aggregation layer + final fc layer)
					if self.place_dis[l] == 'Y':
						# select the data for calculating the loss (make sure source # == target #)
						size_loss = min(feat_source[l].size(0), feat_target[l].size(0)) # choose the smaller number
						# select
						feat_source_sel = feat_source[l][:size_loss]
						feat_target_sel = feat_target[l][:size_loss]

						# break into multiple batches to avoid "out of memory" issue
						size_batch = min(256,feat_source_sel.size(0))
						feat_source_sel = feat_source_sel.view((-1,size_batch) + feat_source_sel.size()[1:])
						feat_target_sel = feat_target_sel.view((-1,size_batch) + feat_target_sel.size()[1:])

						if self.dis_DA == 'CORAL':
							losses_coral = [CORAL(feat_source_sel[t], feat_target_sel[t]) for t in range(feat_source_sel.size(0))]
							loss_coral = sum(losses_coral)/len(losses_coral)
							loss_discrepancy += loss_coral
						elif self.dis_DA == 'DAN':
							losses_mmd = [mmd_rbf(feat_source_sel[t], feat_target_sel[t], kernel_mul=kernel_muls[l], kernel_num=kernel_nums[l], fix_sigma=fix_sigma_list[l], ver=2) for t in range(feat_source_sel.size(0))]
							loss_mmd = sum(losses_mmd) / len(losses_mmd)

							loss_discrepancy += loss_mmd
						else:
							raise NameError('not in dis_DA!!!')

			self.alpha = self.alpha.type_as(loss_discrepancy)
			loss += self.alpha * loss_discrepancy

		# (II) adversarial discriminative model: adversarial loss
		if self.adv_DA is not None and self.use_target is not None:
			loss_adversarial = 0
			pred_domain_all = []
			pred_domain_target_all = []

			for l in range(len(self.place_adv)):
				if self.place_adv[l] == 'Y':

					# reshape the features (e.g. 128x5x2 --> 640x2)
					pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
					pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

					# prepare domain labels
					source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
					source_domain_label = source_domain_label.type_as(source_data)
					target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
					source_domain_label = source_domain_label.type_as(source_domain_label)
					source_domain_label = source_domain_label.type_as(target_domain_label)
					domain_label = torch.cat((source_domain_label,target_domain_label),0)

					domain_label = domain_label

					pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single),0)
					pred_domain_all.append(pred_domain)
					pred_domain_target_all.append(pred_domain_target_single)

					if self.pred_normalize == 'Y':  # use the uncertainly method (in construction......)
						pred_domain = pred_domain / pred_domain.var().log()
					loss_adversarial_single = self.criterion_domain(pred_domain, domain_label.type_as(pred_domain).long())

					loss_adversarial += loss_adversarial_single

			loss += loss_adversarial

		# (III) other loss
		# 1. entropy loss for target data
		if self.add_loss_DA == 'target_entropy' and self.use_target is not None:
			loss_entropy_verb = cross_entropy_soft(out_target[0])
			loss_entropy_noun = cross_entropy_soft(out_target[1])

			if self.train_metric == "all":
				loss += self.gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)
			elif self.train_metric == "noun":
				loss += self.gamma * loss_entropy_noun
			elif self.train_metric == "verb":
				loss += self.gamma * loss_entropy_verb
			else:
				raise Exception("invalid metric to train")
			#loss += gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)

		# 3. attentive entropy loss
		if self.add_loss_DA == 'attentive_entropy' and self.use_attn is not None and self.use_target is not None:
			loss_entropy_verb = attentive_entropy(torch.cat((out_verb, out_target[0]),0), pred_domain_all[1])
			loss_entropy_noun = attentive_entropy(torch.cat((out_noun, out_target[1]), 0), pred_domain_all[1])

			if self.train_metric == "all":
				loss += self.gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)
			elif self.train_metric == "noun":
				loss += self.gamma * loss_entropy_noun
			elif self.train_metric == "verb":
				loss += self.gamma * loss_entropy_verb
			else:
				raise Exception("invalid metric to train")
			#loss += gamma * 0.5*(loss_entropy_verb + loss_entropy_noun)
		# measure accuracy and record loss
		pred_verb = out_verb
		prec1_verb, prec5_verb = self.accuracy(pred_verb.data, label_verb, topk=(1, 5))
		pred_noun = out_noun
		prec1_noun, prec5_noun = self.accuracy(pred_noun.data, label_noun, topk=(1, 5))
		prec1_action, prec5_action = self.multitask_accuracy((pred_verb.data, pred_noun.data), (label_verb, label_noun), topk=(1, 5))
		prec1_action, prec5_action = torch.tensor(prec1_action), torch.tensor(prec5_action)

		# measure elapsed time
		batch_time  = time.time() - self.end
		self.end = time.time()

		# Pred normalise 
		if self.pred_normalize == 'Y': # use the uncertainly method (in contruction...)
			out_source = out_source / out_source.var().log()
			out_target = out_target / out_target.var().log()


		#======= return log_metrics ======#

		log_metrics = {'Loss Total': loss, 'Prec@1 Verb': prec1_verb, 'Prec@5 Verb': prec5_verb, 'Prec@1 Noun': prec1_noun, 'Prec@5 Noun': prec5_noun, 'Prec@1 Action': prec1_action, 'Prec@5 Action': prec5_action}


		return loss_classification, loss_adversarial, log_metrics

	


	def training_step(self, train_batch, batch_idx):
		"""
		Automatically called by lightning while training a single batch
		Args:
			train_batch: an item of the dataloader(s) passed with the trainer
			batch_idx: the batch index (which batch is currently being trained)
		Returns:
			The loss(es) caluclated after performing an optimiser step
		"""
		self._update_batch_epoch_factors(batch_idx)

		task_loss, adv_loss, log_metrics = self.compute_loss(train_batch, split_name="T")
		if self.current_epoch < self._init_epochs:
			# init phase doesn't use few-shot learning
			# ad-hoc decision but makes models more comparable between each other
			loss = task_loss
		else:
			loss = task_loss + self.lamb_da * adv_loss

		log_metrics = get_aggregated_metrics_from_dict(log_metrics)
		log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
		log_metrics["T_total_loss"] = loss
		log_metrics["T_adv_loss"] = adv_loss
		log_metrics["T_task_loss"] = task_loss

		for key in log_metrics:
			self.log(key, log_metrics[key])

		return {
		"loss": log_metrics['Loss Total'],  # required, for backward pass
		# "progress_bar": {"class_loss": task_loss},
		# "log": log_metrics,
		}

	def accuracy(self, output, target, topk=(1,)):
		"""Computes the precision@k for the specified values of k"""
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res
	
	def multitask_accuracy(self, outputs, labels, topk=(1,)):
		"""
		Args:
			outputs: tuple(torch.FloatTensor), each tensor should be of shape
				[batch_size, class_count], class_count can vary on a per task basis, i.e.
				outputs[i].shape[1] can be different to outputs[j].shape[j].
			labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
			topk: tuple(int), compute accuracy at top-k for the values of k specified
				in this parameter.
		Returns:
			tuple(float), same length at topk with the corresponding accuracy@k in.
		"""
		max_k = int(np.max(topk))
		task_count = len(outputs)
		batch_size = labels[0].size(0)
		all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
		all_correct = all_correct.type_as(labels[0])

		
		for output, label in zip(outputs, labels):
			_, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
			# Flip batch_size, class_count as .view doesn't work on non-contiguous
			max_k_idx = max_k_idx.t()
			correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
			all_correct.add_(correct_for_task)

		accuracies = []
		for k in topk:
			all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
			accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
			accuracies.append(accuracy_at_k)
		return tuple(accuracies)
		
	def validation_step(self, batch, batch_idx):
		"""
		Automatically called by lightning while validating a single batch
		Args:
			batch: an item of the dataloader(s) passed with the trainer
			batch_idx: the batch index (which batch is currently being validated)
			dataset_idx: in case of multiple dataloaders, this indicates which dataloader is being used
		Returns:
			The loss(es) caluclated 
		"""

		(val_data, val_label, _) = batch
		i = batch_idx

		val_size_ori = val_data.size()  # original shape
		batch_val_ori = val_size_ori[0]

		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_val_ori < self.batch_size[0]:
			val_data_dummy = torch.zeros(self.batch_size[0] - batch_val_ori, val_size_ori[1], val_size_ori[2])
			val_data_dummy = val_data_dummy.type_as(val_data)
			val_data = torch.cat((val_data, val_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		gpu_count = 1
		if val_data.size(0) % gpu_count != 0:
			val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
			val_data_dummy = val_data_dummy.type_as(val_data)
			val_data = torch.cat((val_data, val_data_dummy))

		val_label_verb = val_label[0]
		val_label_noun = val_label[1]
		with torch.no_grad():

			if self.baseline_type == 'frame':
				val_label_verb_frame = val_label_verb.unsqueeze(1).repeat(1,self.val_segments).view(-1) # expand the size for all the frames
				val_label_noun_frame = val_label_noun.unsqueeze(1).repeat(1, self.val_segments).view(-1)  # expand the size for all the frames

			# compute output
			_, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = self(val_data, val_data, is_train=False, reverse=False)

			# ignore dummy tensors
			attn_val, out_val, out_val_2, pred_domain_val, feat_val = removeDummy(attn_val, out_val, out_val_2, pred_domain_val, feat_val, batch_val_ori)

			# measure accuracy and record loss
			label_verb = val_label_verb_frame if self.baseline_type == 'frame' else val_label_verb
			label_noun = val_label_noun_frame if self.baseline_type == 'frame' else val_label_noun

			# store the embedding
			# if self.tensorboard:
			# 	self.feat_val_display = feat_val[1] if self.current_epoch == 0 else torch.cat((self.feat_val_display, feat_val[1]), 0)
			# 	self.label_val_verb_display = label_verb if self.current_epoch == 0 else torch.cat((self.label_val_verb_display, label_verb), 0)
			# 	self.label_val_noun_display = label_noun if self.current_epoch == 0 else torch.cat((self.label_val_noun_display, label_noun), 0)

			pred_verb = out_val[0]
			pred_noun = out_val[1]

			if self.baseline_type == 'tsn':
				pred_verb = pred_verb.view(val_label.size(0), -1, self.num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)
				pred_noun = pred_noun.view(val_label.size(0), -1, self.num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)

			loss_verb = self.criterion(pred_verb, label_verb)
			loss_noun = self.criterion(pred_noun, label_noun)


			loss = 0.5 * (loss_verb + loss_noun)
		
			prec1_verb, prec5_verb = self.accuracy(pred_verb.data, label_verb, topk=(1, 5))
			prec1_noun, prec5_noun = self.accuracy(pred_noun.data, label_noun, topk=(1, 5))
			prec1_action, prec5_action = self.multitask_accuracy((pred_verb.data, pred_noun.data), (label_verb, label_noun),
															topk=(1, 5))

		# measure elapsed time
		batch_time = time.time() - self.end_val
		self.end_val = time.time()

		#======= return dictionary and loggings ======#
		
		result_dict = {'batch_time': batch_time, 'loss': loss.item(), 'top1_verb': prec1_verb.item(), 'top5_verb': prec5_verb.item(), 'top1_noun': prec1_noun.item(), 'top5_noun': prec5_noun.item(), 'top1_action': prec1_action, 'top5_action': prec5_action}
	
		self.log("Prec@1 Verb", result_dict["top1_verb"], prog_bar=True)
		self.log("Prec@1 Noun", result_dict["top1_noun"], prog_bar=True)
		self.log("Prec@1 Action", result_dict["top1_action"], prog_bar=True)
		self.log("Prec@5 Verb", result_dict["top5_verb"], prog_bar=True)
		self.log("Prec@5 Noun", result_dict["top5_noun"], prog_bar=True)
		self.log("Prec@5 Action", result_dict["top5_action"], prog_bar=True)
		self.log("Loss total", result_dict["loss"], prog_bar=True)

		return result_dict	
		
	def validation_epoch_end(self, validation_step_outputs):
		"""
		Automatically called by lightning after validation ends for an epoch
		Args:
			validation_step_outputs: list of values returned by the validation_step after each batch
		"""

		self.end_val = time.time()
		# evaluate on validation set

		if self.labels_available:

			self.losses_val = 0
			self.prec1_val = 0
			self.prec1_verb_val = 0
			self.prec1_noun_val = 0
			self.prec5_val = 0
			self.prec5_verb_val = 0
			self.prec5_noun_val = 0

			count = 0
			for dict in validation_step_outputs:
				count+=1
				self.losses_val += dict["loss"]
				self.prec1_val += dict["top1_action"]
				self.prec1_verb_val += dict["top1_verb"]
				self.prec1_noun_val += dict["top1_noun"]
				self.prec5_val += dict["top5_action"]
				self.prec5_verb_val += dict["top5_verb"]
				self.prec5_noun_val += dict["top5_noun"]

			if not (count == 0):
				self.losses_val /= count
				self.prec1_val /= count
				self.prec1_verb_val /= count
				self.prec1_noun_val /= count
				self.prec5_val /= count
				self.prec5_verb_val /= count
				self.prec5_noun_val /= count

			# remember best prec@1 and save checkpoint
			if self.train_metric == "all":
				prec1 = self.prec1_val
			elif self.train_metric == "noun":
				prec1 = self.prec1_noun_val
			elif self.train_metric == "verb":
				prec1 = self.prec1_verb_val
			else:
				raise Exception("invalid metric to train")
				
			is_best = prec1 > self.best_prec1
			if is_best:
				

				line_update = ' ==> updating the best accuracy' if is_best else ''
				line_best = "Best score {} vs current score {}".format(self.best_prec1, prec1) + line_update
				log_info( line_best)
				# val_short_file.write('%.3f\n' % prec1)

			self.best_prec1 = max(prec1, self.best_prec1)

	def test_step(self, batch, batch_idx):
		"""
		Automatically called by lightning while testing/infering on a batch
		Args:
			batch: an item of the dataloader(s) passed with the trainer
			batch_idx: the batch index (which batch is currently being evaulated)
		Returns:
			The loss(es) caluclated 
		"""

		return self.validation_step(batch, batch_idx)

def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
	attn = attn[:batch_size]
	if isinstance(out_1, (list, tuple)):
		out_1 = (out_1[0][:batch_size], out_1[1][:batch_size])
	else:
		out_1 = out_1[:batch_size]
	out_2 = out_2[:batch_size]
	pred_domain = [pred[:batch_size] for pred in pred_domain]
	feat = [f[:batch_size] for f in feat]

	return attn, out_1, out_2, pred_domain, feat

# will be imported directly later
def get_aggregated_metrics_from_dict(input_metric_dict):
	"""Get a dictionary of the mean metric values (to log) from a dictionary of metric values"""
	metric_dict = {}
	for metric_name, metric_value in input_metric_dict.items():
		metric_dim = len(metric_value.shape)
		if metric_dim == 0:
			metric_dict[metric_name] = metric_value
		else:
			metric_dict[metric_name] = metric_value.double().mean()
	return metric_dict

# will be imported directly later	
def get_aggregated_metrics(metric_name_list, metric_outputs):
    """Get a dictionary of the mean metric values (to log) from metric names and their values"""
    metric_dict = {}
    for metric_name in metric_name_list:
        metric_dim = len(metric_outputs[0][metric_name].shape)
        if metric_dim == 0:
            metric_value = torch.stack([x[metric_name] for x in metric_outputs]).mean()
        else:
            metric_value = torch.cat([x[metric_name] for x in metric_outputs]).double().mean()
        metric_dict[metric_name] = metric_value.item()
    return metric_dict

def get_metrics_from_parameter_dict(parameter_dict, device):
    """Get a key-value pair from the hyperparameter dictionary"""
    return {k: torch.tensor(v, device=device) for k, v in parameter_dict.items()}
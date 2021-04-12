from torch import nn

from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
import TRNmodule_lightning
import math

from colorama import init
from colorama import Fore, Back, Style

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

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

class VideoModel(pl.LightningModule):
	def __init__(self, num_class, baseline_type, frame_aggregation, modality,
				train_segments=5, val_segments=25,
				base_model='resnet101', path_pretrained='', new_length=None,
				before_softmax=True,
				dropout_i=0.5, dropout_v=0.5, use_bn='none', ens_DA='none',
				crop_num=1, partial_bn=True, verbose=True, add_fc=1, fc_dim=1024,
				n_rnn=1, rnn_cell='LSTM', n_directions=1, n_ts=5,
				use_attn='TransAttn', n_attn=1, use_attn_frame='none',
				share_params='Y'):
		super(VideoModel, self).__init__()
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
			print(("""
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

	def _prepare_DA(self, num_class, base_model, modality): # convert the model to DA framework
		if base_model == 'c3d': # C3D mode: in construction...
			from C3D_model import C3D
			model_test = C3D()
			self.feature_dim = model_test.fc7.in_features
		elif base_model == "TBN" and modality=="ALL":
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
			raise ValueError(Back.RED + 'add at least one fc layer')

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
		if self.use_bn != 'none':  # S & T: use AdaBN (ICLRW 2017) approach
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
			self.TRN = TRNmodule_lightning.RelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
		elif self.frame_aggregation == 'trn-m':  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 256
			self.TRN = TRNmodule_lightning.RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
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

		# 1. source feature layers (video-level)
		self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_video_source.weight, 0, std)
		constant_(self.fc_feature_video_source.bias, 0)

		self.fc_feature_video_source_2 = nn.Linear(feat_video_dim, feat_video_dim)
		normal_(self.fc_feature_video_source_2.weight, 0, std)
		constant_(self.fc_feature_video_source_2.bias, 0)

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
		if self.use_bn != 'none':  # S & T: use AdaBN (ICLRW 2017) approach
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
		super(VideoModel, self).train(mode)
		count = 0
		if self._enable_pbn:
			print("Freezing BatchNorm2D except the first one.")
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

	def forward(self, input_source, input_target, beta, mu, is_train, reverse):
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
			raise ValueError(Back.RED + 'not enough fc layer')

		feat_fc_source = self.fc_feature_shared_source(feat_base_source)
		feat_fc_target = self.fc_feature_shared_target(feat_base_target) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target)

		# adaptive BN
		if self.use_bn != 'none':
			feat_fc_source, feat_fc_target = self.domainAlign(feat_fc_source, feat_fc_target, is_train, 'shared', self.alpha.item(), num_segments, 1)

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
		pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta)
		pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)

		pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

		if self.use_attn_frame != 'none': # attend the frame-level features only
			feat_fc_source = self.get_attn_feat_frame(feat_fc_source, pred_fc_domain_frame_source)
			feat_fc_target = self.get_attn_feat_frame(feat_fc_target, pred_fc_domain_frame_target)

		#=== source layers (frame-level) ===#

		pred_fc_source = (self.fc_classifier_source_verb(feat_fc_source), self.fc_classifier_source_noun(feat_fc_source))
		pred_fc_target = (self.fc_classifier_target_verb(feat_fc_target) if self.share_params == 'N' else self.fc_classifier_source_verb(feat_fc_target),
						  self.fc_classifier_target_noun(feat_fc_target) if self.share_params == 'N' else self.fc_classifier_source_noun(feat_fc_target))
		if self.baseline_type == 'frame':
			feat_all_source.append(pred_fc_source.view((batch_source, num_segments) + pred_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
			feat_all_target.append(pred_fc_target.view((batch_target, num_segments) + pred_fc_target.size()[-1:]))

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
			pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
			pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)

			# transferable attention
			if self.use_attn != 'none': # get the attention weighting
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

			if self.use_bn != 'none':
				feat_fc_video_source_3_1, feat_fc_video_target_3_1 = self.domainAlign(feat_fc_video_source_3_1, feat_fc_video_target_3_1, is_train, 'temconv_1', self.alpha.item(), num_segments, 1)

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
			feat_fc_video_source = GradReverse.apply(feat_fc_video_source, mu)
			feat_fc_video_target = GradReverse.apply(feat_fc_video_target, mu)

		pred_fc_video_source = (self.fc_classifier_video_verb_source(feat_fc_video_source), self.fc_classifier_video_noun_source(feat_fc_video_source))
		pred_fc_video_target = (self.fc_classifier_video_verb_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_verb_source(feat_fc_video_target),
									 self.fc_classifier_video_noun_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_noun_source(feat_fc_video_target))

		if self.baseline_type == 'video': # only store the prediction from classifier 1 (for now)
			feat_all_source.append(pred_fc_video_source[0].view((batch_source,) + pred_fc_video_source[0].size()[-1:]))
			feat_all_target.append(pred_fc_video_target[0].view((batch_target,) + pred_fc_video_target[0].size()[-1:]))
			feat_all_source.append(pred_fc_video_source[1].view((batch_source,) + pred_fc_video_source[1].size()[-1:]))
			feat_all_target.append(pred_fc_video_target[1].view((batch_target,) + pred_fc_video_target[1].size()[-1:]))

		#=== adversarial branch (video-level) ===#
		pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
		pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)

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


	self.optimizerName = 'SGD'
	self.loss_type = 'nll'
	self.lr = 0.0001
	self.momentum = 0.9
	self.weight_decay = 1e-4
	def configure_optimizers(self):

		if self.optimizerName == 'SGD':
			print(Fore.YELLOW + 'using SGD')
			optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
		elif self.optimizerName == 'Adam':
			print(Fore.YELLOW + 'using Adam')
			optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
		else:
			print(Back.RED + 'optimizer not support or specified!!!')
			exit()
		
		return optimizer

	def training_step(self, train_batch, batch_idx):
		## schedule for parameters
		alpha = 2 / (1 + math.exp(-1 * (self.current_epoch) / args.epochs)) - 1 if args.alpha < 0 else args.alpha

		## schedule for learning rate
		if args.lr_adaptive == 'loss':
			adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, '>')
		elif args.lr_adaptive == 'none' and epoch in args.lr_steps:
			adjust_learning_rate(optimizer, args.lr_decay)

		# define loss function (criterion) and optimizer
		if self.loss_type == 'nll':
			criterion = torch.nn.CrossEntropyLoss().cuda()
			criterion_domain = torch.nn.CrossEntropyLoss().cuda()
		else:
			raise ValueError("Unknown loss type")

		# train for one epoch
		loss_c, attn_epoch_source, attn_epoch_target = train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, train_file, train_short_file, alpha, beta, gamma, mu)
		
		if args.save_attention >= 0:
			attn_source_all = torch.cat((attn_source_all, attn_epoch_source.unsqueeze(0)))  # save the attention values
			attn_target_all = torch.cat((attn_target_all, attn_epoch_target.unsqueeze(0)))  # save the attention values

		# update the recorded loss_c
		loss_c_previous = loss_c_current
		loss_c_current = loss_c

		# evaluate on validation set
		if epoch % args.eval_freq == 0 or epoch == args.epochs:
			if target_set.labels_available:
				prec1_val, prec1_verb_val, prec1_noun_val = validate(target_loader, model, criterion, num_class, epoch, val_file, writer_val)
				# remember best prec@1 and save checkpoint
				if args.train_metric == "all":
					prec1 = prec1_val
				elif args.train_metric == "noun":
					prec1 = prec1_noun_val
				elif args.train_metric == "verb":
					prec1 = prec1_verb_val
				else:
					raise Exception("invalid metric to train")
				is_best = prec1 > best_prec1
				if is_best:
					best_prec1 = prec1_val

				line_update = ' ==> updating the best accuracy' if is_best else ''
				line_best = "Best score {} vs current score {}".format(best_prec1, prec1) + line_update
				print(Fore.YELLOW + line_best)
				val_short_file.write('%.3f\n' % prec1)

				best_prec1 = max(prec1, best_prec1)

				if args.tensorboard:
					writer_val.add_text('Best_Accuracy', str(best_prec1), epoch)
				if args.save_model:
					save_checkpoint({
						'epoch': epoch,
						'arch': args.arch,
						'state_dict': model.state_dict(),
						'optimizer' : optimizer.state_dict(),
						'best_prec1': best_prec1,
						'prec1': prec1,
					}, is_best, path_exp)

			else:
				save_checkpoint({
					'epoch': epoch,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'best_prec1':  0.0,
					'prec1': 0.0,
				}, False, path_exp)

		return loss




# train for one epoch

def train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log, log_short, alpha, beta, gamma, mu):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_a = AverageMeter()  # adversarial loss
	losses_d = AverageMeter()  # discrepancy loss
	losses_e_verb = AverageMeter()
	losses_e_noun = AverageMeter()
	losses_s = AverageMeter()  # ensemble loss
	losses_c = AverageMeter()
	losses_c_verb = AverageMeter()  # classification loss
	losses_c_noun = AverageMeter()  # classification loss
	losses = AverageMeter()
	top1_verb = AverageMeter()
	top5_verb = AverageMeter()
	top1_noun = AverageMeter()
	top5_noun = AverageMeter()
	top1_action = AverageMeter()
	top5_action = AverageMeter()

	if args.no_partialbn:
		model.module.partialBN(False)
	else:
		model.module.partialBN(True)

	# switch to train mode
	model.train()

	end = time.time()
	data_loader = enumerate(zip(source_loader, target_loader))

	# step info
	start_steps = epoch * len(source_loader)
	total_steps = args.epochs * len(source_loader)

	# initialize the embedding
	if args.tensorboard:
		feat_source_display = None
		feat_source_display_noun = None
		feat_source_display_verb = None
		label_source_verb_display = None
		label_source_noun_display = None
		label_source_domain_display = None

		feat_target_display = None
		feat_target_display_noun = None
		feat_target_display_verb = None
		label_target_noun_display = None
		label_target_verb_display = None
		label_target_domain_display = None

	attn_epoch_source = torch.Tensor()
	attn_epoch_target = torch.Tensor()
	for i, ((source_data, source_label, source_id), (target_data, target_label, target_id)) in data_loader:
		# setup hyperparameters
		p = float(i + start_steps) / total_steps
		beta_dann = 2. / (1. + np.exp(-1.0 * p)) - 1
		beta = [beta_dann if beta[i] < 0 else beta[i] for i in range(len(beta))] # replace the default beta if value < 0
		if args.dann_warmup:
		    beta_new = [beta_dann*beta[i] for i in range(len(beta))]
		else:
			beta_new = beta
		source_size_ori = source_data.size()  # original shape
		target_size_ori = target_data.size()  # original shape
		batch_source_ori = source_size_ori[0]
		batch_target_ori = target_size_ori[0]
		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_source_ori < args.batch_size[0]:
			source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
			source_data = torch.cat((source_data, source_data_dummy))
		if batch_target_ori < args.batch_size[1]:
			target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
			target_data = torch.cat((target_data, target_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if source_data.size(0) % gpu_count != 0:
			source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
			source_data = torch.cat((source_data, source_data_dummy))
		if target_data.size(0) % gpu_count != 0:
			target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
			target_data = torch.cat((target_data, target_data_dummy))

		# measure data loading time
		data_time.update(time.time() - end)

		source_label_verb = source_label[0].cuda(non_blocking=True) # pytorch 0.4.X
		source_label_noun = source_label[1].cuda(non_blocking=True)  # pytorch 0.4.X

		target_label_verb = target_label[0].cuda(non_blocking=True) # pytorch 0.4.X
		target_label_noun = target_label[1].cuda(non_blocking=True)  # pytorch 0.4.X

		if args.baseline_type == 'frame':
			source_label_verb_frame = source_label_verb.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
			source_label_noun_frame = source_label_noun.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
			target_label_verb_frame = target_label_verb.unsqueeze(1).repeat(1, args.num_segments).view(-1)
			target_label_noun_frame = target_label_noun.unsqueeze(1).repeat(1, args.num_segments).view(-1)

		label_source_verb = source_label_verb_frame if args.baseline_type == 'frame' else source_label_verb  # determine the label for calculating the loss function
		label_target_verb = target_label_verb_frame if args.baseline_type == 'frame' else target_label_verb

		label_source_noun = source_label_noun_frame if args.baseline_type == 'frame' else source_label_noun  # determine the label for calculating the loss function
		label_target_noun = target_label_noun_frame if args.baseline_type == 'frame' else target_label_noun
		#====== pre-train source data ======#
		if args.pretrain_source:
			#------ forward pass data again ------#
			_, out_source, out_source_2, _, _, _, _, _, _, _ = model(source_data, target_data, beta_new, mu, is_train=True, reverse=False)

			# ignore dummy tensors
			out_source_verb = out_source[0][:batch_source_ori]
			out_source_noun = out_source[1][:batch_source_ori]
			out_source_2 = out_source_2[:batch_source_ori]

			#------ calculate the loss function ------#
			# 1. calculate the classification loss
			out_verb = out_source_verb
			out_noun = out_source_noun
			label_verb = label_source_verb
			label_noun = label_source_noun

			# MCD not used
			loss_verb = criterion(out_verb, label_verb)
			loss_noun = criterion(out_noun, label_noun)
			if args.train_metric == "all":
				loss = 0.5 * (loss_verb + loss_noun)
			elif args.train_metric == "noun":
				loss = loss_noun  # 0.5*(loss_verb+loss_noun)
			elif args.train_metric == "verb":
				loss = loss_verb  # 0.5*(loss_verb+loss_noun)
			else:
				raise Exception("invalid metric to train")
			#if args.ens_DA == 'MCD' and args.use_target != 'none':
			#	loss += criterion(out_source_2, label)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()

			if args.clip_gradient is not None:
				total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
				if total_norm > args.clip_gradient and args.verbose:
					print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

			optimizer.step()


		#====== forward pass data ======#
		attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta_new, mu, is_train=True, reverse=False)

		# ignore dummy tensors
		attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_source_ori)
		attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)


		# Pred normalise not use
		#if args.pred_normalize == 'Y': # use the uncertainly method (in contruction...)
		#	out_source = out_source / out_source.var().log()
		#	out_target = out_target / out_target.var().log()

		# store the embedding
		if args.tensorboard:
			feat_source_display_noun = feat_source[0] if i == 0 else torch.cat((feat_source_display_noun, feat_source[0]), 0)
			feat_source_display_verb = feat_source[1] if i==0 else torch.cat((feat_source_display_verb, feat_source[1]), 0)
			feat_source_display = feat_source[2] if i == 0 else torch.cat((feat_source_display, feat_source[2]), 0)

			label_source_verb_display = label_source_verb if i==0 else torch.cat((label_source_verb_display, label_source_verb), 0)
			label_source_noun_display = label_source_noun if i == 0 else torch.cat((label_source_noun_display, label_source_noun),0)
			label_source_domain_display = torch.zeros(label_source_verb.size(0)) if i==0 else torch.cat((label_source_domain_display, torch.zeros(label_source_verb.size(0))), 0)

			feat_target_display_noun = feat_target[0] if i==0 else torch.cat((feat_target_display_noun, feat_target[0]), 0)
			feat_target_display_verb = feat_target[1] if i == 0 else torch.cat((feat_target_display_verb, feat_target[1]), 0)
			feat_target_display = feat_target[2] if i == 0 else torch.cat((feat_target_display, feat_target[2]), 0)

			label_target_verb_display = label_target_verb if i==0 else torch.cat((label_target_verb_display, label_target_verb), 0)
			label_target_noun_display = label_target_noun if i == 0 else torch.cat((label_target_noun_display, label_target_noun), 0)
			label_target_domain_display = torch.ones(label_target_verb.size(0)) if i==0 else torch.cat((label_target_domain_display, torch.ones(label_target_verb.size(0))), 0)

		#====== calculate the loss function ======#
		# 1. calculate the classification loss
		out_verb = out_source[0]
		out_noun = out_source[1]
		label_verb = label_source_verb
		label_noun = label_source_noun

		#Sv not used
		#if args.use_target == 'Sv':
		#	out = torch.cat((out, out_target))
		#	label = torch.cat((label, label_target))

		loss_verb = criterion(out_verb, label_verb)
		loss_noun = criterion(out_noun, label_noun)
		if args.train_metric == "all":
			loss_classification = 0.5*(loss_verb+loss_noun)
		elif args.train_metric == "noun":
			loss_classification = loss_noun# 0.5*(loss_verb+loss_noun)
		elif args.train_metric == "verb":
			loss_classification = loss_verb  # 0.5*(loss_verb+loss_noun)
		else:
			raise Exception("invalid metric to train")

		#MCD  not used
		#if args.ens_DA == 'MCD' and args.use_target != 'none':
		#	loss_classification += criterion(out_source_2, label)


		losses_c_verb.update(loss_verb.item(), out_verb.size(0)) # pytorch 0.4.X
		losses_c_noun.update(loss_noun.item(), out_noun.size(0))  # pytorch 0.4.X
		loss = loss_classification
		losses_c.update(loss_classification.item(), out_verb.size(0))

		# 2. calculate the loss for DA
		# (I) discrepancy-based approach: discrepancy loss
		if args.dis_DA != 'none' and args.use_target != 'none':
			loss_discrepancy = 0

			kernel_muls = [2.0]*2
			kernel_nums = [2, 5]
			fix_sigma_list = [None]*2

			if args.dis_DA == 'JAN':
				# ignore the features from shared layers
				feat_source_sel = feat_source[:-args.add_fc]
				feat_target_sel = feat_target[:-args.add_fc]

				size_loss = min(feat_source_sel[0].size(0), feat_target_sel[0].size(0))  # choose the smaller number
				feat_source_sel = [feat[:size_loss] for feat in feat_source_sel]
				feat_target_sel = [feat[:size_loss] for feat in feat_target_sel]

				loss_discrepancy += JAN(feat_source_sel, feat_target_sel, kernel_muls=kernel_muls, kernel_nums=kernel_nums, fix_sigma_list=fix_sigma_list, ver=2)

			else:
				# extend the parameter list for shared layers
				kernel_muls.extend([kernel_muls[-1]]*args.add_fc)
				kernel_nums.extend([kernel_nums[-1]]*args.add_fc)
				fix_sigma_list.extend([fix_sigma_list[-1]]*args.add_fc)

				for l in range(0, args.add_fc + 2):  # loss from all the features (+2 because of frame-aggregation layer + final fc layer)
					if args.place_dis[l] == 'Y':
						# select the data for calculating the loss (make sure source # == target #)
						size_loss = min(feat_source[l].size(0), feat_target[l].size(0)) # choose the smaller number
						# select
						feat_source_sel = feat_source[l][:size_loss]
						feat_target_sel = feat_target[l][:size_loss]

						# break into multiple batches to avoid "out of memory" issue
						size_batch = min(256,feat_source_sel.size(0))
						feat_source_sel = feat_source_sel.view((-1,size_batch) + feat_source_sel.size()[1:])
						feat_target_sel = feat_target_sel.view((-1,size_batch) + feat_target_sel.size()[1:])

						if args.dis_DA == 'CORAL':
							losses_coral = [CORAL(feat_source_sel[t], feat_target_sel[t]) for t in range(feat_source_sel.size(0))]
							loss_coral = sum(losses_coral)/len(losses_coral)
							loss_discrepancy += loss_coral
						elif args.dis_DA == 'DAN':
							losses_mmd = [mmd_rbf(feat_source_sel[t], feat_target_sel[t], kernel_mul=kernel_muls[l], kernel_num=kernel_nums[l], fix_sigma=fix_sigma_list[l], ver=2) for t in range(feat_source_sel.size(0))]
							loss_mmd = sum(losses_mmd) / len(losses_mmd)

							loss_discrepancy += loss_mmd
						else:
							raise NameError('not in dis_DA!!!')

			losses_d.update(loss_discrepancy.item(), feat_source[0].size(0))
			loss += alpha * loss_discrepancy

		# (II) adversarial discriminative model: adversarial loss
		if args.adv_DA != 'none' and args.use_target != 'none':
			loss_adversarial = 0
			pred_domain_all = []
			pred_domain_target_all = []

			for l in range(len(args.place_adv)):
				if args.place_adv[l] == 'Y':

					# reshape the features (e.g. 128x5x2 --> 640x2)
					pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
					pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

					# prepare domain labels
					source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
					target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
					domain_label = torch.cat((source_domain_label,target_domain_label),0)

					domain_label = domain_label.cuda(non_blocking=True)

					pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single),0)
					pred_domain_all.append(pred_domain)
					pred_domain_target_all.append(pred_domain_target_single)

					if args.pred_normalize == 'Y':  # use the uncertainly method (in construction......)
						pred_domain = pred_domain / pred_domain.var().log()
					loss_adversarial_single = criterion_domain(pred_domain, domain_label)

					loss_adversarial += loss_adversarial_single

			losses_a.update(loss_adversarial.item(), pred_domain.size(0))
			loss += loss_adversarial

		# (III) other loss
		# 1. entropy loss for target data
		if args.add_loss_DA == 'target_entropy' and args.use_target != 'none':
			loss_entropy_verb = cross_entropy_soft(out_target[0])
			loss_entropy_noun = cross_entropy_soft(out_target[1])
			losses_e_verb.update(loss_entropy_verb.item(), out_target[0].size(0))
			losses_e_noun.update(loss_entropy_noun.item(), out_target[1].size(0))
			if args.train_metric == "all":
				loss += gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)
			elif args.train_metric == "noun":
				loss += gamma * loss_entropy_noun
			elif args.train_metric == "verb":
				loss += gamma * loss_entropy_verb
			else:
				raise Exception("invalid metric to train")
			#loss += gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)

		# # 2. discrepancy loss for MCD (CVPR 18)
		# Not used
		# if args.ens_DA == 'MCD' and args.use_target != 'none':
		# 	_, _, _, _, _, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta, mu, is_train=True, reverse=True)
		#
		# 	# ignore dummy tensors
		# 	_, out_target, out_target_2, _, _ = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)
		#
		# 	loss_dis = -dis_MCD(out_target, out_target_2)
		# 	losses_s.update(loss_dis.item(), out_target.size(0))
		# 	loss += loss_dis

		# 3. attentive entropy loss
		if args.add_loss_DA == 'attentive_entropy' and args.use_attn != 'none' and args.use_target != 'none':
			loss_entropy_verb = attentive_entropy(torch.cat((out_verb, out_target[0]),0), pred_domain_all[1])
			loss_entropy_noun = attentive_entropy(torch.cat((out_noun, out_target[1]), 0), pred_domain_all[1])
			losses_e_verb.update(loss_entropy_verb.item(), out_target[0].size(0))
			losses_e_noun.update(loss_entropy_noun.item(), out_target[1].size(0))
			if args.train_metric == "all":
				loss += gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)
			elif args.train_metric == "noun":
				loss += gamma * loss_entropy_noun
			elif args.train_metric == "verb":
				loss += gamma * loss_entropy_verb
			else:
				raise Exception("invalid metric to train")
			#loss += gamma * 0.5*(loss_entropy_verb + loss_entropy_noun)
		# measure accuracy and record loss
		pred_verb = out_verb
		prec1_verb, prec5_verb = accuracy(pred_verb.data, label_verb, topk=(1, 5))
		pred_noun = out_noun
		prec1_noun, prec5_noun = accuracy(pred_noun.data, label_noun, topk=(1, 5))
		prec1_action, prec5_action = multitask_accuracy((pred_verb.data, pred_noun.data), (label_verb, label_noun), topk=(1, 5))


		losses.update(loss.item())
		top1_verb.update(prec1_verb.item(), out_verb.size(0))
		top5_verb.update(prec5_verb.item(), out_verb.size(0))
		top1_noun.update(prec1_noun.item(), out_noun.size(0))
		top5_noun.update(prec5_noun.item(), out_noun.size(0))
		top1_action.update(prec1_action, out_noun.size(0))
		top5_action.update(prec5_action, out_noun.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()

		loss.backward()

		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
			if total_norm > args.clip_gradient and args.verbose:
				print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			line = 'Train: [{0}][{1}/{2}], lr: {lr:.5f}\t' + \
				   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
				   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
				   'Prec@1 {top1_verb.val:.3f} ({top1_verb.avg:.3f})\t' + \
				   'Prec@1 {top1_noun.val:.3f} ({top1_noun.avg:.3f})\t' + \
				   'Prec@1 {top1_action.val:.3f} ({top1_action.avg:.3f})\t' + \
				   'Prec@5 {top5_verb.val:.3f} ({top5_verb.avg:.3f})\t' + \
				   'Prec@5 {top5_noun.val:.3f} ({top5_noun.avg:.3f})\t' + \
				   'Prec@5 {top5_action.val:.3f} ({top5_action.avg:.3f})\t' + \
				   'Loss {loss.val:.4f} ({loss.avg:.4f})   loss_verb {loss_verb.avg:.4f}   loss_noun {loss_noun.avg:.4f}\t'

			if args.dis_DA != 'none' and args.use_target != 'none':
				line += 'alpha {alpha:.3f}  loss_d {loss_d.avg:.4f}\t'

			if args.adv_DA != 'none' and args.use_target != 'none':
				line += 'beta {beta[0]:.3f}, {beta[1]:.3f}, {beta[2]:.3f}  loss_a {loss_a.avg:.4f}\t'

			if args.add_loss_DA != 'none' and args.use_target != 'none':
				line += 'gamma {gamma:.6f}  loss_e_verb {loss_e_verb.avg:.4f} loss_e_noun {loss_e_noun.avg:.4f}\t'

			if args.ens_DA != 'none' and args.use_target != 'none':
				line += 'mu {mu:.6f}  loss_s {loss_s.avg:.4f}\t'

			line = line.format(
				epoch, i, len(source_loader), batch_time=batch_time, data_time=data_time, alpha=alpha, beta=beta_new, gamma=gamma, mu=mu,
				loss=losses, loss_verb=losses_c_verb, loss_noun=losses_c_noun, loss_d=losses_d, loss_a=losses_a,
				loss_e_verb=losses_e_verb, loss_e_noun=losses_e_noun, loss_s=losses_s, top1_verb=top1_verb,
				top1_noun=top1_noun, top5_verb=top5_verb, top5_noun=top5_noun, top1_action=top1_action, top5_action=top5_action,
				lr=optimizer.param_groups[0]['lr'])

			if i % args.show_freq == 0:
				print(line)

			log.write('%s\n' % line)

		# adjust the learning rate for ech step (e.g. DANN)
		if args.lr_adaptive == 'dann':
			adjust_learning_rate_dann(optimizer, p)

		# save attention values w/ the selected class
		if args.save_attention >= 0:
			attn_source = attn_source[source_label==args.save_attention]
			attn_target = attn_target[target_label==args.save_attention]
			attn_epoch_source = torch.cat((attn_epoch_source, attn_source.cpu()))
			attn_epoch_target = torch.cat((attn_epoch_target, attn_target.cpu()))

	# update the embedding every epoch
	if args.tensorboard:
		n_iter_train = epoch * len(source_loader) # calculate the total iteration
		# embedding

		writer_train.add_scalar("loss/verb", losses_c_verb.avg, epoch)
		writer_train.add_scalar("loss/noun", losses_c_noun.avg, epoch)
		writer_train.add_scalar("acc/verb", top1_verb.avg, epoch)
		writer_train.add_scalar("acc/noun", top1_noun.avg, epoch)
		writer_train.add_scalar("acc/action", top1_action.avg, epoch)
		if args.adv_DA != 'none' and args.use_target != 'none':
			writer_train.add_scalar("loss/domain", loss_adversarial,epoch)
		# indicies_source = np.random.randint(0,len(feat_source_display),150)
		# indicies_target = np.random.randint(0, len(feat_target_display), 150)
		# label_source_verb_display = label_source_verb_display[indicies_source]
		# label_target_verb_display = label_target_verb_display[indicies_target]
		# feat_source_display = feat_source_display[indicies_source]
		# feat_target_display = feat_target_display[indicies_target]

	log_short.write('%s\n' % line)
	return losses_c.avg, attn_epoch_source.mean(0), attn_epoch_target.mean(0)

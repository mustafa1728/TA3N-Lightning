import os

from models_lightning import VideoModel

import torch
import torch.backends.cudnn as cudnn

import logging
logging.basicConfig(format='%(asctime)s  |  %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

from tensorboardX import SummaryWriter

def set_hyperparameters(model, args):
    model.optimizerName = args.optimizer
    model.loss_type = args.loss_type
    model.lr = args.lr
    model.momentum = args.momentum
    model.weight_decay = args.weight_decay
    model.epochs = args.epochs
    model.batch_size = args.batch_size
    model.eval_freq = args.eval_freq

    model.lr_adaptive = args.lr_adaptive
    model.lr_decay = args.lr_decay
    model.lr_steps = args.lr_steps

    model.alpha = args.alpha
    model.beta = args.beta
    model.gamma = args.gamma
    model.mu = args.mu

    model.train_metric = args.train_metric
    model.dann_warmup = args.dann_warmup

    model.tensorboard = True
    model.path_exp = model.modality + '/'
    if not os.path.isdir(model.path_exp):
        os.makedirs(model.path_exp)
    model.writer_train = SummaryWriter(model.path_exp + '/tensorboard_train')  # for tensorboardX
    model.writer_val = SummaryWriter(model.path_exp + '/tensorboard_val')  # for tensorboardX

    model.pretrain_source = args.pretrain_source
    model.clip_gradient = args.clip_gradient

    model.dis_DA = args.dis_DA
    model.use_target = args.use_target
    model.add_fc = args.add_fc
    model.place_dis = args.place_dis
    model.place_adv = args.place_adv
    model.pred_normalize = args.pred_normalize
    model.add_loss_DA = args.add_loss_DA
    model.print_freq = args.print_freq
    model.show_freq = args.show_freq
    model.ens_DA = args.ens_DA

    model.arch = args.arch
    model.save_model = args.save_model
    model.labels_available = True
    model.adv_DA = args.adv_DA

    if model.loss_type == 'nll':
        model.criterion = torch.nn.CrossEntropyLoss()
        model.criterion_domain = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss type")

def initialise_trainer(args):
    
    logging.debug('Baseline:' + args.baseline_type)
    logging.debug('Frame aggregation method:' + args.frame_aggregation)

    logging.debug('target data usage:' + args.use_target)
    if args.use_target == 'none':
        logging.debug('no Domain Adaptation')
    else:
        if args.dis_DA != 'none':
            logging.debug('Apply the discrepancy-based Domain Adaptation approach:'+ args.dis_DA)
            if len(args.place_dis) != args.add_fc + 2:
                logging.error('len(place_dis) should be equal to add_fc + 2')
                raise ValueError('len(place_dis) should be equal to add_fc + 2')

        if args.adv_DA != 'none':
            logging.debug('Apply the adversarial-based Domain Adaptation approach:'+ args.adv_DA)

        if args.use_bn != 'none':
            logging.debug('Apply the adaptive normalization approach:'+ args.use_bn)

    # determine the categories
    #want to allow multi-label classes.

    #Original way to compute number of classes
    ####class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
    ####num_class = len(class_names)

    #New approach
    num_class_str = args.num_class.split(",")
    #single class
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
        for num in num_class_str:
            num_class.append(int(num))

    #=== check the folder existence ===#
    path_exp = args.exp_path + args.modality + '/'
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)


    #=== initialize the model ===#
    logging.info('preparing the model......')
    model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
            train_segments=args.num_segments, val_segments=args.val_segments, 
            base_model=args.arch, path_pretrained=args.pretrained,
            add_fc=args.add_fc, fc_dim = args.fc_dim,
            dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
            use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
            n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
            use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
            verbose=args.verbose, share_params=args.share_params)

    if args.optimizer == 'SGD':
        logging.debug('using SGD')
        model.optimizerName = 'SGD'
    elif args.optimizer == 'Adam':
        logging.debug( 'using Adam')
        model.optimizerName = 'Adam'
    else:
        logging.error('optimizer not support or specified!!!')
        exit()

    #=== check point ===#
    start_epoch = 1
    logging.debug('checking the checkpoint......')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.debug("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        if args.resume_hp:
            logging.debug("=> loaded checkpoint hyper-parameters")
            model.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        logging.error("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True



    # --- Optimizer ---#
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        model.loss_type = 'nll'
    else:
        raise ValueError("Unknown loss type")

    # --- Parameters ---#
    model.beta = args.beta
    model.gamma = args.gamma
    model.mu = args.mu

    set_hyperparameters(model, args)

    return model

def set_hyperparameters_test(model, args):
    model.batch_size = [args.bS]
    model.alpha = 1
    model.beta = [1, 1, 1]
    model.gamma = 1
    model.mu = 0

def initialise_tester(args):
    # New approach
    num_class_str = args.num_class.split(",")
    # single class
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
    for num in num_class_str:
        num_class.append(int(num))

        
    verb_net = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
        train_segments=args.test_segments if args.baseline_type == 'video' else 1, val_segments=args.test_segments if args.baseline_type == 'video' else 1,
        base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
        dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
        n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
        use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
        verbose=args.verbose, before_softmax=False)

    verb_checkpoint = torch.load(args.weights)

    verb_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(verb_checkpoint['state_dict'].items())}
    verb_net.load_state_dict(verb_base_dict)
    # verb_net = torch.nn.DataParallel(verb_net)
    set_hyperparameters_test(verb_net, args)
    verb_net.eval()

    if args.noun_weights is not None:
        noun_net = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
                        train_segments=args.test_segments if args.baseline_type == 'video' else 1,
                        val_segments=args.test_segments if args.baseline_type == 'video' else 1,
                        base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
                        dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
                        n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
                        use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
                        verbose=args.verbose, before_softmax=False)
        noun_checkpoint = torch.load(args.noun_weights)

        noun_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(noun_checkpoint['state_dict'].items())}
        noun_net.load_state_dict(noun_base_dict)
        # noun_net = torch.nn.DataParallel(noun_net.cuda())
        set_hyperparameters_test(noun_net, args)
        noun_net.eval()
    else:
        noun_net = None

    

    return (verb_net, noun_net)
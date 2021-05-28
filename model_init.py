import os
from colorama import init
from colorama import Fore, Back, Style
from models_lightning import VideoModel
import torch.backends.cudnn as cudnn
import torch


def initialise_trainer(args):

    print(Fore.GREEN + 'Baseline:', args.baseline_type)
    print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)

    print(Fore.GREEN + 'target data usage:', args.use_target)
    if args.use_target == 'none':
        print(Fore.GREEN + 'no Domain Adaptation')
    else:
        if args.dis_DA != 'none':
            print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
            if len(args.place_dis) != args.add_fc + 2:
                raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

        if args.adv_DA != 'none':
            print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

        if args.use_bn != 'none':
            print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

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
    print(Fore.CYAN + 'preparing the model......')
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
        print(Fore.YELLOW + 'using SGD')
        model.optimizerName = 'SGD'
    elif args.optimizer == 'Adam':
        print(Fore.YELLOW + 'using Adam')
        model.optimizerName = 'Adam'
    else:
        print(Back.RED + 'optimizer not support or specified!!!')
        exit()

    #=== check point ===#
    start_epoch = 1
    print(Fore.CYAN + 'checking the checkpoint......')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch'])))
        if args.resume_hp:
            print("=> loaded checkpoint hyper-parameters")
            model.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))

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
    return model


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
        noun_net.eval()
    else:
        noun_net = None

    return (verb_net, noun_net)
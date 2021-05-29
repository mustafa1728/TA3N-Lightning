import argparse


parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('num_class', type=str, default="classInd.txt")
parser.add_argument('modality', type=str, choices=['ALL', 'Audio','RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('test_target_data', type=str)
parser.add_argument('result_json', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--noun_target_data', type=str, default=None)
parser.add_argument('--noun_weights', type=str, default=None)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--test_segments', type=int, default=5)
parser.add_argument('--add_fc', default=1, type=int, metavar='M', help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)')
parser.add_argument('--fc_dim', type=int, default=512, help='dimension of added fc')
parser.add_argument('--baseline_type', type=str, default='frame', choices=['frame', 'video', 'tsn'])
parser.add_argument('--frame_aggregation', type=str, default='avgpool', choices=['avgpool', 'rnn', 'temconv', 'trn-m', 'none'], help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--dropout_i', type=float, default=0)
parser.add_argument('--dropout_v', type=float, default=0)

#------ RNN ------
parser.add_argument('--n_rnn', default=1, type=int, metavar='M',
                    help='number of RNN layers (e.g. 0, 1, 2, ...)')
parser.add_argument('--rnn_cell', type=str, default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('--n_directions', type=int, default=1, choices=[1, 2],
                    help='(bi-) direction RNN')
parser.add_argument('--n_ts', type=int, default=5, help='number of temporal segments')

# ========================= DA Configs ==========================
parser.add_argument('--share_params', type=str, default='Y', choices=['Y', 'N'])
parser.add_argument('--use_bn', type=str, default='none', choices=['none', 'AdaBN', 'AutoDIAL'])
parser.add_argument('--use_attn_frame', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism for frames only')
parser.add_argument('--use_attn', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism')
parser.add_argument('--n_attn', type=int, default=1, help='number of discriminators for transferable attention')

# ========================= Monitor Configs ==========================
parser.add_argument('--top', default=[1, 3, 5], nargs='+', type=int, help='show top-N categories')
parser.add_argument('--verbose', default=False, action="store_true")

# ========================= Runtime Configs ==========================
parser.add_argument('--save_confusion', type=str, default=None)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--save_attention', type=str, default=None)
parser.add_argument('--max_num', type=int, default=-1, help='number of videos to test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--bS', default=2, help='batch size', type=int, required=False)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')


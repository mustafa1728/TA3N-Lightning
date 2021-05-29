
train_file = None
train_short_file = None
val_file = None
val_short_file = None

val_best_file = None

path_exp = ""
start_epoch = 0

def open_log_files(args):
    global train_file, train_short_file, val_file, val_short_file, val_best_file

    global path_exp, start_epoch

    path_exp = args.exp_path + args.modality + '/'
    #--- open log files ---#
    if args.resume:
        train_file = open(path_exp + 'train.log', 'a')
        train_short_file = open(path_exp + 'train_short.log', 'a')
        val_file = open(path_exp + 'val.log', 'a')
        val_short_file = open(path_exp + 'val_short.log', 'a')
        train_file.write('========== start: ' + str(start_epoch) + '\n')  # separation line
        train_short_file.write('========== start: ' + str(start_epoch) + '\n')
        val_file.write('========== start: ' + str(start_epoch) + '\n')
        val_short_file.write('========== start: ' + str(start_epoch) + '\n')
    else:
        train_short_file = open(path_exp + 'train_short.log', 'w')
        val_short_file = open(path_exp + 'val_short.log', 'w')
        train_file = open(path_exp + 'train.log', 'w')
        val_file = open(path_exp + 'val.log', 'w')
        val_best_file = open(path_exp + 'best_val.log', 'a')

def write_log_files(line_time, best_prec1):
    global train_file, train_short_file, val_file, val_short_file, val_best_file

    global path_exp, start_epoch

    train_file.write(line_time)
    train_short_file.write(line_time)

    #--- close log files ---#
    train_file.close()
    train_short_file.close()

    labels_available = True
    if labels_available:
        val_best_file.write('%.3f\n' % best_prec1)
        val_file.write(line_time)
        val_short_file.write(line_time)
        val_file.close()
        val_short_file.close()

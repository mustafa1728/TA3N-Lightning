import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    cyan = "\x1b[36;21m"
    white = "\x1b[37;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format='%(asctime)s | %(levelname)-6s: %(message)s'
    datefmt='%d-%b-%y %H:%M:%S'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

train_file = None
train_short_file = None
val_file = None
val_short_file = None

val_best_file = None

path_exp = ""
start_epoch = 0

def log_info(message):
    logger.info(message)

def log_debug(message):
    logger.debug(message)

def log_error(message):
    logger.error(message)

def log_warning(message):
    logger.warning(message)


def open_log_files(cfg):
    global train_file, train_short_file, val_file, val_short_file, val_best_file

    global path_exp, start_epoch

    path_exp = cfg.PATHS.EXP_PATH
    #--- open log files ---#
    if cfg.TRAINER.RESUME != "":
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
        val_file.write(best_prec1)
        val_short_file.write(best_prec1)
        val_file.close()
        val_short_file.close()

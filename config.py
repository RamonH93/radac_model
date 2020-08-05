import logging
from tensorboard.plugins.hparams import api as hp

LOGLEVEL = logging.DEBUG # any int or DEBUG (10), INFO (20), WARN (30), ERROR (40), CRITICAL (50)
SEED = 1
DATA_LOC = ""
DATA_FILE = "data.csv"
DATA_N = 10000

LOGDIR = 'logs\\hparam_tuning'

# define the hyperparameters
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([100]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

HPARAMS = [HP_NUM_UNITS, HP_BATCH_SIZE, HP_OPTIMIZER]

METRIC_ACCURACY = 'accuracy'

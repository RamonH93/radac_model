import os
import logging
import shutil
from datetime import datetime
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
from tensorboard.plugins.hparams import api as hp
import config as cf
import run
import utils
import generate_dataset as gd
import preprocess_data as ppd
# call before tf import; >'0' to suppress tf module messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf # pylint: disable=wrong-import-position,wrong-import-order
from tensorflow import keras # pylint: disable=wrong-import-position,wrong-import-order
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.autograph.set_verbosity(1)


# Initialize logger
logger = tf.get_logger()
# default style: %(levelname)s:%(name)s:%(message)s
logger.handlers[0].setFormatter(logging.Formatter(
    "({asctime}) {levelname:>8}: {message}", style='{'))
# logger.propagate = False # prevent double tf log output
logger.setLevel(cf.LOGLEVEL)
logger.log(logger.getEffectiveLevel(), '<--- Effective log level')

# Enable to find out which devices your operations and tensors are assigned to
# Only works when imported tf with argument '0'
try:
    tf.debugging.set_log_device_placement(False)
except RuntimeError:
    pass

# clear logs
shutil.rmtree('logs')

# generate dataset
data_src = os.path.join(cf.DATA_LOC, cf.DATA_FILE)
if not os.path.exists(data_src):
    if cf.DATA_LOC == "":
        logger.warning('Saving data to local storage')
    gd.generate_dataset(n=cf.DATA_N, seed=cf.SEED, dest=data_src, logger=logger)
else:
    logger.info('Skipping data generation, file already exists: %s', data_src)

# preprocess data
X, y = ppd.preprocess_data(data_src=data_src, seed=cf.SEED, plot=False, logger=logger)

# write hyperparameters
with tf.summary.create_file_writer(cf.LOGDIR).as_default():
    hp.hparams_config(
        hparams=cf.HPARAMS,
        metrics=[hp.Metric(cf.METRIC_ACCURACY, display_name='Accuracy')],
    )

dist_strat = utils.dist_strategy(logger=logger)

session_num = 0

train_index, test_index = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cf.SEED).split(X, y)) # pylint: disable=line-too-long
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

param_grid = ParameterGrid({h.name: h.domain.values for h in cf.HPARAMS})

for _, paramset in enumerate(param_grid):
    run_name = "run-%d" % session_num
    logger.info('--- Starting trial: %s' % run_name)
    logger.debug(paramset)

    # Define the Keras callbacks
    _logdir = os.path.join(
        cf.LOGDIR,
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-{paramset["optimizer"]}')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=_logdir)
    # hparams_callback = hp.KerasCallback(_logdir, hparams)
    earlystop_callback = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0001,
        patience=3)
    callbacks = []

    run.run(_logdir, X_train, y_train, X_test, y_test, paramset, callbacks, dist_strat)
    session_num += 1

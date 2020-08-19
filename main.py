import os
import logging
import shutil
from datetime import datetime
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tensorflow import keras
import config as cf
import utils
import generate_dataset as gd
import preprocess_data as ppd
import run

def main():
    # clear logs
    if os.path.exists('logs'):
        shutil.rmtree('logs')

    # Initialize logger
    logger = tf.get_logger()
    # default style: %(levelname)s:%(name)s:%(message)s
    logger.handlers[0].setFormatter(logging.Formatter(
        "({asctime}) {levelname:>8}: {message}", style='{'))
    logger.setLevel(cf.LOGLEVEL)
    logger.log(logger.getEffectiveLevel(), '<--- Effective log level')

    # Enable to find out which devices your operations and tensors are assigned to
    try:
        tf.debugging.set_log_device_placement(False)
    except RuntimeError:
        pass

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
    logger.debug("Preprocessed data successfully")

    # write hyperparameters
    with tf.summary.create_file_writer(cf.LOGDIR).as_default():
        hp.hparams_config(
            hparams=cf.HPARAMS,
            metrics=[hp.Metric(cf.METRIC_ACCURACY, display_name='Accuracy')],
        )
    logger.debug("Wrote hyperparameters")

    dist_strat = utils.dist_strategy(logger=logger)
    logger.debug("Selected distribution strategy")

    session_num = 0

    train_indices, test_indices = next(StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=cf.SEED
        ).split(X, y))
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    logger.debug("Split train/test")

    param_grid = ParameterGrid({h.name: h.domain.values for h in cf.HPARAMS})

    for _, paramset in enumerate(param_grid):
        run_name = "run-%d" % session_num
        logger.info('--- Starting trial: %s' % run_name)
        logger.debug(paramset)

        # Define the Keras callbacks
        _logdir = os.path.join(
            cf.LOGDIR,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-{paramset["optimizer"]}')
        tensorboard = keras.callbacks.TensorBoard(log_dir=_logdir, histogram_freq=1)
        cp_path = os.path.join(_logdir, 'weights.ckpt')
        os.makedirs(os.path.dirname(cp_path))
        checkpoint = keras.callbacks.ModelCheckpoint(
            cp_path,
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch'
        )
        # hparams = {h: paramset[h.name] for h in cf.HPARAMS}
        # hparams_callback = hp.KerasCallback(_logdir, hparams)
        earlystopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0001,
            patience=3)
        callbacks = [tensorboard, checkpoint, earlystopping]

        # Resets all state generated previously by Keras
        keras.backend.clear_session()

        run.run(_logdir, X_train, y_train, X_test, y_test, paramset, callbacks, dist_strat)
        session_num += 1

main()

from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils import compute_class_weight
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras

import utils


def train_test_model(run_name, X_train, y_train, X_test, y_test, paramset, callbacks, config, dist_strat, show=False):  # pylint: disable=line-too-long
    logger = config['run_params']['logger']

    model = keras.models.Sequential([
        keras.Input(shape=(X_train.shape[1], ), name='inputs'),
        # keras.Input(shape=list(dataset.take(1).as_numpy_iterator())[0][0].shape)) #tfdataset
        keras.layers.Dense(paramset['num_units'],
                           input_shape=(X_train.shape[1], ),
                           activation=tf.nn.relu,
                           name='dense'),
        keras.layers.Dense(paramset['num_units'] * 2,
                           input_shape=(X_train.shape[1], ),
                           activation=tf.nn.relu,
                           name='dense_1'),
        keras.layers.Dense(paramset['num_units'],
                           input_shape=(X_train.shape[1], ),
                           activation=tf.nn.relu,
                           name='dense_2'),
        # kernel_initializer=keras.initializers.Ones(),
        # bias_initializer=keras.initializers.Zeros(),
        keras.layers.Dense(
            1,
            input_shape=(paramset['num_units'], ),
            activation=tf.nn.sigmoid,
            # activation=tf.nn.softmax,
            name='output'),
    ])

    # Experimental custom loss function
    def dice_coef_loss(y_pred, y_true, alpha=0.3, beta=0.7, smooth=1e-10):
        return keras.backend.square(tf.subtract(y_true, y_pred))
        # labels = keras.backend.flatten(labels)
        # predictions = keras.backend.flatten(predictions)
        # truepos = tf.reduce_sum(labels * predictions)
        # fp_and_fn = (alpha * tf.reduce_sum(predictions * (1 - labels))
        #          + beta * tf.reduce_sum((1 - predictions) * labels))

        # return -(truepos + smooth) / (truepos + smooth + fp_and_fn)
        # print(y_true)
        # print(y_pred)
        # keras.backend.print_tensor(y_pred)
        # y_true = tf.cast(y_true, 'int32')
        # y_pred = tf.cast(y_pred, 'int32')
        # print(y_true)
        # print(y_pred)
        # # y_true = keras.backend.flatten(y_true)
        # # y_pred = keras.backend.flatten(y_pred)
        # smooth = 1.
        # # intersection = keras.backend.sum(y_true, y_pred)
        # intersection = tf.reduce_sum(y_true * y_pred)
        # # print(intersection)
        # l = tf.reduce_sum(y_true * y_true)
        # r = tf.reduce_sum(y_pred * y_pred)
        # # print(f'l: {l}')
        # # print(f'r: {r}')
        # dice = 1 - ((2 * intersection + smooth) / (l + r + smooth))
        # dice = tf.reduce_mean(dice, name='dice_coef')
        # return 1 - dice
        # return 1 - (
        #     (2. * intersection + smooth) /
        #     (keras.backend.sum(y_true) + keras.backend.sum(y_pred) + smooth))

    with dist_strat.scope():
        model.compile(
            optimizer=paramset['optimizer'],
            # optimizer='SGD',
            loss='binary_crossentropy',
            metrics=utils.parse_metrics(config['hyperparameters']['metrics']),
            # run_eagerly=True
        )

    class_weight = compute_class_weight('balanced',
                                        classes=np.unique(y_train),
                                        y=y_train)
    history = model.fit(
        X_train,
        y_train,
        batch_size=paramset['batch_size'],
        # batch_size=len(X_train),
        class_weight=class_weight,
        epochs=50,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        shuffle=True)

    # logger.info(model.evaluate(X_test, y_test))
    y_pred = model.predict(X_test)
    # y_pred = np.rint(y_pred)
    # logger.debug(np.asarray(np.unique(y_pred, return_counts=True)).tolist())
    # logger.debug(np.asarray(np.unique(y_test, return_counts=True)).tolist())

    discrimination_threshold = 0.5
    tn, fp, fn, tp = utils.flatten(
        confusion_matrix(y_test, y_pred > discrimination_threshold))
    logger.info(
        f'tp:{tp},fp:{fp},tn:{tn},fn:{fn},acc:{(tp+tn)/len(y_test)}, tnr:{tn/(tn+fp)}'
    )

    logdir_figs = config['run_params']['logdir'] / 'figs'
    Path.mkdir(logdir_figs, parents=True, exist_ok=True)
    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-model.png'

    keras.utils.plot_model(model, logdir_fig, show_shapes=True, rankdir='LR')
    if show:
        plt.axis('off')
        plt.imshow(mpimg.imread(logdir_fig))
        plt.show()

    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-cm.png'
    utils.plot_cm(y_test,
                  y_pred > discrimination_threshold,
                  logdir_fig,
                  paramset,
                  p=0.5)

    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-roc.png'
    utils.plot_roc(y_test, y_pred, logdir_fig, paramset)

    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-metrics.png'
    utils.plot_metrics(history,
                       paramset['optimizer'],
                       'binary_crossentropy',
                       logdir_fig,
                       show=show)

    metrics = {
        metric: max(history.history[f'val_{metric}'])
        for metric in config['hyperparameters']['metrics']
    }

    return model, metrics


def tune_hparams(X, y, config, dist_strat):
    logger = config['run_params']['logger']
    param_grid = ParameterGrid({
        h.name: h.domain.values
        for h in config['hyperparameters']['hparams']
    })

    # Write hyperparameters to TensorBoard
    with tf.summary.create_file_writer(str(
            config['run_params']['logdir'])).as_default():
        hp.hparams_config(
            hparams=config['hyperparameters']['hparams'],
            metrics=[
                hp.Metric(metric)
                for metric in config['hyperparameters']['metrics']
            ],
        )
    logger.debug("Wrote hyperparameters")

    # Run each hyperparameter configuration
    best_hparams = {}
    best_tune_metric_value = 0
    for session_num, paramset in enumerate(param_grid):
        run_name = f'run={session_num+1}'
        logger.info(f'--- Starting trial: {run_name}/{len(param_grid)}')
        logger.info(paramset)

        # Run k-fold cross-validation on paramset
        # no need to shuffle because of shuffle in preprocessing
        paramset_metrics = {
            metric: []
            for metric in config['hyperparameters']['metrics']
        }
        for fold_num, (train_idx, test_idx) in enumerate(
                StratifiedKFold(config['run_params']['cv_n_splits']).split(
                    X, y)):
            fold_name = f'fold={fold_num+1}'
            logger.info(
                f"--- Starting cross-validation: {fold_name}/{config['run_params']['cv_n_splits']}"
                f" of {run_name}/{len(param_grid)}")
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)

            # Define the Keras callbacks
            logdir_cv = config['run_params']['logdir'] / \
                            f'{config["run_params"]["cv_n_splits"]}-fold-cv' / \
                            f'{run_name}-' \
                            f'batch_size={paramset["batch_size"]}-' \
                            f'num_units={paramset["num_units"]}-' \
                            f'optimizer={paramset["optimizer"]}-' \
                            f'{fold_name}-' \
                            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            Path.mkdir(logdir_cv, parents=True, exist_ok=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir=str(logdir_cv),
                                                      histogram_freq=1)
            earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=0.0001,
                                                          patience=5,
                                                          verbose=1)
            callbacks = [tensorboard, earlystopping]

            # Resets all state generated previously by Keras
            keras.backend.clear_session()

            _, metrics = train_test_model(run_name,
                                          X_train,
                                          y_train,
                                          X_test,
                                          y_test,
                                          paramset,
                                          callbacks,
                                          config,
                                          dist_strat,
                                          show=False)
            for metric in metrics.keys():
                paramset_metrics[metric].append(metrics[metric])

        # Collect paramset statistics
        paramset_stats = {
            metric: mean(paramset_metrics[metric])
            for metric in paramset_metrics.keys()
        }

        # Write paramset run metrics to tensorboard
        logdir_paramset = config['run_params']['logdir'] / \
                            f'hparam_tuning' / \
                            f'{run_name}-' \
                            f'batch_size={paramset["batch_size"]}-' \
                            f'num_units={paramset["num_units"]}-' \
                            f'optimizer={paramset["optimizer"]}-' \
                            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        Path.mkdir(logdir_paramset, parents=True, exist_ok=True)
        hparams = {
            h: paramset[h.name]
            for h in config['hyperparameters']['hparams']
        }
        with tf.summary.create_file_writer(str(logdir_paramset)).as_default():
            # record the values used in this trial
            hp.hparams(hparams, trial_id=f'{run_name}_{paramset}')
            for metric in paramset_stats.keys():
                tf.summary.scalar(metric, paramset_stats[metric], step=1)

        # Compare to previous best paramset and replace it when better
        tune_metric_mean = paramset_stats[config['hyperparameters']
                                          ['tune_metric']]
        if abs(tune_metric_mean) > abs(best_tune_metric_value):
            best_hparams = paramset
            best_tune_metric_value = tune_metric_mean

    return best_hparams


def train_final_model(X, y, paramset, config, dist_strat):
    final_model_path = config['run_params']['logdir'] / 'final'
    run_name = 'run_final'

    # Split data
    test_size = 0.2
    data_len = len(y)
    split_len = int(test_size * data_len)
    train_split_idx = data_len - 2 * split_len
    test_split_idx = data_len - split_len

    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:test_split_idx], y[
        train_split_idx:test_split_idx]
    X_test, y_test = X[test_split_idx:], y[test_split_idx:]

    # Define the Keras callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir=str(final_model_path),
                                              histogram_freq=1)
    ckpt_path = str(final_model_path / 'saved_model')
    checkpoint = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor=f"val_{config['hyperparameters']['tune_metric']}",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch')
    callbacks = [
        tensorboard,
        checkpoint,
    ]

    # Resets all state generated previously by Keras
    keras.backend.clear_session()

    model, _ = train_test_model(run_name,
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                paramset,
                                callbacks,
                                config,
                                dist_strat,
                                show=False)

    # Save in .h5 format too for visualization compatibility
    model.save(Path(ckpt_path) / 'saved_model.h5', save_format='h5')
    # model = keras.models.load_model(str(Path(ckpt_path) / 'saved_model.h5'))

    logdir_figs = config['run_params']['logdir'] / 'figs'
    Path.mkdir(logdir_figs, parents=True, exist_ok=True)
    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'\
                               f'-{run_name}-test-model.png'
    keras.utils.plot_model(model, logdir_fig, show_shapes=True, rankdir='LR')

    y_pred = model.predict(X_test)
    logdir_figs = config['run_params']['logdir'] / 'figs'
    Path.mkdir(logdir_figs, parents=True, exist_ok=True)
    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-test-cm.png'
    utils.plot_cm(y_test, y_pred > 0.5, logdir_fig, paramset, p=0.5)

    logdir_fig = logdir_figs / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}-test-roc.png'
    utils.plot_roc(y_test, y_pred, logdir_fig, paramset)

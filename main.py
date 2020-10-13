import logging
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import ParameterGrid, KFold, train_test_split
from sklearn.utils import shuffle
import generate_dataset as gd
import preprocess_data as ppd
import run
import utils


def main():
    # Load configurations
    config = utils.load_config()

    # Initialize logger
    logger = tf.get_logger()

    # default style: %(levelname)s:%(name)s:%(message)s
    logger.handlers[0].setFormatter(logging.Formatter(
        "({asctime}) {levelname:>8}: {message}", style='{'))
    logger.setLevel(config['loglevel'])
    logger.log(logger.getEffectiveLevel(), '<--- Effective log level')

    logger.info('Loaded configurations')
    logger.debug(config)

    # clear logs
    if Path('logs').exists():
        shutil.rmtree('logs')
    logger.info('Cleared previous logs')

    # Enable to find out which devices your operations and tensors are assigned to
    try:
        tf.debugging.set_log_device_placement(config['log_device_placement'])
        logger.debug(f'Log device placement: {config["log_device_placement"]}')
    except RuntimeError:
        logger.warning('Log device placement: ', 'failed to set')

    # generate dataset
    if not config['data_src'].exists():
        # TODO: implement cloud storage e.g. Google Drive
        logger.warning('Saving data to local storage')
        gd.generate_dataset(
            n=config['data_n'],
            seed=config['seed'],
            dest=config['data_src'],
            logger=logger
            )
    else:
        logger.info('Skipping data generation, file already exists: %s', config['data_src'])

    # preprocess data
    X, y = ppd.preprocess_data(
        config,
        plot=False,
        logger=logger
        )
    logger.info("Preprocessed data successfully")

    # write hyperparameters
    with tf.summary.create_file_writer(str(config['logdir'])).as_default():
        hp.hparams_config(
            hparams=config['hyperparameters']['hparams'],
            metrics=[
                hp.Metric(config['hyperparameters']['metric_accuracy'], display_name='Accuracy')],
        )
    logger.debug("Wrote hyperparameters")

    dist_strat = utils.dist_strategy(logger=logger)
    logger.debug(f"Selected distribution strategy {dist_strat}")

    param_grid = ParameterGrid(
        {h.name: h.domain.values for h in config['hyperparameters']['hparams']})

    best_paramset = {}
    best_avg_acc = 0.0
    for session_num, paramset in enumerate(param_grid):
        run_name = "run=%d" % session_num
        logger.info('--- Starting trial: %s' % run_name)
        logger.info(paramset)

        logdir_paramset = config['logdir'] / \
                            f'hparam_tuning' / \
                            f'{run_name}-' \
                            f'batch_size={paramset["batch_size"]}-' \
                            f'num_units={paramset["num_units"]}-' \
                            f'optimizer={paramset["optimizer"]}-' \
                            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        Path.mkdir(logdir_paramset, parents=True, exist_ok=True)

        # Run k-fold cross-validation
        # no need to shuffle because of shuffle in preprocessing
        # TODO decide simple vs stratified sampling
        accuracies = []
        for fold, (train_idx, test_idx) in enumerate(KFold(config['cv_n_splits']).split(X, y)):
            logger.info(f'--- Starting cross-validation fold: {fold+1}/{config["cv_n_splits"]}')
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Define the Keras callbacks
            logdir_cv = config['logdir'] / \
                            f'{config["cv_n_splits"]}-fold-cv' / \
                            f'{run_name}-' \
                            f'batch_size={paramset["batch_size"]}-' \
                            f'num_units={paramset["num_units"]}-' \
                            f'optimizer={paramset["optimizer"]}-' \
                            f'fold={fold+1}-' \
                            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            Path.mkdir(logdir_cv, parents=True, exist_ok=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir=str(logdir_cv), histogram_freq=1)
            ckpt_path = str(logdir_cv / 'weights.ckpt')
            
            hparams = {h: paramset[h.name] for h in config['hyperparameters']['hparams']}
            # hyperparameters = hp.KerasCallback(_logdir, hparams)
            earlystopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=5)
            callbacks = [
                tensorboard,
                # checkpoint,
                # hyperparameters,
                earlystopping
                ]

            # Resets all state generated previously by Keras
            keras.backend.clear_session()

            accuracy = run.train_test_model(run_name,
                                            X_train,
                                            y_train,
                                            X_test,
                                            y_test,
                                            paramset,
                                            callbacks,
                                            dist_strat,
                                            show=False)
            accuracies.append(accuracy)
        avg_acc = sum(accuracies)/len(accuracies)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_paramset = paramset
        logger.debug(accuracies)
        logger.debug(avg_acc)

        # Write paramset run metrics to tensorboard
        with tf.summary.create_file_writer(str(logdir_paramset)).as_default():
            # record the values used in this trial
            hp.hparams(hparams, trial_id=f'{run_name}_{paramset}') # pylint: disable=line-too-long
            tf.summary.scalar(config['hyperparameters']['metric_accuracy'], avg_acc, step=1)

            # # parallel coordinates test
            # pcdf = pd.DataFrame(paramset, index=[0])
            # pcdf['best_val_accuracy'] = accuracy
            # print(pcdf)
            # pd.plotting.parallel_coordinates(pcdf, 'optimizer')
            # plt.show()
            # optcol = pcdf.pop('optimizer')
            # # bscol = pcdf.pop('batch_size')
            # from yellowbrick.features import Rank2D
            # pcdf = pcdf.to_numpy()
            # visualizer = Rank2D(algorithm="pearson")
            # visualizer.fit_transform(pcdf)
            # visualizer.draw()
            # yellowbrick.features.parallel_coordinates(pcdf, bscol)
            # opt = 1 if optcol[0] == 'adam' else 0
            # visualizer = yellowbrick.features.parallel_coordinates(
                # X=pcdf,
                # y=optcol,
                # features=list(pcdf.columns),
                # classes=['adam', 'sgd'])

    # best_paramset = {'batch_size': 100, 'num_units': 50, 'optimizer': 'adam'}
    X, y = shuffle(X, y, random_state=config['seed'])
    X_val, X = X[int(0.8 * len(X)):], X[:int(0.8 * len(X))]
    y_val, y = y[int(0.8 * len(y)):], y[:int(0.8 * len(y))]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config['seed'],
        shuffle=True)

    # Define the Keras callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir=str(config['logdir']), histogram_freq=1)
    ckpt_path = str(config['logdir'] / 'weights.ckpt')
    checkpoint = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    earlystopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5)
    callbacks = [
        tensorboard,
        checkpoint,
        # hyperparameters,
        # earlystopping
        ]

    # Resets all state generated previously by Keras
    keras.backend.clear_session()

    run.train_test_model(
        'run_final',
        X_train,
        y_train,
        X_test,
        y_test,
        best_paramset,
        callbacks,
        dist_strat,
        show=False
        )

    model = keras.models.load_model(ckpt_path)
    logger.info(model.evaluate(X_val, y_val))
    y_pred = model.predict(X_val)
    y_pred = np.rint(y_pred)

    correct = 0
    incorrect = 0
    for pred, val in zip(y_pred, y_val):
        if pred == val:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct / (correct + incorrect)
    logger.info(accuracy)

if __name__ == '__main__':
    main()

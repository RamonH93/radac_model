import logging
from pathlib import Path
from datetime import datetime
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold
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
        data_src=config['data_src'],
        seed=config['seed'],
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

    # TODO simple vs stratified sampling.
    cv_n_splits = 5
    kf = KFold(
        n_splits=cv_n_splits,
        shuffle=True,
        random_state=config['seed']
    )
    # Making sure to use the same folds for each paramset later
    train_idxs = []
    test_idxs = []
    for train_idx, test_idx in kf.split(X, y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)

    param_grid = ParameterGrid(
        {h.name: h.domain.values for h in config['hyperparameters']['hparams']})
    for session_num, paramset in enumerate(param_grid):
        run_name = "run-%d" % session_num
        logger.info('--- Starting trial: %s' % run_name)
        logger.info(paramset)

        # Define the Keras callbacks
        _logdir = config['logdir'] / f'{run_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}-{paramset["optimizer"]}' # pylint: disable=line-too-long
        Path.mkdir(_logdir, parents=True, exist_ok=True)
        tensorboard = keras.callbacks.TensorBoard(log_dir=str(_logdir), histogram_freq=1)
        # tensorboard = keras.callbacks.TensorBoard(log_dir=str(_logdir))
        cp_path = str(_logdir / 'weights.ckpt')
        checkpoint = keras.callbacks.ModelCheckpoint(
            cp_path,
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch'
        )
        hparams = {h: paramset[h.name] for h in config['hyperparameters']['hparams']}
        # hyperparameters = hp.KerasCallback(_logdir, hparams)
        # earlystopping = keras.callbacks.EarlyStopping(
        #     monitor='val_accuracy',
        #     min_delta=0.0001,
        #     patience=10)
        callbacks = [
            tensorboard,
            checkpoint,
            # hyperparameters,
            # earlystopping
            ]

        # Resets all state generated previously by Keras
        keras.backend.clear_session()

        # Run k-fold cross-validation
        accuracies = []
        for i in range(cv_n_splits):
            X_train = X[train_idxs[i]]
            y_train = y[train_idxs[i]]
            X_test = X[test_idxs[i]]
            y_test = y[test_idxs[i]]

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
        logger.debug(accuracies)
        logger.debug(avg_acc)

        # Write paramset accuracy score to tensorboard
        with tf.summary.create_file_writer(str(_logdir)).as_default():
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

main()

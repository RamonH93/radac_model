import logging
import shutil
from pathlib import Path

import tensorflow as tf

import explainers as exp
import generate_dataset as gd
import preprocess_data as ppd
import run
import utils


def main():
    # Initialize logger
    logger = tf.get_logger()

    # default style: %(levelname)s:%(name)s:%(message)s
    logger.handlers[0].setFormatter(
        logging.Formatter("({asctime}) {levelname:>8}: {message}", style='{'))

    # Load configurations
    config = utils.load_config()
    config['logger'] = logger
    logger.info('Loaded configurations')

    # Set loglevel
    logger.setLevel(config['loglevel'])
    logger.log(logger.getEffectiveLevel(), '<--- Effective log level')

    # clear logs
    if Path('logs').exists():
        shutil.rmtree('logs')
    logger.info('Cleared previous logs')

    # Log device placement to find out which devices your operations and tensors are assigned to
    try:
        tf.debugging.set_log_device_placement(config['log_device_placement'])
        logger.debug(f'Log device placement: {config["log_device_placement"]}')
    except RuntimeError:
        logger.warning('Log device placement: ', 'failed to set')

    # Select CPU/GPU/TPU distribution strategy
    dist_strat = utils.dist_strategy(logger=logger)
    logger.debug(f"Selected distribution strategy {dist_strat}")

    # Generate or load dataset
    if not config['data_src'].exists():
        # TODO: implement cloud storage e.g. Google Drive
        logger.warning('Saving data to local storage')
        gd.generate_dataset(n=config['data_n'],
                            seed=config['seed'],
                            dest=config['data_src'],
                            logger=logger)
    else:
        logger.info('Skipping data generation, file already exists: %s',
                    config['data_src'])

    # Pre-process data
    X, y = ppd.preprocess_data(config, plot=False)
    logger.info("Preprocessed data successfully")

    # Hyperparameter tuning
    best_hparams, best_avg_acc = run.tune_hparams(X, y, config, dist_strat)
    logger.info(
        f"Found best paramset: {best_hparams} with accuracy: {best_avg_acc}")

    # Train final model
    # best_hparams = {'batch_size': 100, 'num_units': 50, 'optimizer': 'adam'}
    run.train_final_model(X,
                          y,
                          paramset=best_hparams,
                          config=config,
                          dist_strat=dist_strat)

    # Explain instance exp_n using LIME
    exp.lime_explainer(X, y, config)


if __name__ == '__main__':
    main()

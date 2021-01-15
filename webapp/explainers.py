from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
from sklearn.utils import shuffle
from tensorflow import keras

import utils


# Explain instance exp_n using LIME
def exp_daemon(gp):
    # Fixes rare memory error
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    params = gp.get_params()
    de = params['de']
    me = params['me']
    config = params['config']
    logger = config['run_params']['logger']
    data_src = config['run_params']['data_src']
    X = params['X']
    y = params['y']
    final_model_path = str(
        Path(__file__).parent / 'saved_models' /
        'titanic') if str(data_src) == 'train_titanic.csv' else str(
            Path(__file__).parent / 'saved_models' / 'amazon')
    tf.keras.utils.get_custom_objects().update(
        {'MCC': utils.MatthewsCorrelationCoefficient})
    model = keras.models.load_model(final_model_path, compile=True)

    X, y = shuffle(X, y, random_state=config['debugging']['seed'])

    test_size = 0.2
    data_len = len(y)
    split_len = int(test_size * data_len)
    train_split_idx = data_len - 2 * split_len
    test_split_idx = data_len - split_len
    X_train, _ = X[:train_split_idx], y[:train_split_idx]
    # X_val, y_val = X[train_split_idx:test_split_idx], y[train_split_idx:test_split_idx]
    X_test, y_test = X[test_split_idx:], y[test_split_idx:]

    # logger.debug(f'len(data): {data_len}')
    # logger.debug(f'split_len: {split_len}')
    # logger.debug(f'train_split_idx: {train_split_idx}')
    # logger.debug(f'test_split_idx: {test_split_idx}')
    # logger.debug(f'len(y_train): {len(y_train)}')
    # logger.debug(f'len(y_val): {len(y_val)}')
    # logger.debug(f'len(y_test): {len(y_test)}')

    logger.info(model.evaluate(X_test, y_test))
    y_pred = model.predict(X_test)
    # logger.debug(y_pred)
    y_pred = np.rint(y_pred)
    logger.debug(np.asarray(np.unique(y_pred, return_counts=True)).tolist())
    logger.debug(np.asarray(np.unique(y_test, return_counts=True)).tolist())

    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, true in zip(y_pred, y_test):
        if pred == true:
            if true == 1:
                tp += 1
            else:
                tn += 1
        else:
            if true == 1:
                fn += 1
            else:
                fp += 1
    logger.info(
        f'tp:{tp},fp:{fp},tn:{tn},fn:{fn},acc:{(tp+tn)/len(y_test)}, tnr:{tn/(tn+fp)}'
    )

    if str(data_src) == 'train_titanic.csv':
        explainer = LimeTabularExplainer(
            training_data=X_train,
            # feature_names=[
            #     'RESOURCE',
            #     'MGR_ID',
            #     'ROLE_ROLLUP_1',
            #     'ROLE_ROLLUP_2',
            #     'ROLE_DEPTNAME',
            #     'ROLE_TITLE',
            #     'ROLE_FAMILY_DESC',
            #     'ROLE_FAMILY',
            #     'ROLE_CODE'
            #     ],['MGR_ID', 'ROLE_ROLLUP_2', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE'
            feature_names=[
                'Pclass', 'Sex', 'SibSp', 'Parch', 'FareBin', 'AgeBin',
                'Embarked_C', 'Embarked_Q', 'Embarked_S'
            ],
            categorical_features=['Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S'],
            class_names=['Survived', 'NOT Survived'],
            random_state=config['debugging']['seed'],
        )
    else:
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=[f'f_{x}' for x in range(len(X_train[0]))],
            # feature_names=[
            #     'RESOURCE',
            #     'MGR_ID',
            #     'ROLE_ROLLUP_1',
            #     'ROLE_ROLLUP_2',
            #     'ROLE_DEPTNAME',
            #     'ROLE_TITLE',
            #     'ROLE_FAMILY_DESC',
            #     'ROLE_FAMILY',
            #     'ROLE_CODE'
            #     ],['MGR_ID', 'ROLE_ROLLUP_2', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE'
            class_names=['PERMIT', 'DENY'],
            random_state=config['debugging']['seed'],
        )

    logger.info("LIME Explainer Daemon ready.")

    # logger.debug(y_test[exp_n])
    # logger.debug(X_test[exp_n])
    # logger.debug(model.predict(np.array([X_test[exp_n]]))[0][0])
    # y_compl = np.ndarray(shape=(len(y_test), 2), dtype=np.int32)
    # for idx, lbl in enumerate(y_test):
    #     y_compl[idx] = np.array((lbl, 1 - lbl))
    # logger.debug(y_compl)

    def predict_with_opposite_class_preds(val):
        preds = model.predict(val)
        expanded = np.ndarray(shape=(len(preds), 2), dtype=np.float32)
        for idx, prd in enumerate(preds):
            expanded[idx] = np.array((prd[0], 1. - prd[0]))
        return expanded

    while True:
        de.wait()
        if params['to_explain'] == 'original':
            to_explain = params['instance']
        else:
            to_explain = params['mod']
        exp = explainer.explain_instance(to_explain,
                                         predict_with_opposite_class_preds)
        logger.debug(exp.as_list())
        # plt.show(exp.as_pyplot_figure())
        exp.save_to_file(
            str(Path(__file__).parent / 'explanations' / 'explanation.html'))
        logger.info(f'Generated explanation for instance {str(to_explain)}')
        me.set()
        me.clear()

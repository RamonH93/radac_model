from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras

import preprocess_data as ppd
import utils
from lime.lime_tabular import LimeTabularExplainer


# import logging
# class GlobalParams:
#     def __init__(self, params):
#         self.params = params

#     def set_params(self, params):
#         self.params = params

#     def get_params(self):
#         return self.params


# init_params = {}

# # Load configurations from file
# config = utils.load_config()
# init_params['config'] = config

# # Initialize logger
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logger.handlers[0].setFormatter(
#     logging.Formatter("({asctime}) {levelname:>8}: {message}", style='{'))
# logger.setLevel(config['debugging']['loglevel'])
# logger.log(logger.getEffectiveLevel(), '<--- Effective log level')
# init_params['logger'] = logger


# X, y = ppd.load_data(config, logger)
# init_params['X'] = X
# init_params['y'] = y
# exp_n = config['run_params']['exp_n']
# init_params['instance'] = ppd.instance_dict_to_df(X.iloc[exp_n].to_dict())
# init_params['mod'] = ppd.instance_dict_to_df(X.iloc[exp_n].to_dict())
# init_params['input_correct'] = True
# init_params['to_explain'] = 'original'
# init_params['feature_names'] = X.columns
# init_params['input_validation'] = {'correct':True, 'col': "", 'val': ""}

# gp = GlobalParams(init_params)

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
    logger = params['logger']
    data_src = config['run_params']['data_src']
    X, y, feature_names, label_names = ppd.preprocess_data(
        params['X'], params['y'], gp)
    final_model_path = str(
        Path(__file__).parent / 'saved_models' /
        'titanic') if str(data_src) == 'train_titanic.csv' else str(
            Path(__file__).parent / 'saved_models' / 'amazon')
    tf.keras.utils.get_custom_objects().update(
        {'MCC': utils.MatthewsCorrelationCoefficient})
    model = keras.models.load_model(final_model_path, compile=True)

    # X, y = shuffle(X, y, random_state=config['debugging']['seed'])

    # test_size = 0.2
    # data_len = len(y)
    # split_len = int(test_size * data_len)
    # train_split_idx = data_len - 2 * split_len
    # test_split_idx = data_len - split_len
    # X_train, _ = X[:train_split_idx], y[:train_split_idx]
    # # X_val, y_val = X[train_split_idx:test_split_idx], y[train_split_idx:test_split_idx]
    # X_test, y_test = X[test_split_idx:], y[test_split_idx:]

    # # logger.debug(f'len(data): {data_len}')
    # # logger.debug(f'split_len: {split_len}')
    # # logger.debug(f'train_split_idx: {train_split_idx}')
    # # logger.debug(f'test_split_idx: {test_split_idx}')
    # # logger.debug(f'len(y_train): {len(y_train)}')
    # # logger.debug(f'len(y_val): {len(y_val)}')
    # # logger.debug(f'len(y_test): {len(y_test)}')

    # logger.info(model.evaluate(X_test, y_test))
    # y_pred = model.predict(X_test)
    # # logger.debug(y_pred)
    # y_pred = np.rint(y_pred)
    # logger.debug(np.asarray(np.unique(y_pred, return_counts=True)).tolist())
    # logger.debug(np.asarray(np.unique(y_test, return_counts=True)).tolist())

    # tp, fp, tn, fn = 0, 0, 0, 0
    # for pred, true in zip(y_pred, y_test):
    #     if pred == true:
    #         if true == 1:
    #             tp += 1
    #         else:
    #             tn += 1
    #     else:
    #         if true == 1:
    #             fn += 1
    #         else:
    #             fp += 1
    # logger.info(
    #     f'tp:{tp},fp:{fp},tn:{tn},fn:{fn},acc:{(tp+tn)/len(y_test)}, tnr:{tn/(tn+fp)}'
    # )

    if str(data_src) == 'train_titanic.csv':
        cols = ['Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        categorical_indices = [np.where(feature_names == c)[0][0] for c in cols]
    else:
        categorical_indices = range(len(feature_names))

    explainer = LimeTabularExplainer(
        training_data=X,
        training_labels=y,
        feature_names=feature_names,
        categorical_features=categorical_indices,
        verbose=True,
        class_names=label_names,
        feature_selection='none',
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

    # from lime.submodular_pick import SubmodularPick
    # import pandas as pd

    # sp_obj = SubmodularPick(
    #     explainer,
    #     X,
    #     predict_with_opposite_class_preds,
    #     sample_size=20,
    #     num_features=5,
    #     num_exps_desired=10
    # )

    # logger.info("LIME SubmodularPick ready.")

    # # W=pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])
    # W_pick = pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.sp_explanations]).fillna(0)
    # W_pick['prediction'] = [this.available_labels()[0] for this in sp_obj.sp_explanations]

    # #Making a dataframe of all the explanations of sampled points
    # W=pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.explanations]).fillna(0)
    # W['prediction'] = [this.available_labels()[0] for this in sp_obj.explanations]
 
 
    # #Plotting the aggregate importances
    # np.abs(W.drop("prediction", axis=1)).mean(axis=0).sort_values(ascending=False).head(
    #     25
    # ).sort_values(ascending=True).plot(kind="barh")
    
    # #Aggregate importances split by classes
    # grped_coeff = W.groupby("prediction").mean()
    
    # grped_coeff = grped_coeff.T
    # grped_coeff["abs"] = np.abs(grped_coeff.iloc[:, 0])
    # grped_coeff.sort_values("abs", inplace=True, ascending=False)
    # grped_coeff.head(25).sort_values("abs", ascending=True).drop("abs", axis=1).plot(
    #     kind="barh",
    # )
    # plt.show()

    while True:
        de.wait()
        if params['to_explain'] == 'original':
            to_explain = ppd.process_row(params['instance'], gp)
        else:
            to_explain = ppd.process_row(params['mod'], gp)

        print(model.predict(np.array([to_explain]))[0])
        exp = explainer.explain_instance(to_explain,
                                         predict_with_opposite_class_preds)

        logger.info(
            f"LIME generated explanation for instance {str(to_explain)}.")
        logger.debug(exp.as_list())
        # plt.show(exp.as_pyplot_figure())
        exp.save_to_file(
            str(Path(__file__).parent / 'explanations' / 'explanation.html'))
        logger.info(f'Lime explanation saved to file.')

        me.set()
        me.clear()

# exp_daemon(gp)

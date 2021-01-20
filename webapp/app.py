import logging
import threading
import os
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd

import explainers as exp
import preprocess_data as ppd
import utils


class GlobalParams:
    def __init__(self, params):
        self.params = params

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params


init_params = {}

# Load configurations from file
config = utils.load_config()
init_params['config'] = config

# Initialize logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("({asctime}) {levelname:>8}: {message}", style='{'))
logger.setLevel(config['debugging']['loglevel'])
logger.log(logger.getEffectiveLevel(), '<--- Effective log level')
init_params['logger'] = logger


X, y = ppd.load_data(config, logger)
init_params['X'] = X
init_params['y'] = y
exp_n = config['run_params']['exp_n']
init_params['instance'] = ppd.instance_dict_to_df(X.iloc[exp_n].to_dict())
init_params['mod'] = ppd.instance_dict_to_df(X.iloc[exp_n].to_dict())
init_params['input_correct'] = True
init_params['to_explain'] = 'original'
init_params['feature_names'] = X.columns
init_params['input_validation'] = {'correct':True, 'col': "", 'val': ""}
# if 'titanic' in config['run_params']['data_src'].name:
#     init_params['feature_names'] = [
#         'Pclass', 'Sex', 'SibSp', 'Parch', 'FareBin', 'AgeBin', 'Embarked_C',
#         'Embarked_Q', 'Embarked_S'
#     ]
# else:
#     init_params['feature_names'] = [
#         'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
#         'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
#         'ROLE_CODE'
#     ]
# init_params['feature_names'] = [f'f_{x}' for x in range(len(X[0]))]

# Instantiate threading events to support waiting for tasks to complete
de = threading.Event() # Daemon event flag
me = threading.Event() # Main event flag
init_params['de'] = de
init_params['me'] = me

gp = GlobalParams(init_params)

# Spawn Explainer Daemon
process_name = f'App-{os.getpid()}'
thread_name = f'ExplainerDaemon-{np.random.randint(10000)}'
threading.Thread(target=exp.exp_daemon,
                 name=thread_name,
                 args=(gp, ),
                 daemon=True).start()
logger.info(f'{process_name} spawned {thread_name}')

#################################################################################
###### CREATE APP #######
#################################################################################

app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))


@app.before_first_request
def before_first_request():
    pass


@app.route("/")
def index():
    params = gp.get_params()
    data_src = params['config']['run_params']['data_src']
    return render_template('index.html', dataset=str(data_src))


@app.route("/rowselect", methods=["POST", "GET"])
def rowselect():
    params = gp.get_params()
    exp_n = params['config']['run_params']['exp_n']
    if request.method == 'POST':
        datadict = request.get_json()
        logger.debug('/rowselect ' + str(datadict))
        if 'sample' in datadict:
            exp_n = int(datadict['sample'])
            params['config']['run_params']['exp_n'] = exp_n
            params['instance'] = ppd.instance_dict_to_df(X.iloc[exp_n].to_dict())
            params['to_explain'] = 'original'
            gp.set_params(params)
    return render_template('rowselect.html', len=len(X), index=exp_n)


@app.route("/samplemod", methods=["POST", "GET"])
def samplemod():
    params = gp.get_params()
    config = params['config']
    exp_n = config['run_params']['exp_n']
    length = len(params['instance'])
    if len(params['instance']) == 1:
        length = len(params['instance'].values[0])
    # if request.method == 'GET':
    #     logger.debug("/samplemod GET " + str(exp_n) + str(params['instance']))
    if request.method == 'POST':
        datadict = request.get_json()
        # logger.debug('/samplemod POST ' + str(datadict))
        print(datadict)
        mod = ppd.instance_dict_to_df(datadict)
        params['mod'] = mod

        # Server side input validation
        input_validation = {'correct':True, 'col': "", 'val': ""}
        for col in mod:
            if mod[col][0] not in params['X'][col].unique().tolist():
                input_validation = {'correct':False, 'col': col, 'val': mod[col][0]}
                logger.error(f"[{col}] Entered invalid value {mod[col][0]}")

        params['to_explain'] = 'modified'
        params['input_validation'] = input_validation
        # logger.debug("/samplemod POST " + str(exp_n) + str(mod))
        gp.set_params(params)
    return render_template('samplemod.html',
                           rowidx=exp_n,
                           len=length,
                           instance=params['instance'],
                           instance_mod=params['mod'],
                           feature_names=params['feature_names'],
                           input_validation=params['input_validation'])


@app.route("/explanation")
def explanation():
    logger.debug('/explanation')

    de.set()  # send event to wake exp_daemon to generate new explanation
    de.clear()
    me.wait()  # wait for event that exp_daemon finished explanation generation
    return send_from_directory('explanations', filename='explanation.html')


@app.route("/explanation/<path:path>")
def reroute(path):
    return explanation()


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static',
                               filename='favicon.ico',
                               mimetype='image/vnd.microsoft.icon')



#################################################################################
###### LAUNCH APP ####### ***COMMENT AWAY IN PRODUCTION***
#################################################################################
app.run(debug=True, use_reloader=False)

# train LIME op geprocessde data
# TODO process rij naar model compatibel
# TODO doe voorspelling
# TODO geef explanations weer voor model compatibel
# TODO vervang explanation weergave met originele features
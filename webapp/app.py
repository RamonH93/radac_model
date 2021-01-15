import logging
import threading
import os
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
import numpy as np

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
de = threading.Event()
me = threading.Event()
init_params['de'] = de
init_params['me'] = me
config = utils.load_config()
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("({asctime}) {levelname:>8}: {message}", style='{'))
logger.setLevel(config['debugging']['loglevel'])
logger.log(logger.getEffectiveLevel(), '<--- Effective log level')
config['run_params']['logger'] = logger
init_params['config'] = config
X, y = ppd.preprocess_data(config, plot=False)
init_params['X'] = X
init_params['y'] = y
exp_n = config['run_params']['exp_n']
init_params['instance'] = X[exp_n]
init_params['mod'] = X[exp_n]
init_params['to_explain'] = 'original'
# init_params['feature_names'] = [
#     'Pclass', 'Sex', 'SibSp', 'Parch', 'FareBin', 'AgeBin', 'Embarked_C',
#     'Embarked_Q', 'Embarked_S'
# ]
init_params['feature_names']=[f'f_{x}' for x in range(len(X[0]))]
gp = GlobalParams(init_params)

process_name = f'App-{os.getpid()}'
thread_name = f'ExplainerDaemon-{np.random.randint(10000)}'
threading.Thread(target=exp.exp_daemon,
                 name=thread_name,
                 args=(gp, ),
                 daemon=True).start()
logger.info(f'{process_name} spawned {thread_name}')

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
            params['instance'] = params['X'][exp_n]
            params['to_explain'] = 'original'
            gp.set_params(params)
    return render_template('rowselect.html', len=len(X), index=exp_n)


@app.route("/samplemod", methods=["POST", "GET"])
def samplemod():
    params = gp.get_params()
    config = params['config']
    exp_n = config['run_params']['exp_n']
    logger.debug("/samplemod " + str(exp_n) + str(params['instance']))
    gp.set_params(params)
    if request.method == 'POST':
        datadict = request.get_json()
        logger.debug('/samplemod ' + str(datadict))
        mod = []
        for idx in range(len(datadict)):
            mod.append(int(datadict[str(idx)]))
        mod = np.array(mod)
        params['mod'] = mod
        params['to_explain'] = 'modified'
        logger.debug("/samplemod " + str(exp_n) + str(mod))
        gp.set_params(params)
    return render_template('samplemod.html',
                           rowidx=exp_n,
                           len=len(params['instance']),
                           instance=params['instance'],
                           instance_mod=params['mod'],
                           feature_names=params['feature_names'])


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


app.run(debug=True, use_reloader=False)

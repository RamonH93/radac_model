from csv import QUOTE_ALL
from datetime import datetime
# from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
# from pandas_profiling import ProfileReport
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from generate_pips import (
    CONFIDENTIALITY_LVLS,
    CLEARANCE_LVLS,
    FOLDER,
    SEED,
)

CONFIDENTIALITY_LVLS_DTYPE = CategoricalDtype(CONFIDENTIALITY_LVLS, ordered=True)
CLEARANCE_LVLS_DTYPE = CategoricalDtype(CLEARANCE_LVLS, ordered=True)
WEEKDAYS = range(5)

DTYPES = {
    'id': 'int64',                      #! drop
    'date': 'datetime64',               #? ohe day or bin week/weekend
    'time': 'datetime64',               #? ohe hour int or bin
    'request_location': 'category',     #* ohe
    'resource': 'string',               #! drop until policy
    'owner': 'category',                #* ohe
    'category': 'category',             #! drop until policy
    'resource_unit': 'category',        #! drop until policy
    'resource_department': 'category',  #! drop until policy
    'confidentiality_level':
    CONFIDENTIALITY_LVLS_DTYPE,         #? cat.codes
    'email': 'string',                  #* ohe
    'name': 'category',                 #! drop
    'age': 'int64',                     #? nothing/or bin
    'country': 'category',              #* ohe
    'city': 'category',                 #! drop until policy
    'company': 'category',              #! drop until policy
    'person_unit': 'category',          #! drop until policy
    'person_department': 'category',    #! drop until policy
    'job': 'category',                  #! drop until policy
    'phone': 'string',                  #! drop until policy
    'clearance_level':
    CLEARANCE_LVLS_DTYPE,               #? cat.codes
}

def prepare_request(request: pd.DataFrame):
    # df = pd.DataFrame(request).transpose()
    # print(df.info())
    # print(df.transpose())
    return request

def main():
    # person_exists_policy - 1: 28587, 0: 1413
    # resource_exists_policy - 1: 30000, 0: 0
    # location_policy - 1: 28481, 0: 1519
    # resource_in_unit_policy - 1: 28501, 0: 1499
    # owner_or_department_policy - 1: 27135, 0: 2865
    # weekday_policy - 1: 28493, 0: 1507
    # time_policy - 1: 28633, 0: 1367
    # company_policy - 1: 28442, 0: 1558
    # clearance_policy - 1: 28575, 0: 1425
    policy = 'company_policy'
    df = pd.read_csv(FOLDER / FOLDER / 'policy_denied_reqs' / f'denied_reqs_{policy}.csv').astype(DTYPES)
    # df = pd.read_csv(FOLDER / 'requests.csv').astype(DTYPES)
    ids = df['id'].values
    resources_df = pd.read_csv(FOLDER / 'requests.csv').astype(DTYPES)

    # print(df.dtypes)
    # print(df.iloc[0])

    # ProfileReport(df, minimal=True).to_file(FOLDER / 'requests.html')

    to_drop = [
        'id',
        'resource',
        'category',
        'resource_unit',
        'resource_department',
        'name',
        'city',
        'company',
        'person_unit',
        'person_department',
        'job',
        'phone',
    ]
    to_ohe = [
        'request_location',
        # 'owner',
        # 'email',
        'country',
    ]

    #! drop columns
    df.drop(to_drop, axis=1, inplace=True)

    # print(pd.DataFrame([df['clearance_level'], df['clearance_level'].cat.codes]).transpose().tail(10))

    #! preprocess date
    new_date_series = pd.Series(np.zeros(df['date'].size, dtype='int64'))
    new_resources_date_series = pd.Series(np.zeros(resources_df['date'].size, dtype='int64'))
    for i in range(new_date_series.size):
        #? as binned week and weekend days
        # new_date_series.iloc[i] = int(df['date'].iloc[i].weekday() in WEEKDAYS

        #? as ohe days of week
        new_date_series.iloc[i] = df['date'].iloc[i].weekday()
        if i == 0:
            to_ohe.append('date')
    
    for i in range(new_resources_date_series.size):
        #? as binned week and weekend days
        # new_date_series.iloc[i] = int(df['date'].iloc[i].weekday() in WEEKDAYS

        #? as ohe days of week
        new_resources_date_series.iloc[i] = resources_df['date'].iloc[i].weekday()
    df['date'] = new_date_series
    resources_df['date'] = new_resources_date_series

    #! preprocess time
    new_time_series = pd.Series(np.zeros(df['time'].size, dtype='int64'))
    for i in range(new_time_series.size):
        #? as seconds
        new_time_series.iloc[i] = (
            df['time'].iloc[i].hour * 60 * 60 +
            df['time'].iloc[i].minute * 60 +
            df['time'].iloc[i].second
        )
    df['time'] = new_time_series.astype('int64')

    #! preprocess email, owner
    print(f'{datetime.now()} Starting is_owner_series...')
    is_owner_series = pd.Series(np.zeros(df['email'].size, dtype='int64'))
    for i in range(is_owner_series.size):
        is_owner_series[i] = int(df['email'].iloc[i] == df['owner'].iloc[i])
    df.drop(['email', 'owner'], axis=1, inplace=True)
    df['is_owner'] = is_owner_series
    print(f'{datetime.now()} Finished is_owner_series.')

    df['confidentiality_level'] = resources_df['confidentiality_level'].cat.codes
    df['age'] = df['age']
    df['clearance_level'] = resources_df['clearance_level'].cat.codes

    print(f'{datetime.now()} Starting ohe...')
    for icol in to_ohe:
        print(f'{datetime.now()} Starting one hot encoding of {icol}...')
        ohe_orig_cols = pd.get_dummies(resources_df[icol], prefix=icol).columns
        one_hot = pd.get_dummies(df[icol], prefix=icol)
        for jcol in ohe_orig_cols:
            if jcol in one_hot.columns:
                df[jcol] = one_hot[jcol]
            else:
                df[jcol] = 0
        print(f'{datetime.now()} {icol}: {len(ohe_orig_cols)} columns.')
        df.drop(icol, inplace=True, axis=1)
    print(f'{datetime.now()} Finished: {len(df.columns)} columns.')
    print(df.columns)

    X = df.values
    y_a = pd.read_csv(FOLDER / 'labels.csv').iloc[ids-1]['action'].values
    y_r = pd.read_csv(FOLDER / 'labels.csv').iloc[ids-1]['riskscore'].values
    print(f'{datetime.now()} Started normalizing..')
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    y_a = [[y] for y in y_a]
    y_r = [[y] for y in y_r]
    y_r = min_max_scaler.fit_transform(y_r)
    print(f'{datetime.now()} Finished normalizing.')

    print(f'{datetime.now()} Started shuffling..')
    X, y_a, y_r = shuffle(X, y_a, y_r, random_state=SEED)
    print(f'{datetime.now()} Finished shuffling.')

    # np.savez(FOLDER / FOLDER / 'policy_denied_reqs' / f'denied_reqs_{policy}.npz', X=X, y_a=y_a, y_r=y_r)

    # df.to_csv(FOLDER / 'preprocessed.csv')
    # df.to_hdf(FOLDER / 'preprocessed.h5', key='requests', mode='w')
    print(f'{datetime.now()} Preprocessed successfully.')

if __name__ == '__main__':
    main()
    # df = pd.read_csv(FOLDER / 'requests.csv', index_col=0)
    # df['action'] = pd.read_csv(FOLDER / 'labels.csv', index_col=0)['action']
    # df.to_csv(FOLDER / 'requests_weka.csv', quoting=QUOTE_ALL, index=False)

from csv import QUOTE_ALL
from datetime import datetime
# from pathlib import Path

# import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
# from pandas_profiling import ProfileReport

from generate_pips import (
    FOLDER,
    CONFIDENTIALITY_LVLS,
    CLEARANCE_LVLS,
)

CONFIDENTIALITY_LVLS_DTYPE = CategoricalDtype(CONFIDENTIALITY_LVLS, ordered=True)
CLEARANCE_LVLS_DTYPE = CategoricalDtype(CLEARANCE_LVLS, ordered=True)
WEEKDAYS = range(5)

DTYPES = {
    'id': 'int64',                      #! drop
    'date': 'datetime64',               #? ohe day or bin week/weekend
    'time': 'datetime64',               #? ohe hour int or bin
    'request_location': 'category',     #* ohe
    'resource': 'string',               #! drop
    'owner': 'category',                #* ohe
    'category': 'category',             #! drop
    'resource_unit': 'category',        #! drop until policy
    'resource_department': 'category',  #! drop until policy
    'confidentiality_level':
    CONFIDENTIALITY_LVLS_DTYPE,         #? cat.codes
    'email': 'string',                  #* ohe
    'name': 'category',                 #! drop
    'age': 'int64',                     #? nothing/or bin
    'country': 'category',              #* ohe
    'city': 'category',                 #! drop
    'company': 'category',              #! drop
    'person_unit': 'category',          #! drop until policy
    'person_department': 'category',    #! drop until policy
    'job': 'category',                  #! drop until policy
    'phone': 'string',                  #! drop
    'clearance_level':
    CLEARANCE_LVLS_DTYPE,               #? cat.codes
}

def prepare_request(request: pd.DataFrame):
    # df = pd.DataFrame(request).transpose()
    # print(df.info())
    # print(df.transpose())
    return request

def main():
    df = pd.read_csv(FOLDER / 'requests.csv', index_col=0).astype(DTYPES)

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
        'owner',
        'email',
        'country',
    ]

    #! drop columns
    df.drop(to_drop, inplace=True, axis=1)

    # print(pd.DataFrame([df['clearance_level'], df['clearance_level'].cat.codes]).transpose().tail(10))

    #! preprocess date
    new_date_series = df['date'].copy()
    for i in range(len(new_date_series)):
        #? as binned week and weekend days
        # new_date_series.iloc[i] = int(df['date'].iloc[i].weekday() in WEEKDAYS

        #? as ohe days of week
        new_date_series.iloc[i] = df['date'].iloc[i].weekday()
        if i == 0:
            to_ohe.append('date')
    df['date'] = new_date_series.astype('category')

    #! preprocess time
    new_time_series = df['time'].copy()
    for i in range(len(new_time_series)):
        #? as seconds
        new_time_series.iloc[i] = (
            df['time'].iloc[i].hour * 60 * 60 +
            df['time'].iloc[i].minute * 60 +
            df['time'].iloc[i].second
        )
    df['time'] = new_time_series.astype('int64')
    df['confidentiality_level'] = df['confidentiality_level'].cat.codes
    df['age'] = df['age']
    df['clearance_level'] = df['clearance_level'].cat.codes

    print(f'{datetime.now()} Starting ohe...')
    for col in to_ohe:
        print(f'{datetime.now()} Starting one hot encoding of {col}...')
        one_hot = pd.get_dummies(df[col], prefix=col)
        df.drop(col, inplace=True, axis=1)
        df = df.join(one_hot)
    print(f'{datetime.now()} Finished: {len(df.columns)} columns.')

    df.to_hdf(FOLDER / 'preprocessed.h5', key='requests', mode='w')
    print(f'{datetime.now()} Preprocessed successfully.')

if __name__ == '__main__':
    main()
    # df = pd.read_csv(FOLDER / 'requests.csv', index_col=0).set_index('id')
    # df['action'] = pd.read_csv(FOLDER / 'labels.csv', index_col=0)['action']
    # df.to_csv(FOLDER / 'requests_weka.csv', quoting=QUOTE_ALL)

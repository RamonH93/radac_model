from pathlib import Path
from pandas_profiling import ProfileReport
from pandas.api.types import CategoricalDtype
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from datetime import datetime

from generate_pips import (
    FOLDER,
    CONFIDENTIALITY_LVLS,
    CLEARANCE_LVLS,
)

CONFIDENTIALITY_LVLS_DTYPE = CategoricalDtype(CONFIDENTIALITY_LVLS, ordered=True)
CLEARANCE_LVLS_DTYPE = CategoricalDtype(CLEARANCE_LVLS, ordered=True)

DTYPES = {
    'id': 'int64',
    'date': 'datetime64',
    'time': 'datetime64',
    'request_location': 'category',
    'resource': 'string',
    'owner': 'category',
    'category': 'category',
    'unit': 'category',
    'department': 'category',
    'confidentiality_level': CONFIDENTIALITY_LVLS_DTYPE,
    'email': 'string',
    'name': 'category',
    'age': 'int64',
    'country': 'category',
    'city': 'category',
    'company': 'category',
    'job': 'category',
    'phone': 'string',
    'clearance_level': CLEARANCE_LVLS_DTYPE,
}

def prepare_request(request: pd.DataFrame):
    df = pd.DataFrame(request).transpose()
    # print(df.info())
    # print(df.transpose())
    return df

def main():
    requests_df = pd.read_csv(FOLDER / 'requests.csv', index_col=0).astype(DTYPES)
    labels_df = pd.read_csv(FOLDER / 'labels.csv', index_col=0)

    # print(requests_df.astype(DTYPES)['confidentiality_level'].head(5))
    # print(requests_df.dtypes)
    # print(requests_df.iloc[0])

    # ProfileReport(requests_df, minimal=True).to_file(FOLDER / 'requests.html')
    requests_df.drop([
        'id',
        'time',
        'city',
        'phone',
    ], inplace=True, axis=1)
    print(f'{datetime.now()} Starting...')
    for col in requests_df.columns:
        print(f'{datetime.now()} Starting one hot encoding of {col}...')
        one_hot = pd.get_dummies(requests_df[col], prefix=col)
        requests_df.drop(col, inplace=True, axis=1)
        requests_df = requests_df.join(one_hot)
    print(f'{datetime.now()} Finished: {len(requests_df.columns)} columns.')

    requests_df.to_hdf(FOLDER / 'preprocessed.h5', key='stage', mode='w')
    print(f'{datetime.now()} Preprocessed successfully.')

    # for i in range(1):
    #     request = requests_df.iloc[i]
    #     req_complete = prepare_request(request)
    #     label = labels_df['action'].iloc[i]

if __name__ == '__main__':
    main()

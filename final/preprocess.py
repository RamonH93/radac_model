from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

from generate_pips import (
    CLEARANCE_LVLS,
    CONFIDENTIALITY_LVLS,
    FOLDER,
)
from policyinformationpoint import PolicyInformationPoint

FOLDER = Path('final')

def prepare_request(request: pd.DataFrame):
    pip = PolicyInformationPoint()
    # resource = request['resource']
    # resource_data = pip.get_resource_attributes(resource)
    df = pd.DataFrame(request).transpose()
    # df = pd.merge(request, resource_data).convert_dtypes()
    print(df.info())
    print(df.transpose())
    return df

def main():
    requests_df = pd.read_csv(FOLDER / 'requests.csv', index_col=0, nrows=5)
    labels_df = pd.read_csv(FOLDER / 'labels.csv', index_col=0, nrows=1)

    for i in range(1):
        request = requests_df.iloc[i]
        req_complete = prepare_request(request)
        label = labels_df['action'].iloc[i]

if __name__ == '__main__':
    main()

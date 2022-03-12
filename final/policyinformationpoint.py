from pathlib import Path

import pandas as pd

from generate_pips import FOLDER

FOLDER = Path('final')

class PolicyInformationPoint:
    def __init__(self) -> None:
        self.employees_df = pd.read_csv(FOLDER / 'finalfinal' / 'employees.csv', index_col=0)
        self.resources_df = pd.read_csv(FOLDER / 'finalfinal' / 'resources.csv', index_col=0)

    def get_employee_attributes(self, email: str) -> pd.DataFrame:
        df = self.employees_df.loc[self.employees_df['email'] == email]
        if df.shape[0] == 1:
            res = df
        else:
            res = None
        return res

    def get_resource_attributes(self, resource: str) -> pd.DataFrame:
        df = self.resources_df.loc[self.resources_df['resource'] == resource]
        if df.shape[0] == 1:
            res = df
        else:
            res = None
        return res

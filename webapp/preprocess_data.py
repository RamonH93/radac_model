import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_data(config, logger=None):
    if 'titanic' in config['run_params']['data_src'].name:
        df = pd.read_csv(config['run_params']['data_src'])
        y = df.pop('Survived')
        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
                inplace=True,
                axis=1)

    elif 'amazon' in config['run_params']['data_src'].name:
        df = pd.read_csv(config['run_params']['data_src'])
        y = df.pop('ACTION')
        df.drop([
            'MGR_ID', 'ROLE_ROLLUP_2', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
            'ROLE_CODE'
        ],
                inplace=True,
                axis=1)

    else:
        raise ValueError(f'no parser for {config["run_params"]["data_src"]}')

    # X, y = shuffle(df, y, random_state=config['debugging']['seed'])
    return df, y


def instance_dict_to_df(instance):
    mod = pd.DataFrame(instance, index=[0])

    for col in mod.columns:
        mod[col] = pd.to_numeric(mod[col], errors='ignore')
    mod = mod.convert_dtypes()
    return mod


def preprocess_data(X, y, gp, plot=False):
    params = gp.get_params()
    config = params['config']
    logger = params['logger']

    if 'amazon' in config['run_params']['data_src'].name:
        X, y = process_amazon(X, y, logger, plot)

    elif 'titanic' in config['run_params']['data_src'].name:
        X, y = process_titanic(X, y, logger, plot)

    else:
        raise ValueError(f'no parser for {config["run_params"]["data_src"]}')

    X, y = shuffle(X, y, random_state=config['debugging']['seed'])
    return X, y


def process_amazon(X, y, logger=None, plot=False):
    df = X.copy()

    # df.drop([
    #     'MGR_ID', 'ROLE_ROLLUP_2', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
    #     'ROLE_CODE'
    # ],
    #         inplace=True,
    #         axis=1)

    # if logger:
    #     logger.debug('\n' + str(df.info()))
    #     logger.debug('\n' + str(df.describe()))
    #     logger.debug('\n' + str(df.nunique()))
    #     logger.debug(df.isnull().sum())

    # y = df.pop('ACTION')
    if plot:
        y.value_counts().plot(kind='pie')
        plt.show()

    for col in df.columns:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df.drop(col, inplace=True, axis=1)
        df = df.join(one_hot)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
    # df = ohe.fit_transform(df.values)
    # print(df)
    # ProfileReport(df, minimal=True).to_file('amazon_drop_ohe.html')

    # TPUs cant handle uint8, GPU cant handle uint32
    df = df.astype(np.int32)
    y = y.astype(np.int32)
    if logger:
        logger.debug(df.dtypes)

    # logger.debug(df.value_counts('RESOURCE'))
    # logger.debug(df.value_counts('MGR_ID'))
    # logger.debug(df.value_counts('ROLE_ROLLUP_1'))
    # logger.debug(df.value_counts('ROLE_ROLLUP_2'))
    # logger.debug(df.value_counts('ROLE_DEPTNAME'))
    # logger.debug(df.value_counts('ROLE_TITLE'))
    # logger.debug(df.value_counts('ROLE_FAMILY_DESC'))
    # logger.debug(df.value_counts('ROLE_FAMILY'))
    # logger.debug(df.value_counts('ROLE_CODE'))

    X = df.values
    y = y.values

    return X, y


def process_titanic(X, y, logger=None, plot=False):
    df = X.copy()

    # ProfileReport(df).to_file('titanic.html')

    # y = df.pop('Survived')
    # if logger:
    #     logger.debug(y.head)

    # df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
    # if logger:
    #     logger.debug(df.head)
    #     logger.debug(df.dtypes)

    # if logger:
    #     logger.debug(df.isnull().sum())

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # if logger:
    #     logger.debug(df.isnull().sum())

    df['FareBin'] = pd.qcut(df['Fare'], 4)
    df['FareBin'] = df['FareBin'].cat.codes
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)
    df['AgeBin'] = df['AgeBin'].cat.codes
    df.drop(['Fare', 'Age'], inplace=True, axis=1)

    # if logger:
    #     logger.debug(df.shape)

    df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
    one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    # if logger:
    #     logger.debug(one_hot)
    df.drop('Embarked', inplace=True, axis=1)
    df = df.join(one_hot)

    # if logger:
    #     logger.debug(df.head)
    #     logger.debug(df.dtypes)
    #     logger.debug(df.info())

    # if logger:
    #     logger.debug('\n' + str(df.describe()))
    # ProfileReport(df.join(y)).to_file('titanic_preprocessed.html')

    X = df.to_numpy()

    # if logger:
    #     logger.debug(X)
    # X = StandardScaler().fit_transform(X)
    # if logger:
    #     logger.debug(X)

    y = y.to_numpy()

    return X, y


def process_row(row, gp):
    params = gp.get_params()
    config = params['config']

    if 'amazon' in config['run_params']['data_src'].name:
        processed_row = process_row_amazon(row, gp)

    elif 'titanic' in config['run_params']['data_src'].name:
        processed_row = process_row_titanic(row, gp)

    else:
        raise ValueError(f'no parser for {config["run_params"]["data_src"]}')

    return processed_row


def process_row_amazon(row, gp):
    params = gp.get_params()
    origX = params['X']
    rowdf = row.copy()

    for icol in origX.columns:
        orig_ohe_cols = pd.get_dummies(origX[icol], prefix=icol).columns
        one_hot = pd.get_dummies(rowdf[icol], prefix=icol)
        for jcol in orig_ohe_cols:
            rowdf[jcol] = 0
        for jcol in one_hot:
            rowdf[jcol] = one_hot[jcol]
        rowdf.drop(icol, inplace=True, axis=1)

    # TPUs cant handle uint8, GPU cant handle uint32
    rowdf = rowdf.astype(np.int32)

    rowdf = rowdf.to_numpy()[0]

    return rowdf

def process_row_titanic(row, gp):
    params = gp.get_params()
    origX = params['X']
    rowdf = row.copy()

    # Determine fareBin
    fareBins = pd.qcut(origX['Fare'], 4).cat.categories
    rowdf['FareBin'] = fareBins.get_loc(rowdf['Fare'].values[0])

    # Determine ageBin
    ages = origX['Age'].copy()
    ages.fillna(origX['Age'].median(), inplace=True)
    ageBins = pd.cut(ages.astype(int), 5).cat.categories
    rowdf['AgeBin'] = ageBins.get_loc(rowdf['Age'].values[0])

    # Binary sexes
    rowdf['Sex'] = np.where(rowdf['Sex'] == 'male', 1, 0)

    # One-hot encoded embarkments
    orig_ohe_cols = pd.get_dummies(origX['Embarked'], prefix='Embarked').columns
    one_hot = pd.get_dummies(rowdf['Embarked'], prefix='Embarked')
    for col in orig_ohe_cols:
        rowdf[col] = 0
    for col in one_hot:
        rowdf[col] = one_hot[col]

    rowdf.drop(['Fare', 'Age', 'Embarked'], inplace=True, axis=1)
    rowdf = rowdf.to_numpy()[0]

    return rowdf

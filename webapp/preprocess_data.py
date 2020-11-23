import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def preprocess_data(config, plot=False) -> (np.ndarray, np.ndarray):
    logger = config['run_params']['logger']
    processed = False

    if 'data' in config['run_params']['data_src'].name:
        df = pd.read_csv(config['run_params']['data_src'])

        del df['ID']
        del df['date_time']
        del df['lat']
        del df['lng']
        del df['clearance']
        del df['confi_lvl']

        for col in df.columns:
            if df[col].dtype == object:
                one_hot = pd.get_dummies(df[col])
                del df[col]
                df = df.join(one_hot)

        # ProfileReport(df).to_file('pr_tmp.html')

        y = df.pop('access_granted')

        # if plot:
        #     df['clearance'].value_counts().plot(kind='pie')
        #     plt.show()
        #     df['confi_lvl'].value_counts().plot(kind='pie')
        #     plt.show()
        #     plt.matshow(df.filter([
        #         'Aaliyah Otto-Post',
        #         'clearance',
        #         'confi_lvl',
        #         'access_granted'
        #         ]).corr(), cmap=plt.get_cmap('afmhot_r'), interpolation='none')
        #     plt.gca().xaxis.tick_bottom()
        #     plt.gca().set_title('correlation_matrix')
        #     plt.show()

        # TPUs cant handle uint8, GPU cant handle uint32
        logger.debug(df.dtypes)
        logger.debug(df.isnull().sum())
        # df = df.astype(np.int32)
        # y = y.astype(np.int32)
        # if logger:
        #     logger.debug(df.dtypes)

        X = df.values
        y = y.values
        X, y = shuffle(X, y, random_state=config['debugging']['seed'])
        processed = True

    if 'amazon' in config['run_params']['data_src'].name:
        df = pd.read_csv(config['run_params']['data_src'])


        df.drop(['MGR_ID', 'ROLE_ROLLUP_2', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE'],
                inplace=True,
                axis=1)

        # if logger:
        #     logger.debug('\n' + str(df.info()))
        #     logger.debug('\n' + str(df.describe()))
        #     logger.debug('\n' + str(df.nunique()))
        #     logger.debug(df.isnull().sum())

        y = df.pop('ACTION')
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
        # df = df.astype(np.int32)
        # y = y.astype(np.int32)
        # if logger:
        #     logger.debug(df.dtypes)

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
        X, y = shuffle(X, y, random_state=config['debugging']['seed'])
        processed = True

    if 'titanic' in config['run_params']['data_src'].name:
        df = pd.read_csv(config['run_params']['data_src'])

        # ProfileReport(df).to_file('titanic.html')

        y = df.pop('Survived')
        # if logger:
        #     logger.debug(y.head)

        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
                inplace=True,
                axis=1)
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

        X, y = shuffle(X, y, random_state=config['debugging']['seed'])
        processed = True

    if not processed:
        raise ValueError(f'no parser for {config["run_params"]["data_src"]}')

    # expanded = np.ndarray(shape=(len(y), 2), dtype=np.int32)
    # for idx, prd in enumerate(y):
    #     expanded[idx] = np.array((prd, np.abs(1-prd)))

    return X, y

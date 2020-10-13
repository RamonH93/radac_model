import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def preprocess_data(config, plot=False, logger=None) -> (np.ndarray, np.ndarray):
    if str(config['data_src']) == 'data.csv':
        df = pd.read_csv(config['data_src'])

        del df['ID']
        del df['date_time']
        del df['lat']
        del df['lng']
        del df['clearance']
        del df['confi_lvl']

        # for col in df.columns:
        #     if df[col].dtype == object:
        #         one_hot = pd.get_dummies(df[col])
        #         del df[col]
        #         df = df.join(one_hot)

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
        if logger:
            logger.debug(df.dtypes)
            logger.debug(df.isnull().sum())
        # df = df.astype(np.int32)
        # y = y.astype(np.int32)
        # if logger:
        #     logger.debug(df.dtypes)

        X = df.values
        y = y.values
        X, y = shuffle(X, y, random_state=config['seed'])


    if str(config['data_src']) == 'train_amazon.csv':
        df = pd.read_csv(config['data_src'])
        df = df.head(32000)
        y = df.pop('ACTION')

        if plot:
            y.value_counts().plot(kind='pie')
            plt.show()

        # TPUs cant handle uint8, GPU cant handle uint32
        if logger:
            logger.debug(df.dtypes)
            logger.debug(df.isnull().sum())
        # df = df.astype(np.int32)
        # y = y.astype(np.int32)
        # if logger:
        #     logger.debug(df.dtypes)

        logger.debug(df.value_counts('RESOURCE'))
        logger.debug(df.value_counts('MGR_ID'))
        logger.debug(df.value_counts('ROLE_ROLLUP_1'))
        logger.debug(df.value_counts('ROLE_ROLLUP_2'))
        logger.debug(df.value_counts('ROLE_DEPTNAME'))
        logger.debug(df.value_counts('ROLE_TITLE'))
        logger.debug(df.value_counts('ROLE_FAMILY_DESC'))
        logger.debug(df.value_counts('ROLE_FAMILY'))
        logger.debug(df.value_counts('ROLE_CODE'))

        X = df.values
        y = y.values
        X, y = shuffle(X, y, random_state=config['seed'])


    if str(config['data_src']) == 'train_titanic.csv':
        df = pd.read_csv(config['data_src'])

        # ProfileReport(df).to_file('df.html')

        y = df.pop('Survived')
        logger.debug(y.head)

        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
        logger.debug(df.head)
        logger.debug(df.dtypes)

        logger.debug(df.isnull().sum())
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        logger.debug(df.isnull().sum())


        df['FareBin'] = pd.qcut(df['Fare'], 4)
        df['FareBin'] = df['FareBin'].cat.codes
        df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)
        df['AgeBin'] = df['AgeBin'].cat.codes
        df.drop(['Fare', 'Age'], inplace=True, axis=1)

        logger.debug(df.shape)

        df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
        one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
        logger.debug(one_hot)
        df.drop('Embarked', inplace=True, axis=1)
        df = df.join(one_hot)

        logger.debug(df.head)
        logger.debug(df.dtypes)
        logger.debug(df.info())

        logger.debug('\n' + str(df.describe()))
        # ProfileReport(df).to_file('X.html')

        X = df.to_numpy()
        logger.debug(X)
        X = StandardScaler().fit_transform(X)
        logger.debug(X)
        y = y.to_numpy()

        X, y = shuffle(X, y, random_state=config['seed'])


    if 'X' not in locals() or 'y' not in locals():
        raise ValueError(f'no parser for {config["data_src"]}')

    return X, y

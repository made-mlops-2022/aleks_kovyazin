import numpy as np
import pandas  as pd
import sys, os
sys.path.append('..')


def gen_data():
    print('current spot', os.getcwd())
    print('current files', os.listdir())
    df = pd.read_csv('dags/packages/data/raw/ds/data.csv')
    df_tmp = pd.DataFrame()
    for col in df.columns:
        single_series = pd.Series(np.random.choice(df[col].values, 10))
        single_series.name = col
        df_tmp = pd.concat([df_tmp, single_series], axis=1)
    df_new = pd.concat([df,df_tmp],axis=0)
    df_new.reset_index(drop=True, inplace=True)
    df_new[:-1].to_csv('dags/packages/data/raw/ds/data.csv')
    df_new[-1:].to_csv('dags/packages/data/raw/ds/target.csv')


if __name__ == '__main__':
    gen_data()
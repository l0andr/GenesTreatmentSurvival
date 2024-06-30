import pandas as pd
import numpy as np

def prepare_columns_for_model(df_formodel: pd.DataFrame,verbose: int = 0):
    column_for_dropping = []
    for column in df_formodel.columns:
        if df_formodel[column].dtype == object:
            df_formodel[column] = df_formodel[column].astype('category')
            df_formodel[column] = df_formodel[column].cat.codes

        if len(df_formodel[column].unique().tolist()) == 1:
            if verbose >= 1:
                print(f"{column} will be dropped (variance = 0)")
            column_for_dropping.append(column)
        if sorted(df_formodel[column].unique().tolist()) == [0, 1]:
            if verbose >= 1:
                print(f"{column} will be converted to boolean")
            df_formodel[column] = df_formodel[column].astype('bool')

        if len(df_formodel[column].unique().tolist()) < 5 and not df_formodel[column].dtype == bool:
            print(f"{column} will be converted to categorical")
            df_formodel[column] = df_formodel[column].astype('category')
            df_formodel[column] = df_formodel[column].cat.codes

    df_formodel.drop(columns=column_for_dropping, inplace=True)
    return df_formodel

def prepare_columns_for_analysis(df_formodel: pd.DataFrame,verbose: int = 0):
    column_for_dropping = []
    for column in df_formodel.columns:
        try:
            # replace 'none' to nan
            df_formodel[column] = df_formodel[column].replace('none', np.nan)
            if len(df_formodel[column].unique().tolist()) == 1:
                if verbose >= 1:
                    print(f"{column} will be dropped (variance = 0)")
                column_for_dropping.append(column)
                continue
            if df_formodel[column].dtype == bool:
                if verbose >= 1:
                    print(f"{column} remain boolean")
                continue
            #check if column have string values and continue if yes
            if df_formodel[column].dtype == object:
                #check if most of values in column are not numeric
                if df_formodel[column].str.isnumeric().sum() / len(df_formodel[column]) < 0.5:
                    if verbose >= 1:
                        print(f"{column} remain with type {df_formodel[column].dtype}")
                    df_formodel[column] = df_formodel[column].astype('str')
                else:
                    if verbose >= 1:
                        print(f"{column} will be converted to float64")
                    df_formodel[column] = df_formodel[column].astype(np.float64)
            if len(df_formodel[column].unique().tolist()) < 6:
                #check if column have only string values
                print(df_formodel[column].unique().tolist())
                if sorted(df_formodel[column].unique().tolist()) == [0, 1] or sorted(
                        df_formodel[column].unique().tolist()) == [False, True]:
                    if verbose >= 1:
                        print(f"{column} will be converted to boolean")
                    df_formodel[column] = df_formodel[column].astype('bool')
                else:
                    if verbose >= 1:
                        print(f"{column} will be converted to categorical with values {df_formodel[column].unique().tolist()}")
                    df_formodel[column] = df_formodel[column].astype('category')
            else:
                #check that pandas can convert column to numeric
                try:
                    df_formodel[column] = df_formodel[column].astype(np.float64)
                    if verbose >= 1:
                        print(f"{column} converted to float64")
                except:
                    if verbose >= 1:
                        print(f"{column} remians with type {df_formodel[column].dtype}")
        except Exception as e:
            print(f"Error with column {column} {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error with column {column} {e}")
    df_formodel.drop(columns=column_for_dropping, inplace=True)
    return df_formodel


def convert_to_float_and_normalize(df: pd.DataFrame):
    '''
       Function convert each column of datafarme by the following rules:
       step 1:
       1. If column have string format for each unique values will corespond unique float value from range 0-1
       2. if column have date format convert it to number of seconds from 1970-01-01 00:00:00
       3. if column in integer convert it to float
       step 2:
       On start of this step all columns should have float fromat.
       1. For each column calculate mean and std
       2. For each column calculate (x-mean)/std

       :param df: Dataframe with several columns of different type
       :return: df: Dataframe with normalized columns
       '''
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
        elif df[col].dtype == np.datetime64:
            df[col] = df[col].astype(np.int64) // 10 ** 9
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.float64)
    #for col in df.columns:
    #    df[col] = df[col].fillna(df[col].mean()).astype(np.float64)
    #    df[col] = (df[col] - df[col].mean()) / df[col].std()
    #    df[col] = (df[col] - df[col].min())
    #    df[col] = df[col] / df[col].max()
    return df

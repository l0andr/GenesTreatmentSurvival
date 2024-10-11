'''
This is script that used for manipulation with csv tables. It's expected some csv table with named columns on input
and some csv table with named columns on output.
following operations are supported:
    - delete specified columns
    - filter rows by specified column values

'''

import pandas as pd
import argparse
if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input', type=str, required=True, help='input csv file')
    argparse.add_argument('--input2', type=str, required=False, help='second input csv file to combine with first one',
                        default="")
    argparse.add_argument('--cohort_labels', type=str, required=False, help='Labels for different dataframe in join one',default="")
    argparse.add_argument('--output', type=str, required=True, help='output csv file')
    argparse.add_argument('--delete_columns', type=str, required=False, help='comma separated list of columns to delete')
    argparse.add_argument('--filter_column', type=str, required=False, help='column name to filter')
    argparse.add_argument('--filter_values', type=str, required=False, help='comma separated list of values to filter')
    args = argparse.parse_args()
    if args.cohort_labels:
        if len(args.cohort_labels.split(',')) != 2:
            raise Exception('cohort_labels should be comma separated list of two labels')
    df = pd.read_csv(args.input)
    if args.input2:
        df2 = pd.read_csv(args.input2)
        # create set of columns common for both datasets
        common_columns = set(df.columns).intersection(set(df2.columns))
        print(f"Will join two input dataset. Common columns is {[common_columns]} ")
        #set of columns that aree only in df
        only_df_columns = set(df.columns).difference(set(df2.columns))
        #set of columns that are only in df2
        only_df2_columns = set(df2.columns).difference(set(df.columns))
        #drop columns that are only in one dataset
        df = df.drop(columns=only_df_columns)
        df2 = df2.drop(columns=only_df2_columns)
        if args.cohort_labels:
            df['cohort'] = args.cohort_labels.split(',')[0]
            df2['cohort'] = args.cohort_labels.split(',')[1]
        #concat two datasets
        df = pd.concat([df,df2],axis=0)


    if args.delete_columns:
        df = df.drop(columns=args.delete_columns.split(','))
    if args.filter_column and args.filter_values:
        print("table_transform.py: filtering rows. filter_column: ", args.filter_column, " filter_values: ", args.filter_values)
        if args.filter_column not in df.columns:
            raise Exception(f'Column {args.filter_column} not found in input file')
        list_for_filter = args.filter_values.split(',')
        nvalue_before = len(df)
        df = df[df[args.filter_column].astype(str).isin(list_for_filter)]
        if len(df) == 0:
            raise Exception(f'No rows left after filtering')
        print(f"table_transform.py: Filtered {nvalue_before - len(df)} rows. remain {len(df)} rows")
    df.to_csv(args.output, index=False)

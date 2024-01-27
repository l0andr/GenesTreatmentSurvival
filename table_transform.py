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
    argparse.add_argument('--output', type=str, required=True, help='output csv file')
    argparse.add_argument('--delete_columns', type=str, required=False, help='comma separated list of columns to delete')
    argparse.add_argument('--filter_column', type=str, required=False, help='column name to filter')
    argparse.add_argument('--filter_values', type=str, required=False, help='comma separated list of values to filter')
    args = argparse.parse_args()

    df = pd.read_csv(args.input)
    if args.delete_columns:
        df = df.drop(columns=args.delete_columns.split(','))
    if args.filter_column and args.filter_values:
        if args.filter_column not in df.columns:
            raise Exception(f'Column {args.filter_column} not found in input file')
        df = df[df[args.filter_column].astype(str).isin(args.filter_values.split(','))]
        if len(df) == 0:
            raise Exception(f'No rows left after filtering')
    df.to_csv(args.output, index=False)

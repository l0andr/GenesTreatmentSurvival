import warnings
import argparse
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest

if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser(description="Perform initial survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("--min_cases", help="Minimal number of cases of mutations", type=int, default=3)
    parser.add_argument("--genes", help="Comma separated list of genes of interest ", type=str, default="")
    parser.add_argument("--factors", help="Comma separated list of factors of interest ", type=str, default="")
    parser.add_argument("--status_col", help="Column with status (event occur or not) ", type=str, default="status")
    parser.add_argument("--survival_time_col", help="Time until event ", type=str, default="survival_in_days")
    parser.add_argument("--patient_id_col", help="Patients id", type=str,
                        default="patient_id")

    parser.add_argument("--model_report", help="Path to model report", type=str,
                        default="cox_model_report.pdf")

    parser.add_argument("--verbose", help="Comma separated list of factors of interest ", type=int, default=2)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')

    #if factors are not specified, then all factors will be used
    #if genes are not specified, then all genes will be used
    args = parser.parse_args()
    min_cases = args.min_cases
    show = args.show
    if args.genes == "":
        genes = None
    else:
        genes = args.genes.split(',')
    if args.factors == "":
        factors = None
    else:
        factors = args.factors.split(',')

    df = pd.read_csv(args.input_csv)
    patient_id_col = args.patient_id_col
    status_col = args.status_col
    survival_time_col = args.survival_time_col
    keep_columns = [status_col, survival_time_col]
    ignore_columns = ['distance_from_mean', 'outlier', 'disease-free-status', 'disease-free-time', 'patient_id',
                      'status', 'survival_in_days','number_of_treatments','tmb_percentile_levels','age_level','imuno_duration_levels']
    genes_columns = [col for col in df.columns if col.startswith('gene_')]
    if genes is not None:
        genes_columns = [col for col in genes_columns if col.split('_')[1] in genes]
    # keep only such genes which have at least min_cases
    genes_columns = [col for col in genes_columns if df[col].sum() >= min_cases]

    factor_columns = []
    if factors is not None:
        factor_columns = factors
    else:
        factor_columns = [col for col in df.columns if not col.startswith('gene_') and
                          col not in keep_columns and
                          col not in ignore_columns and
                          not col.endswith('_date')]
    #remove response from factor columns
    factor_columns = [col for col in factor_columns if not col.startswith('response')]
    factor_columns = [col for col in factor_columns if not col.startswith('treatment_')]
    factor_columns = [col for col in factor_columns if not col.startswith('immunotherapy_')]
    factor_columns = [col for col in factor_columns if not col.startswith('immunotherapy_')]

    if args.verbose > 1:
        print(f"Model will be created based on following columns of data:")
        print(f"genes_columns:{genes_columns}")
        print(f"factor_columns:{factor_columns}")
        print(f"status_col:{status_col}")
        print(f"survival_time_col:{survival_time_col}")
        print(f"patient_id_col:{patient_id_col}")

    df_formodel = df[keep_columns + genes_columns + factor_columns]
    #list columns with NaN
    if args.verbose > 1:
        print(f"Columns with NaN values:")
        print(df_formodel.columns[df_formodel.isnull().any()])
    #drop columns where more than half values is NaN
    df_formodel = df_formodel.loc[:, df_formodel.isnull().mean() < .1]
    #list rows with NaN
    if args.verbose > 1:
        print(f"Rows with NaN values:")
        print(df_formodel[df_formodel.isnull().any(axis=1)])
    #drop rows with NaN
    df_formodel = df_formodel.dropna()
    if args.verbose > 1:
        print(f"Data after dropping NaN values:")
        print(df_formodel)
    min_group_size = 3
    #list of columns with 'time' in column name
    time_columns = [col for col in df_formodel.columns if 'time' in col]
    column_for_dropping = [] #['tmb_percentile_levels','immunotherapy_in_days','age','imuno_duration_levels','number_of_treatments','treatment','total_mutations']
    column_for_dropping.extend(time_columns)
    for column in (set(df_formodel.columns) - {status_col, survival_time_col}):
        skip_small_groups = False
        if df_formodel[column].dtype == object:
            df_formodel[column] = df_formodel[column].astype('category')
            df_formodel[column] = df_formodel[column].cat.codes
        if len(df_formodel[column].unique().tolist()) == 1:
            if args.verbose >= 1:
                print(f"{column} will be dropped (variance = 0)")
            column_for_dropping.append(column)
        if sorted(df_formodel[column].unique().tolist()) == [0, 1]:
            if args.verbose >= 1:
                print(f"{column} will be converted to boolean")
            df_formodel[column] = df_formodel[column].astype('bool')
            skip_small_groups = True
        if len(df_formodel[column].unique().tolist()) < 5 and not df_formodel[column].dtype == bool:
            print(f"{column} will be converted to categorical")
            df_formodel[column] = df_formodel[column].astype('category')
            df_formodel[column] = df_formodel[column].cat.codes
            skip_small_groups = True
        if skip_small_groups:
            for j in df_formodel[column].unique():
                if j is None or pd.isna(j):
                    continue
                if df_formodel[column].value_counts()[j] < min_group_size:
                    if args.verbose > 1:
                        print(f"Column {column} has too few cases of {j} {df_formodel[column].value_counts()[j]}, skip it")
                        column_for_dropping.append(column)

    df_formodel.drop(columns=column_for_dropping, inplace=True)
    print(df_formodel)
    # OneHotEncoder of all data except status and survival_time
    Xdf = df_formodel.drop(columns=[status_col, survival_time_col])
    ydf = df_formodel[[status_col,survival_time_col]]
    # convert ydf to numpy array
    aux = [(e1, e2) for e1, e2 in ydf.to_numpy()]
    yarray = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    Xdf_encod = OneHotEncoder().fit_transform(Xdf)
    train_test_split_ratio = 0.3
    if train_test_split_ratio > 0.0:
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xdf_encod, yarray, test_size=train_test_split_ratio, random_state=1)
    else:
        Xtrain = Xdf_encod
        ytrain = yarray
        Xtest = Xtrain
        ytest = ytrain
    rsf = RandomSurvivalForest(n_estimators=100, max_depth=4, random_state=0,oob_score=True)
    rsf.fit(Xtrain, ytrain)
    #Run cross validation of rsf model
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rsf, Xdf_encod, yarray, cv=4)
    if args.verbose > 1:
        print(f"Cross validated concordance index {scores.mean():.2f}  with a standard deviation of {scores.std():.2f}")

    if args.verbose > 1:
        if train_test_split_ratio > 0.0:
            print(f"Model was trained. Test concordance index: {rsf.score(Xtest, ytest)}")
        print(f"Model was trained. In sample concordance index: {rsf.score(Xtrain, ytrain)}")
    warnings.filterwarnings("ignore")

    result = permutation_importance(rsf, Xtest, ytest, n_repeats=400, random_state=2)
    factors = pd.DataFrame(
        {
            k: result[k]
            for k in (
            "importances_mean",
            "importances_std",
        )
        },
        index=Xtrain.columns,
    ).sort_values(by="importances_mean", ascending=False)
    factors["imp_std"] = (factors["importances_mean"] > factors["importances_std"])
    #sort by imp_std (reverse) and then importances_mean and then importances_std
    factors = factors.sort_values(by=["imp_std","importances_mean","importances_std"],ascending=[False,False,False])
    print(factors)
import pandas as pd
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List,Optional
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from lifelines.statistics import logrank_test,multivariate_logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from copy import deepcopy
def compute_median_survival_time(df:pd.DataFrame,column_name:str,
                                    status_column:str="Status",survival_in_days:str="Survival_in_days"):
    median_survival_time = {}
    confidence_intervals = {}
    for s in df[column_name].dropna().unique().tolist():
        mask_treat = df[column_name] == s
        kmf = KaplanMeierFitter()
        kmf.fit(df[survival_in_days][mask_treat], event_observed=df[status_column][mask_treat])
        confidence_intervals[s] = median_survival_times(kmf.confidence_interval_)

        median_survival_time[s] = {"median":kmf.median_survival_time_,
                                   "Conf. int.":[confidence_intervals[s]['KM_estimate_lower_0.95'].iloc[0],
                                       confidence_intervals[s]['KM_estimate_upper_0.95'].iloc[0]]}
    return median_survival_time

def compute_survival_p_values(df:pd.DataFrame,column_name:str,
                     status_column:str="Status",survival_in_days:str="Survival_in_days"):
    p_values = {}
    for s in df[column_name].dropna().unique().tolist():
        mask_treat = df[column_name] == s
        p_values[s] = logrank_test(df[status_column][mask_treat], df[status_column][~mask_treat],
                                   df[survival_in_days][mask_treat], df[survival_in_days][~mask_treat]).p_value
    return p_values

def plot_kaplan_meier(df_pu:pd.DataFrame,column_name:str,
                      status_column:str="Status",survival_in_days:str="Survival_in_days",plot_pvalues=True):

    fig = plt.figure()
    p_values={}
    diff_values = df_pu[column_name].dropna().unique().tolist()
    for s in diff_values:
        mask_treat = df_pu[column_name] == s
        time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
            df_pu[status_column][mask_treat],
            df_pu[survival_in_days][mask_treat],
            conf_type="log-log",
        )
        p_values[s] = logrank_test(df_pu[status_column][mask_treat], df_pu[status_column][~mask_treat],
                                   df_pu[survival_in_days][mask_treat], df_pu[survival_in_days][~mask_treat]).p_value
        if len(diff_values) > 1 and plot_pvalues:
            plt.step(time_treatment, survival_prob_treatment, where="post", label=f"{column_name} = {s} p-value = {p_values[s]:.5f} ")
        else:
            plt.step(time_treatment, survival_prob_treatment, where="post",
                     label=f"{column_name} = {s} ")
        plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")

    plt.ylim(0, 1)
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel(f"time $t$ ({survival_in_days})")
    plt.legend(loc="best")
    return fig

def plot_kaplan_meier2(df_pu:pd.DataFrame,column_name:str,
                      status_column:str="Status",survival_in_days:str="Survival_in_days",plot_pvalues=True):


    diff_values = df_pu[column_name].dropna().unique().tolist()

    fig, ax = plt.subplots(figsize=(18, 8),nrows=1,ncols=1,sharex=True)
    if not isinstance(ax,np.ndarray):
        ax = [ax]
    i = 0
    kmfs = []
    p_values = {}
    at_risk_lables = []
    for s in diff_values:
        mask_treat = df_pu[column_name] == s
        p_values[s] = logrank_test(df_pu[status_column][mask_treat], df_pu[status_column][~mask_treat],
                                   df_pu[survival_in_days][mask_treat], df_pu[survival_in_days][~mask_treat]).p_value
        i+=1
        ix = df_pu[column_name] == s
        kmf = KaplanMeierFitter()
        kmf.fit(df_pu[survival_in_days][ix],df_pu[status_column][ix], label=column_name+" = "+str(s) + f" p-value = {p_values[s]:.5f} " )
        kmf.plot_survival_function(ax=ax[0],ci_legend = True, at_risk_counts = False)
        at_risk_lables.append(f"{column_name} = {s}")
        kmfs.append(kmf)
    add_at_risk_counts(*kmfs,labels=at_risk_lables, ax=ax[0])

    ax[0].set_ylabel("est. probability of survival $\hat{S}(t)$")
    ax[0].set_xlabel(f"time $t$ (days)")
    ax[0].set_title(f"Kaplan-Meier survival estimates [{survival_in_days}]")
    plt.tight_layout()
    #plt.legend(loc="best")
    return fig

def expand_values_for_patients(initial_df:pd.DataFrame,list_of_expand_marks:List[str],patients_id_columns:str):
    cols_with_dot_or_dash = initial_df.columns[initial_df.isin(list_of_expand_marks).any()].tolist()
    initial_df.replace(list_of_expand_marks, np.nan, inplace=True)
    for col in cols_with_dot_or_dash:
        initial_df[col] = initial_df[col].fillna(initial_df.groupby(patients_id_columns)[col].transform('first'))
    return initial_df

def dataframe_normalization(df:pd.DataFrame):
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
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha=0.01, ties='breslow', n_iter=100, tol=1e-09, verbose=0)
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="Perform initial survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("-output_pdf", help="Transformed CSV file", type=str, required=True)
    parser.add_argument("--min_cases", help="Minimal number of cases of mutations", type=int, default=4)
    parser.add_argument("--genes", help="Comma separated list of genes of interest ", type=str, default="", required=True)
    parser.add_argument("--factors", help="Comma separated list of factors of interest ", type=str, default="",required=True)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')

    args = parser.parse_args()
    genes_of_interest = args.genes.split(',')
    factors_of_interst = args.factors.split(',')
    genes_treatment = args.genes.split(',')
    genes_prefix = 'gene_'
    min_number_of_cases_of_mutations = args.min_cases
    output_pdf = args.output_pdf
    df_clean = pd.read_csv(args.input_csv)
    selected_columns = [col for col in df_clean.columns if col.startswith('gene_')]
    filtered_df = df_clean.loc[:, selected_columns]
    non_zero_counts = filtered_df.apply(lambda x: (x != 0).sum())
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #sort non_zero_counts series by values
    non_zero_counts = non_zero_counts.sort_values(ascending=False)
    #filter out genes with less than min_number_of_cases_of_mutations
    non_zero_counts = non_zero_counts[non_zero_counts > min_number_of_cases_of_mutations]

    #plot non_zero_counts series as bar plot

    pp = PdfPages(output_pdf)

    #plt.show()
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(non_zero_counts.index, non_zero_counts.values, color='k', label='All genes')
    #select genes of interest from non_zero_counts series
    non_zero_counts_genes_of_interest = non_zero_counts[non_zero_counts.index.str.contains('|'.join(genes_of_interest))]
    if len(non_zero_counts_genes_of_interest) > 0:
        ax.bar(non_zero_counts_genes_of_interest.index, non_zero_counts_genes_of_interest.values,
               color='k', label='Genes of interets')
    ax.set_ylabel('Number of patients that have mutation in gene')
    ax.set_title('Number of present mutation in genes for whole dataset')
    ax.set_xticks(range(len(non_zero_counts)))

    #set xtecklabels without "gene_" prefix
    ax.set_xticklabels(non_zero_counts.index.str.replace('gene_',''), rotation=45,ha='right')
    for i, v in enumerate(non_zero_counts):
        #remove "gene_" prefix from gene name
        gene_name = non_zero_counts.index[i].replace(genes_prefix,'')
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    #ax.legend()
    ax.grid()
    pp.savefig(fig)

    for non_zero_counts_column in non_zero_counts.index:
        if non_zero_counts[non_zero_counts_column] < min_number_of_cases_of_mutations:
            df_clean.drop(columns=[non_zero_counts_column],inplace=True)

    df_clean = df_clean[df_clean['outlier'] == False]
    df_clean['status'] = df_clean['status'].astype(bool)
    for gi in genes_of_interest:
        gene_column_name = genes_prefix + gi
        if gene_column_name in df_clean.columns:

            fig = plot_kaplan_meier2(df_clean,column_name=gene_column_name,status_column='status',survival_in_days='survival_in_days')
            pp.savefig(fig)
            fig = plot_kaplan_meier2(df_clean, column_name=gene_column_name, status_column='disease-free-status',survival_in_days='disease-free-time')
            pp.savefig(fig)
    for fi in factors_of_interst:
        if fi in df_clean.columns:
            pvals = compute_survival_p_values(df_clean, column_name=fi, status_column='status',
                                              survival_in_days='survival_in_days')
            pvals = compute_survival_p_values(df_clean, column_name=fi, status_column='disease-free-status',
                                              survival_in_days='disease-free-time')
            fig = plot_kaplan_meier2(df_clean,column_name=fi,status_column='status',survival_in_days='survival_in_days')
            pp.savefig(fig)
            fig = plot_kaplan_meier2(df_clean, column_name=fi, status_column='disease-free-status',survival_in_days='disease-free-time')
            pp.savefig(fig)
    #drop df_clean rows where df_clean['treatment'] == 'None'
    df_clean = df_clean[df_clean['treatment'] != np.nan]
    for unique_treatment in df_clean['treatment'].unique():
        df_clean['treatment_'+str(unique_treatment)] = df_clean['treatment'] == unique_treatment
    genes_treatment_analysis = {'factor': [],
                       'pval_survival': [], 'pval_disease_free': [],
                       'median_survival': [], 'median_disease_free': []}
    pp.close()
    if args.show:
        plt.show()
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
    #plot non_zero_counts series as bar plot

    pp = PdfPages(output_pdf)

    #plt.show()
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(non_zero_counts.index, non_zero_counts.values, color='b', label='All genes')
    #select genes of interest from non_zero_counts series
    non_zero_counts_genes_of_interest = non_zero_counts[non_zero_counts.index.str.contains('|'.join(genes_of_interest))]
    if len(non_zero_counts_genes_of_interest) > 0:
        ax.bar(non_zero_counts_genes_of_interest.index, non_zero_counts_genes_of_interest.values,
               color='r', label='Genes of interets')
    ax.set_ylabel('Number of patients that have mutation in gene')
    ax.set_title('Number of present mutation in genes for whole dataset')
    ax.set_xticks(range(len(non_zero_counts)))

    #set xtecklabels without "gene_" prefix
    ax.set_xticklabels(non_zero_counts.index.str.replace('gene_',''), rotation=45,ha='right')
    for i, v in enumerate(non_zero_counts):
        #remove "gene_" prefix from gene name
        gene_name = non_zero_counts.index[i].replace(genes_prefix,'')
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    ax.legend()
    ax.grid()
    pp.savefig(fig)

    for non_zero_counts_column in non_zero_counts.index:
        if non_zero_counts[non_zero_counts_column] < min_number_of_cases_of_mutations:
            df_clean.drop(columns=[non_zero_counts_column],inplace=True)

    df_clean = df_clean[df_clean['outlier'] == False]
    df_clean['status'] = df_clean['status'].astype(bool)
    '''
    data_y = df_clean[['status', 'survival_in_days']].to_records(index=False)
    data_x = df_clean.drop(columns=['survival_in_days','status',"disease-free-status","disease-free-time",
                                    'patient_id','initial_date','initial_report_date','min_date','last_date','death_date'
                                    ,'treatment_date','recurrence_date','outlier'])
    data_x.fillna(0,inplace=True)
    data_x_norm = dataframe_normalization(data_x)
    Xt = OneHotEncoder().fit_transform(data_x_norm.astype('category'))
    set_config(display="text")  # displays text representation of estimators

    estimator = CoxPHSurvivalAnalysis(alpha=0.01, ties='breslow', n_iter=100, tol=1e-09, verbose=0)
    estimator.fit(data_x_norm, data_y)

    data_y = df_clean[['status', 'survival_in_days']].to_records(index=False)
    scores = fit_and_score_features(data_x_norm.values, data_y)
    phazards_surv = pd.Series(scores, index=data_x_norm.columns).sort_values(ascending=False)

    data_y = df_clean[["disease-free-status","disease-free-time"]].to_records(index=False)
    scores = fit_and_score_features(data_x_norm.values, data_y)

    phazards_dfree = pd.Series(scores, index=data_x_norm.columns).sort_values(ascending=False)

    #show columns with NaN values in df_clean
    print(df_clean.columns[df_clean.isna().any()].tolist())
    #show rows with NaN values in df_clean
    print(df_clean[df_clean.isna().any(axis=1)])

    df_clean['patients'] = 'ALL'
    fig = plot_kaplan_meier(df_clean, column_name='patients', status_column='status',
                            survival_in_days='survival_in_days')
    pp.savefig(fig)
    '''
    for gi in genes_of_interest:
        gene_column_name = genes_prefix + gi
        if gene_column_name in df_clean.columns:

            fig = plot_kaplan_meier(df_clean,column_name=gene_column_name,status_column='status',survival_in_days='survival_in_days')
            pp.savefig(fig)
            fig = plot_kaplan_meier(df_clean, column_name=gene_column_name, status_column='disease-free-status',survival_in_days='disease-free-time')
            pp.savefig(fig)
    for fi in factors_of_interst:
        if fi in df_clean.columns:
            pvals = compute_survival_p_values(df_clean, column_name=fi, status_column='status',
                                              survival_in_days='survival_in_days')
            pvals = compute_survival_p_values(df_clean, column_name=fi, status_column='disease-free-status',
                                              survival_in_days='disease-free-time')
            fig = plot_kaplan_meier(df_clean,column_name=fi,status_column='status',survival_in_days='survival_in_days')
            pp.savefig(fig)
            fig = plot_kaplan_meier(df_clean, column_name=fi, status_column='disease-free-status',survival_in_days='disease-free-time')
            pp.savefig(fig)
    #drop df_clean rows where df_clean['treatment'] == 'None'
    df_clean = df_clean[df_clean['treatment'] != np.nan]
    for unique_treatment in df_clean['treatment'].unique():
        df_clean['treatment_'+str(unique_treatment)] = df_clean['treatment'] == unique_treatment
    genes_treatment_analysis = {'factor': [],
                       'pval_survival': [], 'pval_disease_free': [],
                       'median_survival': [], 'median_disease_free': []}
    '''
    for gene in  genes_treatment:
        for unique_treatment in df_clean['treatment'].dropna().unique():
            factor_name = 'treatment_' + str(unique_treatment)+"_"+gene
            df_clean[factor_name] = df_clean['treatment_'+str(unique_treatment)] & df_clean[genes_prefix+gene]
            #calculate number of True and False values in df_clean['treatment_' + str(unique_treatment)+"_"+gene]
            skip_factor = False
            for vc in df_clean[factor_name].value_counts():
                if vc < 4:
                    skip_factor = True
                    break
            if skip_factor:
                continue
            df_sub_sample = df_clean[df_clean['treatment'] == unique_treatment]
            fig = plot_kaplan_meier(df_sub_sample, column_name='treatment_' + str(unique_treatment)+"_"+gene, status_column='status',
                                    survival_in_days='survival_in_days')
            fig.suptitle(factor_name)
            pp.savefig(fig)
            fig = plot_kaplan_meier(df_sub_sample, column_name='treatment_' + str(unique_treatment)+"_"+gene, status_column='disease-free-status',
                                    survival_in_days='disease-free-time')
            fig.suptitle(factor_name)
            pp.savefig(fig)
            pvals_surv = compute_survival_p_values(df_sub_sample, column_name=factor_name, status_column='status',
                                                   survival_in_days='survival_in_days')
            pvals_dfree = compute_survival_p_values(df_sub_sample, column_name=factor_name, status_column='disease-free-status',
                                                    survival_in_days='disease-free-time')
            mst_surv = compute_median_survival_time(df_sub_sample, column_name=factor_name, status_column='status',
                                               survival_in_days='survival_in_days')
            mst_dfree = compute_median_survival_time(df_sub_sample, column_name=factor_name, status_column='disease-free-status',
                                               survival_in_days='disease-free-time')
            genes_treatment_analysis['factor'].append(factor_name)
            genes_treatment_analysis['pval_survival'].append(min(pvals_surv.values()))
            genes_treatment_analysis['pval_disease_free'].append(min(pvals_dfree.values()))
            genes_treatment_analysis['median_survival'].append(str(mst_surv))
            genes_treatment_analysis['median_disease_free'].append(str(mst_dfree))
    results_table = pd.DataFrame(genes_treatment_analysis)
    results_table.to_csv('treatment_factor_importance.csv')
    genes_treatment_analysis2 = {'factor': [],
                                'pval_survival': [], 'pval_disease_free': [],
                                'median_survival': [], 'median_disease_free': []}
    for gene_status in [1,0]:
        for gene in genes_treatment:
                df_sub_sample = df_clean[df_clean[genes_prefix+gene] == gene_status]
                if gene_status:
                    factor_name = "with_mutation_in_"+gene
                else:
                    factor_name = "without_mutation_in_"+gene

                fig = plot_kaplan_meier(df_clean[df_clean[genes_prefix+gene] == gene_status],
                                        column_name='treatment',
                                        status_column='status',
                                        survival_in_days='survival_in_days')
                if gene_status:
                    fig.suptitle(f"Treatments for patients with mutation in {gene}")
                else:
                    fig.suptitle(f"Treatments for patients WITHOUT mutation in {gene}")
                pp.savefig(fig)
                fig = plot_kaplan_meier(df_clean[df_clean[genes_prefix+gene] == gene_status],
                                        column_name='treatment',
                                        status_column='disease-free-status',
                                        survival_in_days='disease-free-time')
                if gene_status:
                    fig.suptitle(f"Treatments for patients with mutation in {gene}")
                else:
                    fig.suptitle(f"Treatments for patients WITHOUT mutation in {gene}")
                pp.savefig(fig)

                pvals_surv = compute_survival_p_values(df_sub_sample, column_name='treatment', status_column='status',
                                                       survival_in_days='survival_in_days')
                pvals_dfree = compute_survival_p_values(df_sub_sample, column_name='treatment',
                                                        status_column='disease-free-status',
                                                        survival_in_days='disease-free-time')
                mst_surv = compute_median_survival_time(df_sub_sample, column_name='treatment', status_column='status',
                                                        survival_in_days='survival_in_days')
                mst_dfree = compute_median_survival_time(df_sub_sample, column_name='treatment',
                                                         status_column='disease-free-status',
                                                         survival_in_days='disease-free-time')
                genes_treatment_analysis2['factor'].append(factor_name)
                genes_treatment_analysis2['pval_survival'].append(min(pvals_surv.values()))
                genes_treatment_analysis2['pval_disease_free'].append(min(pvals_dfree.values()))
                genes_treatment_analysis2['median_survival'].append(str(mst_surv))
                genes_treatment_analysis2['median_disease_free'].append(str(mst_dfree))
    results_table = pd.DataFrame(genes_treatment_analysis2)
    results_table.to_csv('treatment_factor_importance2.csv')
    '''
    pp.close()
    '''
    factor_analysis = {'factor':[],
                       'pval_survival':[],'pval_disease_free':[],
                       'median_survival':[],'median_disease_free':[],
                       'proportional_hazard_survival':[],'proportional_hazard_disease_free':[]}

    data_x.drop(columns=['age'],inplace=True)
    for factor in data_x.columns:
        if factor in df_clean.columns:
            pvals_surv = compute_survival_p_values(df_clean, column_name=factor, status_column='status',
                                              survival_in_days='survival_in_days')
            pvals_dfree = compute_survival_p_values(df_clean, column_name=factor, status_column='disease-free-status',
                                              survival_in_days='disease-free-time')
            factor_analysis['factor'].append(factor)
            factor_analysis['pval_survival'].append(min(pvals_surv.values()))
            factor_analysis['pval_disease_free'].append(min(pvals_dfree.values()))

            if factor in phazards_surv:
                factor_analysis['proportional_hazard_survival'].append(phazards_surv[factor])
            else:
                factor_analysis['proportional_hazard_survival'].append(np.nan)
            if factor in phazards_dfree:
                factor_analysis['proportional_hazard_disease_free'].append(phazards_dfree[factor])
            else:
                factor_analysis['proportional_disease_free'].append(np.nan)
            mst = compute_median_survival_time(df_clean, column_name=factor, status_column='status',
                                               survival_in_days='survival_in_days')
            factor_analysis['median_survival'].append(str(mst))
            mst = compute_median_survival_time(df_clean, column_name=factor, status_column='disease-free-status',
                                              survival_in_days='disease-free-time')
            factor_analysis['median_disease_free'].append(str(mst))

    results_table = pd.DataFrame(factor_analysis)
    results_table.to_csv('factor_importance.csv')
    
    #print(results_table)
    #plt.show()
    '''
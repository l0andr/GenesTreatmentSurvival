#!/usr/bin/env python
import warnings
import argparse
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional

from lifelines import CoxPHFitter
from lifelines.calibration import survival_probability_calibration
from numpy.linalg.linalg import multi_dot
from tqdm import tqdm
from lifelines.utils import k_fold_cross_validation


from lifelines.utils import CensoringType
from lifelines.fitters import RegressionFitter
from lifelines import CRCSplineFitter
from utility_functions import prepare_columns_for_model,convert_to_float_and_normalize

def estimate_model_quality(cph, df, duration_col, event_col,t0=365):
    methods = ["concordance_index", "log_likelihood", "log_ratio_test", "probability_calibration","AIC"]
    score_dict = {}
    try:
        cph.fit(df, duration_col=duration_col, event_col=event_col, show_progress=False)
    except Exception as e:
        for scoring_method in methods:
            score_dict[scoring_method] = None
        return score_dict

    for scoring_method in methods:
        if scoring_method == "concordance_index" or scoring_method == "log_likelihood":
            score = cph.score(df, scoring_method=scoring_method)
        elif scoring_method == "log_ratio_test":
            score = cph.log_likelihood_ratio_test().summary['p'][0]
        elif scoring_method == "probability_calibration":
            ICI, E50 = survival_probability_calibration_mod(cph, df, t0=t0)
            score = ICI
        elif scoring_method == "AIC":
            score = cph.AIC_partial_
        else:
            raise RuntimeError(f"Unknown scoring method:{scoring_method}")
        score_dict[scoring_method] = score
    return score_dict



def survival_probability_calibration_mod(model: RegressionFitter, df: pd.DataFrame, t0: float):
    # Calculation of the Integrated Calibration Index (ICI) and the Expected 50% Calibration Error (E50)
    # Graphical calibration curves and the integrated calibration index (ICI) for
    # competing risk models
    # Peter C. Austin, Hein Putter, Daniele Giardiello and David van Klaveren
    def ccl(p):
        return np.log(-np.log(1 - p))
    T = model.duration_col
    E = model.event_col
    predictions_at_t0 = np.clip(1 - model.predict_survival_function(df, times=[t0]).T.squeeze(), 1e-10, 1 - 1e-10)
    prediction_df = pd.DataFrame({"ccl_at_%d" % t0: ccl(predictions_at_t0), T: df[T], E: df[E]})
    n_knots = 3
    regressors = {"beta_": ["ccl_at_%d" % t0], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}
    crc = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if CensoringType.is_right_censoring(model):
            crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_left_censoring(model):
            crc.fit_left_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_interval_censoring(model):
            crc.fit_interval_censoring(prediction_df, T, E, regressors=regressors)
    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze() - predictions_at_t0).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    return ICI, E50

def grid_search_optimal_penalty_and_l1ratio(df,duration_col, event_col,pen_n_steps,l1ratio_n_steps,calib_t0,verbose=1):
    l1_steps = np.linspace(0.0001, 0.5, l1ratio_n_steps)
    pen_steps = np.linspace(0.0001, 0.9, pen_n_steps)
    scores_all = []
    for l1ratio in tqdm(l1_steps, desc="Optimization l1ratio and penalizer of Cox Model", disable= verbose < 1):
        scores_pen = []
        for pen in pen_steps:
            try:
                cph = CoxPHFitter(penalizer=pen, l1_ratio=l1ratio)
                warnings.simplefilter("ignore", RuntimeWarning)
                scores = estimate_model_quality(cph, df, duration_col=duration_col, event_col=event_col, t0=calib_t0)
                scores_pen.append(scores)
                warnings.resetwarnings()
            except Exception:
                scores_pen.append(None)
                continue
        scores_all.append(scores_pen)
    return scores_all

def extract_metric_from_scores(scores_all,metric):
    metric_all = []
    for scores_pen in scores_all:
        metric_pen = []
        for scores in scores_pen:
            if scores is not None:
                metric_pen.append(scores[metric])
            else:
                metric_pen.append(-1)
        metric_all.append(metric_pen)
    return metric_all
def plot_scores(metric_to_plot,scores_all,l1steps,pensteps):
    l1_steps = np.linspace(0.0001, 0.5, l1steps)
    pen_steps = np.linspace(0.0001, 0.9, pensteps)
    fig = plt.figure()
    array_to_plot = np.array(extract_metric_from_scores(scores_all, metric_to_plot))
    plt.imshow(array_to_plot, cmap='coolwarm', interpolation='nearest')
    plt.xlabel("penalizer")
    plt.ylabel("l1_ratio")
    plt.title(f"Cox Model. optimization of {metric_to_plot} ")
    plt.xticks(np.arange(len(pen_steps)), np.round(pen_steps,3), rotation=90)
    plt.yticks(np.arange(len(l1_steps)), np.round(l1_steps,3))
    plt.colorbar()
    plt.grid()
    return fig


def treeplot(df:pd.DataFrame,df2:Optional[pd.DataFrame]=None,
             coef_col='exp(coef)',coef_col_lower='exp(coef) lower',coef_col_upper='exp(coef) upper',
             logplot=True,
             fig = None,axis = None,tit1="",tit2="",selected_list_of_factors=[]):

    if df2 is None:
        if fig is None:
            fig = plt.figure(figsize=(8,12))
        if axis is None:
            axis = plt.gca()
        ax = [axis]
    else:
        fig,ax = plt.subplots(1,2,sharey=True,figsize=(8,12))
        axis=ax[0]

    y = len(df.index)
    lcol = ''
    ucol = ''
    interval_string = ''
    for col in df.columns:
        if coef_col_lower in col:
            lcol = col
            interval_string = col[10:]
        if coef_col_upper in col:
            ucol = col
    df.sort_values(by=coef_col,inplace=True,ascending=False)
    list_of_factors = []
    list_of_ticks = []
    def plot_one_factor(lb,ub,coef,y,axis,color='black',sagn_color='red'):
        if np.sign(lb) == np.sign(ub):
            color = sagn_color
        axis.plot([lb, ub], [y, y], color=color)
        axis.plot([coef,coef], [y, y], 'o', color=color)
        axis.plot([lb, ub], [y, y], '|', color=color)
        axis.plot([lb, ub], [y, y], '|', color=color)
        if np.sign(lb) == np.sign(ub):
            if np.sign(lb) < 0:
                axis.plot([lb], [y+0.4], '*', color=color)
            if np.sign(lb) > 0:
                axis.plot([ub], [y+0.4], '*', color=color)
    center_line = 1 if not logplot else 0
    for a in ax:
        a.vlines(center_line, 0, len(df.index), color='black', linestyles='--')

    for index,row in df.iterrows():
        if index in selected_list_of_factors:
            factor_color='red'
        else:
            factor_color='black'
        plot_one_factor(np.log2(row[lcol]), np.log2(row[ucol]), np.log2(row[coef_col]),y,axis,color=factor_color)
        if df2 is not None:
            row2 = df2[df2.index==index]
            plot_one_factor(np.log2(row2[lcol]),np.log2(row2[ucol]), np.log2(row2[coef_col]), y, ax[1], color='red')
        list_of_factors.append(index)
        list_of_ticks.append(y)
        y = y - 1
    axis.set_title(tit1)

    #if logplot:
    #    pass #xlabel_str = "log(Hazards ratio) \n" + interval_string + 'Confidence interval'
    #else:
    xlabel_str = "Hazards ratio \n" + interval_string + 'Confidence interval'
    for a in ax:
        a.set_xlabel(xlabel_str)
        #get xticks from the pot and compute exp from them and put back
        xticks = a.get_xticks()
        xticks_labels = [f"{np.exp2(x):.2f}" for x in xticks]
        a.set_xticklabels(xticks_labels)




        a.grid()

    if df2 is not None:
        for index, row in df2.iterrows():
            if index not in list(df.index):
                row2 = df2[df2.index == index]
                plot_one_factor(row2[lcol], row2[ucol], row2[coef_col], y, ax[1], color='red')
                list_of_factors.append(index)
                list_of_ticks.append(y)
                y = y - 1
        ax[1].set_title(tit2)

    plt.yticks(ticks=list_of_ticks,labels=list_of_factors)
    plt.tight_layout()
    return fig,axis

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform initial survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("--min_cases", help="Minimal number of cases of mutations", type=int, default=3)
    parser.add_argument("--genes", help="Comma separated list of genes of interest ", type=str, default=None)
    parser.add_argument("--factors", help="Comma separated list of factors of interest ", type=str, default=None)
    parser.add_argument("--status_col", help="Comma separated list of factors of interest ", type=str, default="status")
    parser.add_argument("--survival_time_col", help="Comma separated list of factors of interest ", type=str, default="survival_in_days")
    parser.add_argument("--patient_id_col", help="Comma separated list of factors of interest ", type=str,
                        default="patient_id")
    parser.add_argument("--calib_t0", help="Time that will be used for model calibration", type=int,
                        default=1900)
    parser.add_argument("--l1ratio", help="Cox optimizator L1\L2 ratio", type=float,
                        default=-1)
    parser.add_argument("--penalizer", help="Optimizator Penalizer value", type=float,
                        default=-1)
    parser.add_argument("--opt_report", help="Path to optimization report", type=str,
                        default="cox_optim_report.pdf")
    parser.add_argument("--model_report", help="Path to model report", type=str,
                        default="cox_model_report.pdf")
    parser.add_argument("--univar", help="Perform univariante analysis use specified list of factors as base and vary other", type=str,
                        default=None)
    parser.add_argument("--verbose", help="Verbose level 0-silent, 3-maximum verbose", type=int, default=1)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')
    parser.add_argument("--plot_outcome", help="If set, plots will be shown", default=False,
                        action='store_true')
    parser.add_argument("--filter_nan", help="If set, rows with empty or NaN cells willbe filtered out", default=False,
                        action='store_true')
    parser.add_argument("--filter_nan_columns", help="comma separated list of columns where NaN will be detected and filetered", default="")
    parser.add_argument("--title", help="Title of plot", type=str, default="")

    #if factors are not specified, then all factors will be used
    #if genes are not specified, then all genes will be used
    args = parser.parse_args()
    input_csv = args.input_csv
    min_cases = args.min_cases
    show = args.show
    global ALPHA
    ALPHA = 0.1
    if args.genes is None or args.genes == "":
        genes = None
    else:
        genes = args.genes.split(',')
    if args.factors == "":
        factors = ""

    elif len(args.factors) > 0:
        factors = args.factors.split(',')
    else:
        factors = None
    if args.univar == "":
        list_of_univar_factors = []
    elif args.univar is not None:
        if args.penalizer < 0 or args.l1ratio < 0:
            raise RuntimeError("Penalizer and l1ratio must be set for univariante analysis")
        list_of_univar_factors= args.univar.split(',')
    df = pd.read_csv(args.input_csv)

    patient_id_col = args.patient_id_col
    status_col = args.status_col
    survival_time_col = args.survival_time_col
    keep_columns = [status_col, survival_time_col]
    #todo add ignore columns
    ignore_columns = ['distance_from_mean', 'outlier','disease-free-status', 'disease-free-time','patient_id','status','survival_in_days']
    genes_columns = []
    if args.genes is None:
        genes_columns = [col for col in df.columns if col.startswith('gene_')]
        genes_columns = [col for col in genes_columns if df[col].sum() >= min_cases]
    if genes is not None:
        for gene in genes:
            genes_columns.extend([col for col in df.columns if col.startswith(f'gene_{gene}')])
    #keep only such genes which have at least min_cases


    factor_columns = []
    if factors is not None and len(factors) > 0:
        factor_columns = factors
    elif factors is None:
        factor_columns = [col for col in df.columns if not col.startswith('gene_') and
                          col not in keep_columns and
                          col not in ignore_columns and
                          not col.endswith('_date')]

    if args.verbose > 1:
        print(f"Model will be created based on following columns of data:")
        print(f"genes_columns:{genes_columns}")
        print(f"factor_columns:{factor_columns}")
        print(f"status_col:{status_col}")
        print(f"survival_time_col:{survival_time_col}")
        print(f"patient_id_col:{patient_id_col}")
    if args.filter_nan:
        # check columns genes and factors and filter out rows with NaN
        if args.verbose >= 1:
            print(f"Number of rows before NaN filtering:{len(df.index)}")
        df = df.dropna(subset=genes_columns + factor_columns)
        if args.verbose >= 1:
            print(f"Number of rows after NaN filtering:{len(df.index)}")
    if args.filter_nan_columns != "":
        columns_to_filter = args.filter_nan_columns.split(',')
        if args.verbose >= 1:
            print(f"Number of rows before NaN in columns {columns_to_filter} filtering:{len(df.index)}")
        df = df.dropna(subset=columns_to_filter)
        if args.verbose >= 1:
            print(f"Number of rows after NaN in columns {columns_to_filter} filtering:{len(df.index)}")

    if args.univar is None:
        list_of_univar_factors = []
    df_formodel = df[list(set(keep_columns + genes_columns + factor_columns + list_of_univar_factors))]
    column_for_dropping = []
    for column in (set(df_formodel.columns) - {status_col, survival_time_col}):
        if len(df_formodel[column].unique().tolist()) == 1:
            if args.verbose >= 1:
                print(f"{column} will be dropped (variance = 0)")
            column_for_dropping.append(column)
        if df_formodel[column].dtype == int and sorted(df_formodel[column].unique().tolist()) == [0,1] :
            if args.verbose >= 1:
                print(f"{column} will be converted to boolean")
            df_formodel[column] = df_formodel[column].astype('bool')
        elif len(df_formodel[column].unique().tolist()) <= 5 and not df_formodel[column].dtype == bool:
            if args.verbose >= 1:
                print(f"{column} will be converted to multiple columns {df_formodel[column].unique().tolist()}")
            for val in df_formodel[column].unique().tolist():
                if val is None or val == np.nan or val == 'nan' or pd.isna(val):
                    continue
                if args.verbose >= 1:
                    print(f"\tBinnary column {column}={val} created")
                df_formodel[f"{column}={val}"] = df_formodel[column] == val
                df_formodel[f'{column}={val}'] = df_formodel[f'{column}={val}'].astype('bool')
                factor_columns.append(f"{column}={val}")
                if column in list_of_univar_factors:
                    list_of_univar_factors.append(f"{column}={val}")

            #remove column from factor_columns
            if column in list_of_univar_factors:
                list_of_univar_factors.remove(column)

            try:
                factor_columns.remove(column)
            except ValueError:
                print(f"Column {column} not found in factor_columns")
                raise ValueError(f"Column {column} not found in factor_columns")
            column_for_dropping.append(column)
        elif (df_formodel[column].dtype == int or  df_formodel[column].dtype ==np.float64 or df_formodel[column].dtype == np.int64) and \
                len(df_formodel[column].unique().tolist()) > 5:
            df_formodel[column] = df_formodel[column].fillna(df_formodel[column].median()).astype(np.float64)



    df_formodel.drop(columns=column_for_dropping, inplace=True)
    #process all columns except status_col and survival_time_col
    #df_formodel_data = convert_to_float_and_normalize(df_formodel[df_formodel.columns.difference([status_col,survival_time_col])])
    df_formodel_data = df_formodel[df_formodel.columns.difference([status_col,survival_time_col])]

    for col in df_formodel_data.columns:
        if col in ['status','survival_in_days']:
            continue
        if df_formodel_data[col].dtype == object:
            df_formodel_data[col] = df_formodel_data[col].astype('category')
            df_formodel_data[col] = df_formodel_data[col].cat.codes
        elif df_formodel_data[col].dtype == np.datetime64:
            df_formodel_data[col] = df_formodel_data[col].astype(np.int64) // 10 ** 9
        elif df_formodel_data[col].dtype == np.int64:
            df_formodel_data[col] = df_formodel_data[col].astype(np.float64)
        elif df_formodel_data[col].dtype == bool:
            df_formodel_data[col] = df_formodel_data[col].astype(np.float64)
        #df_formodel_data[col] = (df_formodel_data[col] - df_formodel_data[col].mean()) / df_formodel_data[col].std()
        df_formodel_data[col] = (df_formodel_data[col] - df_formodel_data[col].min())
        df_formodel_data[col] = df_formodel_data[col] / df_formodel_data[col].max()

    df_formodel = pd.concat([df_formodel_data,df_formodel[[status_col,survival_time_col]]],axis=1)
    #list of columns with NaN values
    nan_columns = df_formodel.columns[df_formodel.isna().any()].tolist()
    if args.verbose > 1:
        print(f"Columns with NaN values:{nan_columns}")

    #remove rows with nan
    if args.verbose > 1:
        print(f"df_formodel.columns:{df_formodel.columns}")
        print(f"df_formodel number of samples:{len(df_formodel.index)}")
    if len(args.opt_report) > 0:
        df_descr = df_formodel.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95], include='all')
        df_descr.to_csv(args.opt_report.split('.')[0] + "_df_descr.csv")
    if len(args.opt_report) and args.verbose > 1:
        df_formodel.to_csv(args.opt_report.split('.')[0] + "_df_formodel.csv")
    # test different penalizer

    calib_t0 = args.calib_t0
    l1ratio_steps = 10
    pen_steps = 10
    best_pen = args.penalizer
    best_l1ratio = args.l1ratio
    if args.l1ratio < 0 or args.penalizer < 0:
        scores_all = grid_search_optimal_penalty_and_l1ratio(df_formodel, duration_col=survival_time_col,
                                                             event_col=status_col, pen_n_steps=pen_steps,
                                                             l1ratio_n_steps=l1ratio_steps,
                                                             calib_t0=calib_t0, verbose=args.verbose)
        ppo = PdfPages(args.opt_report)
        fig = plt.figure()
        buffer = io.StringIO()
        df_formodel.info(buf=buffer)
        plt.text(0.1, 0.1, buffer.getvalue(), fontsize=10)
        plt.axis('off')
        ppo.savefig(fig)
        #plt.show()
        for metric in ["concordance_index", "log_likelihood", "log_ratio_test", "probability_calibration","AIC"]:
            fig = plot_scores(metric, scores_all, l1ratio_steps,pen_steps)
            ppo.savefig(fig)

        best_score = None
        best_scores = None
        best_pen = 0
        best_l1ratio = 0
        for i, l1ratio in enumerate(np.linspace(0.0001,0.5, l1ratio_steps)):
            for j, pen in enumerate(np.linspace(0.0001, 0.9, pen_steps)):
                if scores_all[i][j] is not None:
                    if best_score is None or (scores_all[i][j]["probability_calibration"] > -1 and
                                              scores_all[i][j]["probability_calibration"]*1.05 < best_score):
                        best_score = scores_all[i][j]["probability_calibration"]
                        best_scores = scores_all[i][j]
                        best_pen = pen
                        best_l1ratio = l1ratio
        opt_text = f"Results of grid search optimization:\n" + \
               f"best penalizer:{best_pen}\n" + \
               f"best l1ratio:{best_l1ratio}\n" + \
               f"best_scores:\n"
        for key, value in best_scores.items():
            opt_text += f"    {key}:{value}\n"
        plt.close("all")
        fig = plt.figure()
        plt.text(0.1, 0.1, opt_text, fontsize=12)
        plt.axis('off')
        ppo.savefig(fig)

        fig=plt.figure()
        cph = CoxPHFitter(alpha=ALPHA, penalizer=best_pen, l1_ratio=best_l1ratio)
        cph.fit(df_formodel, duration_col=survival_time_col, event_col=status_col, show_progress=True)
        axes, ICI, E50 = survival_probability_calibration(cph, df_formodel, t0=calib_t0)
        ppo.savefig(fig)
        ppo.close()

    #df_formodel.drop(columns=['drugs'], inplace=True)

    if args.univar is not None:
        common_uni_factors = []
        for cox_factor in sorted(genes_columns + factor_columns):
            if cox_factor in list_of_univar_factors:
                continue
            if args.verbose > 1:
                print(f"Univariante analysis for {cox_factor}")
            cphu = CoxPHFitter(alpha=ALPHA, penalizer=best_pen, l1_ratio=best_l1ratio)
            cphu.fit(df_formodel[list(set(list_of_univar_factors+[cox_factor] + keep_columns))].dropna(), duration_col=survival_time_col, event_col=status_col, show_progress=False)
            #if args.show:
                #cphu.plot()
                #plt.show()
            tbl = cphu.summary
            common_uni_factors.append(tbl[tbl.index==cox_factor])
        df_coomon_uni_factors = pd.concat(common_uni_factors)
        if args.verbose > 1:
            print(f"Univariante factors:")
            print(df_coomon_uni_factors)
    #preselect sagnificant factors on the base of univariant analysis:

    if args.univar is not None:
        multi_factors = list_of_univar_factors
        for col in df_coomon_uni_factors.columns:
            if 'coef lower' in col:
                lcol = col
            if 'coef upper' in col:
                ucol = col
        for index,row in df_coomon_uni_factors.iterrows():
            if np.sign(row[lcol]) == np.sign(row[ucol]):
                multi_factors.append(index)
        if args.verbose > 1:
            print(f"List of factors for multifactor analysis:{multi_factors}")
        df_formodel = df_formodel[list(set(multi_factors + keep_columns))]
    else:
        multi_factors = list(set(genes_columns + factor_columns + keep_columns))
        df_formodel = df_formodel[list(set(genes_columns + factor_columns + keep_columns))]
    #TODO: check that we have not NaNs in data

    if len(multi_factors) > 0:
        cph = CoxPHFitter(alpha=ALPHA,penalizer=best_pen,l1_ratio=best_l1ratio)
        cph.fit(df_formodel, duration_col=survival_time_col, event_col=status_col,show_progress=False)
        score_dict = estimate_model_quality(cph, df_formodel, duration_col=survival_time_col, event_col=status_col,t0=calib_t0)
        if args.verbose > 1:
            print(f"Model scores:\n {score_dict}")
        if args.verbose > 1:
            print(f"Multivariante factors:")
            cph.print_summary()
        if len(list_of_univar_factors) > 0:
            pass
    title_prefix = args.title
    pp = PdfPages(args.model_report)
    if args.univar is not None:
        fig,axis = treeplot(df_coomon_uni_factors, df2=None,tit1=title_prefix+'[Univariant analysis]',logplot=True,selected_list_of_factors=multi_factors)
        pp.savefig(fig)
    if args.show:
        plt.show()
    if args.plot_outcome:
        for factor in df_formodel.columns.difference([status_col,survival_time_col]):
            cph.plot_partial_effects_on_outcome(factor, df_formodel[factor].unique().tolist())
            plt.grid()
            plt.title(f"Partial effect of {factor}")

    if len(multi_factors) > 0:
        if args.verbose > 1:
            print(cph.summary)
        cph.check_assumptions(df_formodel, p_value_threshold=0.01)
        fig,axis = treeplot(cph.summary,tit1=f"{title_prefix} n={len(df_formodel.index)}.\n Cox model concordance index: {cph.concordance_index_:.4f}",logplot=True)
        pp.savefig(fig)
    pp.close()
    if args.show:
        plt.show()
#!/usr/bin/env python
import warnings
import argparse
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lifelines import CoxPHFitter
from lifelines.calibration import survival_probability_calibration
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


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser(description="Perform initial survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("--min_cases", help="Minimal number of cases of mutations", type=int, default=6)
    parser.add_argument("--genes", help="Comma separated list of genes of interest ", type=str, default="")
    parser.add_argument("--factors", help="Comma separated list of factors of interest ", type=str, default="")
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

    parser.add_argument("--verbose", help="Comma separated list of factors of interest ", type=int, default=2)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')

    #if factors are not specified, then all factors will be used
    #if genes are not specified, then all genes will be used
    args = parser.parse_args()
    input_csv = args.input_csv
    min_cases = args.min_cases
    show = args.show
    global ALPHA
    ALPHA = 0.1
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
    ignore_columns = ['distance_from_mean', 'outlier','disease-free-status', 'disease-free-time','patient_id','status','survival_in_days']
    genes_columns = [col for col in df.columns if col.startswith('gene_')]
    if genes is not None:
        genes_columns = [col for col in genes_columns if col.split('_')[1] in genes]
    #keep only such genes which have at least min_cases
    genes_columns = [col for col in genes_columns if df[col].sum() >= min_cases]

    factor_columns = []
    if factors is not None:
        factor_columns = factors
    else:
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

    df_formodel = df[keep_columns + genes_columns + factor_columns]
    #prepare data for model
    df_formodel = df_formodel[df_formodel['treatment'] < 4]
    #fill Na with np.nan
    #df_formodel = df_formodel.fillna(np.inf)
    #Drop column treatment
    #df_formodel.drop(columns=['treatment'],inplace=True)
    df_formodel = df_formodel.dropna()
    column_for_dropping = []
    for column in (set(df_formodel.columns) - {status_col, survival_time_col}):
        if df_formodel[column].dtype == object:
            df_formodel[column] = df_formodel[column].astype('category')
            df_formodel[column] = df_formodel[column].cat.codes

        if len(df_formodel[column].unique().tolist()) == 1:
            if args.verbose >= 1:
                print(f"{column} will be dropped (variance = 0)")
            column_for_dropping.append(column)
        if sorted(df_formodel[column].unique().tolist()) == [0,1]:
            if args.verbose >= 1:
                print(f"{column} will be converted to boolean")
            df_formodel[column] = df_formodel[column].astype('bool')
        
        if len(df_formodel[column].unique().tolist()) < 5 and not df_formodel[column].dtype == bool:
            print(f"{column} will be converted to categorical")
            df_formodel[column] = df_formodel[column].astype('category')
            df_formodel[column] = df_formodel[column].cat.codes

    df_formodel.drop(columns=column_for_dropping, inplace=True)
    #process all columns except status_col and survival_time_col
    df_formodel_data = convert_to_float_and_normalize(df_formodel[df_formodel.columns.difference([status_col,survival_time_col])])
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
        pp = PdfPages(args.opt_report)
        fig = plt.figure()
        buffer = io.StringIO()
        df_formodel.info(buf=buffer)
        plt.text(0.1, 0.1, buffer.getvalue(), fontsize=10)
        plt.axis('off')
        pp.savefig(fig)
        #plt.show()
        for metric in ["concordance_index", "log_likelihood", "log_ratio_test", "probability_calibration","AIC"]:
            fig = plot_scores(metric, scores_all, l1ratio_steps,pen_steps)
            pp.savefig(fig)
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
        pp.savefig(fig)

        fig=plt.figure()
        cph = CoxPHFitter(alpha=ALPHA, penalizer=best_pen, l1_ratio=best_l1ratio)
        cph.fit(df_formodel, duration_col=survival_time_col, event_col=status_col, show_progress=True)
        axes, ICI, E50 = survival_probability_calibration(cph, df_formodel, t0=calib_t0)
        pp.savefig(fig)
        pp.close()

    #df_formodel.drop(columns=['drugs'], inplace=True)
    cph = CoxPHFitter(alpha=ALPHA,penalizer=best_pen,l1_ratio=best_l1ratio)
    cph.fit(df_formodel, duration_col=survival_time_col, event_col=status_col,show_progress=False)
    score_dict = estimate_model_quality(cph, df_formodel, duration_col=survival_time_col, event_col=status_col,t0=calib_t0)
    if args.verbose > 1:
        print(f"Model scores:\n {score_dict}")


    if args.verbose > 1:
        cph.print_summary()

    cph.check_assumptions(df_formodel, p_value_threshold=0.01)


    pp = PdfPages(args.model_report)
    fig = plt.figure()
    cph.plot()
    plt.title(f"n ={len(df_formodel.index)}. Status:{status_col}, survival_time:{survival_time_col}")
    plt.tight_layout()
    pp.savefig(fig)
    pp.close()
    if args.show:
        plt.show()
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts



def plot_piecharts_of_categorial_variables(df_clean:pd.DataFrame):
    #df_tmp = df_clean.loc[:, ~df_clean.columns.str.contains('date', case=False)]
    #df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('gene_', case=False)]
    df_tmp = df_clean.loc[:, df.dtypes == 'category' ]
    nrows = min([3, len(df_tmp.columns)])
    ncols = int(np.ceil(len(df_tmp.columns) / nrows))
    # plot in one figure pie charts of all columns in df_tmp
    fig, ax = plt.subplots(figsize=(18, 10), nrows=nrows, ncols=ncols)
    if nrows == 1:
        ax = [ax]
    for i, col in enumerate(df_tmp.columns):
        df_tmp[col].value_counts(dropna=False).plot.pie(ax=ax[i // ncols, i % ncols], autopct='%.2f', fontsize=10)
        ax[i // ncols, i % ncols].set_title(f'{col}')
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(ax[j // ncols, j % ncols])
    plt.tight_layout()
    return fig, ax

def plot_value_counts(df, columns):
    fig, axes = plt.subplots(nrows=len(columns), figsize=(10, 6))
    if len(columns) == 1:
        axes = [axes]
    i = 0
    for col in columns:
        # Get value counts
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        labels_nan = counts.index
        values = counts.values
        labels = []
        for l in labels_nan:
            if not isinstance(l,str):
                labels.append('no data')
            else:
                labels.append(l)
        # Plot
        ax = axes[i]
        bars = ax.bar(labels, values)
        ax.set_title(f"Value сounts for {col}")
        ax.set_ylabel("Number of Occurrences")
        ax.set_xticks(np.linspace(0,len(labels)-1,len(labels)))
        ax.set_xticklabels(labels, rotation=5, ha='right')

        # Add percentage on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height} ({height/total:.2%})",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom',
                        xytext=(0, 10),
                        textcoords='offset points')
        i = i+1
    plt.tight_layout()
    return fig, axes


def plot_histograms_of_float_values(df_clean:pd.DataFrame):
    # plot in one figure histogram of all float columns with number of unique values more than sqrt(len(df.index))
    df_tmp = df_clean.loc[:, df_clean.dtypes == np.float64]
    #df_tmp = df_tmp.loc[:, df_tmp.nunique() > np.sqrt(len(df_clean.index))/2]
    # and write median value in the title of each axes
    fig, ax = plt.subplots(figsize=(18, 10), nrows=len(df_tmp.columns)//2, ncols=2)
    if not (isinstance(ax,list) or isinstance(ax,np.ndarray)):
        ax = [ax]
    for i, col in enumerate(df_tmp.columns):
        try:
            ax[i//2,i%2].hist(df_tmp[col], bins=int(np.sqrt(len(df_clean.index))))
            ax[i//2,i%2].set_title(f'{col}. Median = {df_tmp[col].median():.2f}')
            ax[i//2,i%2].grid()
        except Exception as e:
            print(f"Error with column {col} {e}")
    plt.tight_layout()
    return fig, ax

def plot_kaplan_meier(df_pu: pd.DataFrame, column_name: str,
                           status_column: str = "Status", survival_in_days: str = "Survival_in_days"):

        diff_values = sorted(df_pu[column_name].dropna().unique().tolist())

        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, sharex=True)
        if not isinstance(ax, np.ndarray):
            ax = [ax]
        i = 0
        kmfs = []
        p_values = {}
        at_risk_lables = []
        for s in diff_values:
            mask_treat = df_pu[column_name] == s
            p_values[s] = logrank_test(df_pu[status_column][mask_treat], df_pu[status_column][~mask_treat],
                                       df_pu[survival_in_days][mask_treat],
                                       df_pu[survival_in_days][~mask_treat]).p_value
            i += 1
            ix = df_pu[column_name] == s
            kmf = KaplanMeierFitter()
            #TODO find more general way to create lables
            full_label = ''
            if column_name.startswith('gene_'):
                label_str = 'mutated' if s == 1 else 'wildtype'
                gene_name = column_name.replace('gene_','')
                full_label = label_str + " " +gene_name
            else:
                label_str = str(s)
                full_label = column_name + " = " + label_str
            kmf.fit(df_pu[survival_in_days][ix], df_pu[status_column][ix],
                    label=full_label + f" p-value = {p_values[s]:.5f} ")
            kmf.plot_survival_function(ax=ax[0], ci_legend=True, at_risk_counts=False)
            at_risk_lables.append(f"{column_name} = {s}")
            kmfs.append(kmf)
        add_at_risk_counts(*kmfs, labels=at_risk_lables, ax=ax[0])
        #compute p-value of multivariate logrank test
        p_value_multivariate = multivariate_logrank_test(df_pu[survival_in_days], df_pu[column_name], df_pu[status_column]).p_value

        ax[0].set_ylabel("est. probability of survival $\hat{S}(t)$")
        ax[0].set_xlabel(f"time $t$ (days)")
        ax[0].set_title(f"Kaplan-Meier survival estimates [{survival_in_days}] ")
        plt.tight_layout()
        return fig

def keep_only_specific_columns(df, keep_columns, ignore_columns):
    return df.loc[:, [col for col in df.columns if
                      col in keep_columns or
                      col not in ignore_columns]]

if __name__ == '__main__':
    list_of_plot_types = ["kaplan_meier", "pieplots", "floathistograms", "valuecounts"]
    parser = argparse.ArgumentParser(description="Plot figres for survival analysis",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_csv", help="Input CSV file", type=str, required=True)
    parser.add_argument("--output_pdf", help="Output file with figures", type=str, required=True)
    parser.add_argument("--plot", help="Type of plot", choices=list_of_plot_types, default="kaplan_meier")
    parser.add_argument("--status_col", help="Column with status (event occur or not) ", type=str, default="status")
    parser.add_argument("--survival_time_col", help="Time until event ", type=str, default="survival_in_days")
    parser.add_argument("--patient_id_col", help="Patients id", type=str, default="patient_id")
    parser.add_argument("--columns", help="One or few columns for plot", type=str, default="")
    parser.add_argument("--min_size_of_group", help="Minimal group for Kaplan-Meier plots as fraction of all casses", type=float, default=0.07)
    parser.add_argument("--max_amount_of_groups", help="Maximum number of groups per factor", type=str,
                        default=10)
    parser.add_argument("--max_survival_length", help="Maximum consider time interval in Kaplan-meier plots", type=float,
                        default=365*5)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')
    parser.add_argument("--verbose", help="Verbose mode", type=int, default=1)

    args = parser.parse_args()
    input_csv = args.input_csv

    status_col = args.status_col
    survival_time_col = args.survival_time_col
    patient_id_col = args.patient_id_col
    show = args.show
    plot_type = args.plot

    df = pd.read_csv(input_csv)
    min_group_size = int(args.min_size_of_group * len(df))
    max_number_of_groups = args.max_amount_of_groups

    pp = PdfPages(args.output_pdf)
    if args.columns == "":
        columns = [col for col in df.columns if col not in [status_col, survival_time_col, patient_id_col]]
    else:
        columns = args.columns.split(',')

    if plot_type == "kaplan_meier":
        i = 0
        #if df[survival_time_col] > args.max_survival_length make df[survival_time_col] = args.max_survival_length
        #and df[status_col] = 0
        df.loc[df[survival_time_col] > args.max_survival_length, status_col] = False
        df.loc[df[survival_time_col] > args.max_survival_length, survival_time_col] = args.max_survival_length

        for col in tqdm.tqdm(columns, desc="Plotting kaplan_meier", disable=args.verbose != 1):
            #check if column is categorial
            i += 1
            if df[col].dtype == 'categorical':
                if args.verbose > 1:
                    print(f"Column {col} is not categorical, skip it")
                continue
            if df[col].nunique() > max_number_of_groups:
                if args.verbose > 1:
                    print(f"Column {col} has too many unique values, skip it")
                continue
            #compute number of cases of each value and skip if one of group have less then min_group_size
            continue_flag = False
            for j in df[col].unique():
                if j is None or pd.isna(j):
                    continue
                if df[col].value_counts()[j] < min_group_size:
                    if args.verbose > 1:
                        print(f"Column {col} has too few cases of {j} {df[col].value_counts()[j]}, skip it")
                    continue_flag = True
            if continue_flag:
                continue

            if args.verbose > 1:
                print(f"Plotting kaplan_meier for column {col} {len(columns)}\{i}. Number of unique values is {df[col].nunique()}. Number of Nulls is {df[col].isnull().sum()}")
            try:
                fig = plot_kaplan_meier(df, col, status_col, survival_time_col)
                pp.savefig(fig)
            except Exception as e:
                print(f"Error while plotting kaplan_meier for column {col}: {str(e)}")
                raise e
    elif plot_type == "pieplots":
        fig, ax = plot_piecharts_of_categorial_variables(df.loc[:,columns])
        pp.savefig(fig)
    elif plot_type == "floathistograms":
        fig, ax = plot_histograms_of_float_values(df.loc[:,columns])
        pp.savefig(fig)
    elif plot_type == "valuecounts":
        fig, ax = plot_value_counts(df, columns)
        pp.savefig(fig)

    if show:
        plt.show()
    pp.close()
    plt.close(fig)
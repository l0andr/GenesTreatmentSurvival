#!/usr/bin/env python

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from typing import List,Optional
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from lifelines.statistics import logrank_test,multivariate_logrank_test
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA

import warnings

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
        ax.set_title(f"Value Counts for {col}")
        ax.set_ylabel("Number of Occurrences")
        ax.set_xticks(np.linspace(0,len(labels)-1,len(labels)))
        ax.set_xticklabels(labels, rotation=5, ha='right')
        # Rotate x-axis labels
        #ax.xticks(rotation=45,ha='right')

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

def plot_number_of_unique_values(df:pd.DataFrame, columns:Optional[List[str]] = None,uthres:Optional[int]=None):
    if columns is None:
        columns_to_check = df.columns
    else:
        columns_to_check = columns
    unique_counts = df[columns_to_check].nunique()
    if uthres is not None:
        unique_counts = unique_counts[unique_counts > uthres]
    unique_counts = unique_counts.sort_values(ascending=False)
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(unique_counts.index, unique_counts.values)
    ax.set_ylabel('Number of Unique Values')
    ax.set_title('Unique Values in Columns')
    ax.set_xticks(range(len(unique_counts)))
    ax.set_xticklabels(unique_counts.index, rotation=45,ha='right')
    for i, v in enumerate(unique_counts):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    return fig,ax

def plot_histograms_of_float_values(df_clean:pd.DataFrame):
    # plot in one figure histogram of all float columns with number of unique values more than sqrt(len(df.index))
    df_tmp = df_clean.loc[:, df_clean.dtypes == np.float64]
    df_tmp = df_tmp.loc[:, df_tmp.nunique() > np.sqrt(len(df_clean.index))/2]
    # and write median value in the title of each axes
    fig, ax = plt.subplots(figsize=(18, 10), nrows=len(df_tmp.columns))
    if not (isinstance(ax,list) or isinstance(ax,np.ndarray)):
        ax = [ax]
    for i, col in enumerate(df_tmp.columns):
        ax[i].hist(df_tmp[col], bins=int(np.sqrt(len(df_clean.index))))
        ax[i].set_title(f'{col}. Median = {df_tmp[col].median():.2f}')
        ax[i].grid()
    return fig, ax

def plot_piecharts_of_categorial_variables(df_clean:pd.DataFrame):
    df_tmp = df_clean.loc[:, ~df_clean.columns.str.contains('date', case=False)]
    df_tmp = df_tmp.loc[:, df_tmp.nunique() <= np.sqrt(len(df_clean.index)) / 2]
    df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('gene_', case=False)]
    # calculate nrows and ncols - number of rows and columns in the figure
    nrows = min([3, len(df_tmp.columns)])
    ncols = int(np.ceil(len(df_tmp.columns) / nrows))
    # plot in one figure pie charts of all columns in df_tmp
    fig, ax = plt.subplots(figsize=(18, 10), nrows=nrows, ncols=ncols)
    if nrows == 1:
        ax = [ax]
    for i, col in enumerate(df_tmp.columns):
        df_tmp[col].value_counts().plot.pie(ax=ax[i // ncols, i % ncols], autopct='%.2f', fontsize=10)
        ax[i // ncols, i % ncols].set_title(f'{col}')
    return fig, ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Exploration Data Analysis report for pre-processed table",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("-output_pdf", help="Exploration data analysis report", type=str, required=True)
    parser.add_argument("-output_csv", help="Output CSV files", type=str, required=True)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')

    args = parser.parse_args()
    genes_prefix = 'gene_'
    duplication_symbols = ['-', '.', 'duplicate']
    patient_id_column = 'patient_name'
    df_clean = pd.read_csv(args.input_csv)


    pp = PdfPages(args.output_pdf)
    # filter out dolumns with float values and dates
    df_tmp = df_clean.loc[:, df_clean.dtypes != np.float64]
    df_tmp = df_tmp.loc[:, df_clean.dtypes != np.datetime64]
    # filter out columns with strings contained datetime
    df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('date', case=False)]
    fig,ax = plot_number_of_unique_values(df_tmp, uthres=2); pp.savefig(fig)
    fig,ax = plot_histograms_of_float_values(df_clean); pp.savefig(fig)
    fig,ax = plot_piecharts_of_categorial_variables(df_clean); pp.savefig(fig)
    #select all columns of df_clean to df_tmp except columns with strings
    df_tmp = df_clean.loc[:, df_clean.dtypes != object]
    df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('date', case=False)]
    df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('gene', case=False)]
    df_tmp = df_tmp.loc[:, df_tmp.nunique() >  5]
    df_tmp.fillna(df_tmp.mean(), inplace=True)
    #normalize data by mean and std values in columns
    df_tmp = (df_tmp - df_tmp.mean()) / df_tmp.std()
    #calculate PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_tmp.values)
    #plot PCA results as scatter plot
    fig, [ax1,ax2] = plt.subplots(figsize=(10, 10),nrows=2)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1],label='inliers')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'PCA by data in columns{list(df_tmp.columns)}')
    #plot mean location as red X symbol
    ax2.scatter(X_pca[:, 0].mean(), X_pca[:, 1].mean(), marker='x', color='red',s = 100,label='mean case')
    #plot by red points that are on the distance more than 2*std from mean
    dth = np.max(np.sqrt((X_pca[:, 0] - X_pca[:, 0].mean()) ** 2 + (X_pca[:, 1] - X_pca[:, 1].mean()) ** 2))*0.7
    ax2.scatter(X_pca[:, 0][np.sqrt((X_pca[:, 0] - X_pca[:, 0].mean()) ** 2 + (X_pca[:, 1] - X_pca[:, 1].mean()) ** 2) > dth ],
                X_pca[:, 1][np.sqrt((X_pca[:, 0] - X_pca[:, 0].mean()) ** 2 + (X_pca[:, 1] - X_pca[:, 1].mean()) ** 2) > dth ],
                marker='o', color='red',label='outliers')
    ax2.legend()
    #plot histogram of distances from mean
    ax1.hist(np.sqrt((X_pca[:, 0] - X_pca[:, 0].mean()) ** 2 + (X_pca[:, 1] - X_pca[:, 1].mean()) ** 2),
            bins=int(np.sqrt(len(df_clean.index)))*2)
    ax1.set_title(f'Distribution of distances from mean case')
    ax1.grid()
    #add column to df_tmp with distances from mean
    df_tmp['distance_from_mean'] = np.sqrt((X_pca[:, 0] - X_pca[:, 0].mean()) ** 2 + (X_pca[:, 1] - X_pca[:, 1].mean()) ** 2)
    #add column to df_clean with distances from mean
    df_clean['distance_from_mean'] = df_tmp['distance_from_mean']
    #add column to df_clean with outliers
    df_clean['outlier'] = np.sqrt((X_pca[:, 0] - X_pca[:, 0].mean()) ** 2 + (X_pca[:, 1] - X_pca[:, 1].mean()) ** 2) > dth
    print(df_clean[df_clean['outlier'] == True])

    pp.savefig(fig)

    #select from df_clean only columns thar starts on gene_
    df_tmp = df_clean.loc[:, df_clean.columns.str.contains('gene_', case=False)]
    #sort columns by number of non zero values
    df_tmp = df_tmp.reindex(df_tmp.sum().sort_values(ascending=False).index, axis=1)
    #sort rows by all columns
    df_tmp = df_tmp.sort_values(by=list(df_tmp.columns), ascending=False)
    #plot heatmap of genes
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.imshow(df_tmp.values, cmap='hot', interpolation='nearest')
    ax.set_title(f'Heatmap of mutations in genes')
    ax.set_xlabel('Genes')
    ax.set_ylabel('Patients')
    ax.set_xticks(range(len(df_tmp.columns)))
    ax.set_xticklabels(df_tmp.columns, rotation=45,ha='right')
    ax.set_yticks(range(len(df_tmp.index)))
    ax.set_yticklabels(df_tmp.index)
    pp.savefig(fig)
    min_number_of_cases_of_mutations = 4
    non_zero_counts = df_tmp.apply(lambda x: (x != 0).sum())
    for non_zero_counts_column in non_zero_counts.index:
        if non_zero_counts[non_zero_counts_column] < min_number_of_cases_of_mutations:
            df_tmp.drop(columns=[non_zero_counts_column], inplace=True)

    #plot heatmap correlation matrix of genes
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(df_tmp.corr().values, cmap='bwr', interpolation='nearest')
    fig.colorbar(im)
    ax.set_title(f'Heatmap of correlation matrix of mutations in genes')
    ax.set_xlabel('Genes')
    ax.set_ylabel('Genes')
    ax.set_xticks(range(len(df_tmp.columns)))
    ax.set_xticklabels(df_tmp.columns, rotation=90,ha='right')
    ax.set_yticks(range(len(df_tmp.columns)))
    ax.set_yticklabels(df_tmp.columns)
    pp.savefig(fig)

    #plot compute heiraarchical clustering of genes
    from scipy.cluster.hierarchy import dendrogram, linkage
    fig, ax = plt.subplots(figsize=(18, 16))
    cmethod = 'average'
    cmetric = 'cosine'
    Z = linkage(np.transpose(df_tmp.values), method = cmethod,metric=cmetric)
    print(Z)
    #plot dendrogram of Z with labels from df_tmp.columns rotated on 90 degree
    dn = dendrogram(Z, labels=df_tmp.columns, ax=ax)
    ax.set_xticklabels(df_tmp.columns, rotation=90, ha='right')
    ax.set_title(
        f'Dendrogram of hierarchy clustering of mutations that are at least in {min_number_of_cases_of_mutations} cases \n'
        f'  method:{cmethod} metric:{cmetric}')
    ax.set_xlabel('Genes')
    ax.set_ylabel('Distance')
    pp.savefig(fig)
    pp.close()

    if args.show:
        plt.show()

    df_clean.to_csv(args.output_csv, index=False)
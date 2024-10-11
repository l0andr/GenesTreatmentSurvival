#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyoncoprint
from matplotlib.backends.backend_pdf import PdfPages


def convert_int_to_rome_number(value:int):
    if value == 1:
        return 'I'
    if value == 2:
        return 'II'
    if value == 3:
        return 'III'
    if value == 4:
        return 'IV'
    if value == 5:
        return 'V'
    if value == 6:
        return 'VI'
    if value == 7:
        return 'VII'
    if value == 8:
        return 'VIII'
    return value


def annotation_replace_rules(column_name, column_values):
    unique_values = sorted(pd.unique(andf.values.ravel()))
    if len(unique_values) > 10:
        return column_values
    if 'stage' in column_name and len(unique_values)<=8:
        for v in unique_values:
            column_values = column_values.replace(v, convert_int_to_rome_number(v))
        return column_values
    elif 'response' in column_name and len(unique_values)<=4:
        column_values = column_values.replace('0', 'Complete Response')
        column_values = column_values.replace('1', 'Partial Response')
        column_values = column_values.replace('2', 'Stable Disease')
        column_values = column_values.replace('3', 'Progressive Disease')
        return column_values
    if len(unique_values) == 2:
        column_values = column_values.replace('0', 'N')
        column_values = column_values.replace('1', 'Y')
        return column_values
    return column_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool for creation of oncoplot",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_mutation", help="Input file with mutations", type=str, required=True)
    parser.add_argument("-output_file", help="Output pdf file with plot", type=str, required=True)
    parser.add_argument("-list_of_factors", help="List of factors to be shown above oncoplot", type=str, required=True)
    parser.add_argument("-list_of_genes", help="List of genes to be shown on oncoplot", type=str, required=True)
    parser.add_argument("--nosortgenes", help="If set genes will not be sorted ", default=False,action='store_true')
    parser.add_argument("--show", help="If set, plots will be shown", default=False,action='store_true')
    parser.add_argument("--number_of_genes",type=int,default=20)
    parser.add_argument("--verbose",type=int,default=1)
    parser.add_argument("--title",type=str,default="")

    args = parser.parse_args()
    number_of_genes = args.number_of_genes
    input_file_mutation = args.input_mutation
    input_df = pd.read_csv(input_file_mutation, sep=',')
    #keep columns started from prefix 'gene_' and column patient_id
    mutation_df = input_df.filter(regex='^gene_|patient_id')
    # replace True to 1 and False to 0
    mutation_df = mutation_df.replace(True, 1)
    mutation_df = mutation_df.replace(False, 0)
    # change type to int for all column started from gene_
    mutation_df = mutation_df.astype({col: 'int' for col in mutation_df.columns if 'gene_' in col})
    list_of_genes = args.list_of_genes.split(',')
    #add for each string in list perfix gene_
    list_of_genes = ['gene_'+x for x in list_of_genes]
    print(f"List of genes:{list_of_genes}")
    #sort columns of mutation_df by list_of_genes
    #if not args.nosortgenes:
    #    mutation_df = mutation_df.reindex(sorted(mutation_df.columns), axis=1)
    mutation_df = mutation_df[['patient_id']+list_of_genes]

    print(mutation_df.columns)
    #obatin number of column with gene mutations
    number_of_genes_in_df = len(mutation_df.columns)
    #compute sum along row and keep only rows from top 20 of biggest sum (exclude )
    # remove prefix gene_ from column names
    mutation_df.columns = mutation_df.columns.str.replace('gene_', '')

    mutation_df.set_index('patient_id', inplace=True)
    mutation_df = mutation_df[mutation_df.sum(axis=0).nlargest(min(number_of_genes, number_of_genes_in_df - 1)).index]
    list_of_genes = args.list_of_genes.split(',')
    mutation_df = mutation_df[list_of_genes]

    mutation_df_sums = mutation_df.sum(axis=0)
    if args.verbose > 1:
        print(f"List of genes for plot:")
        genes_dict = mutation_df_sums.to_dict()
        for key,value in genes_dict.items():
            print(f"{key},",end='')
            if args.verbose > 2:
                print(f"{key}:{value}")

    #list of columns with genes
    genes = mutation_df.columns[mutation_df.columns.str.contains('gene_')].tolist()
    #transpose dataframe make patient_id as columns and genes as rows
    mutation_df = mutation_df.T

    #replace 1 to 'all mutations'
    mutation_df = mutation_df.replace(1, 'all types of mutations')
    list_of_fields_for_annotaion = args.list_of_factors.split(',')
    #select list of fields and patient_id columns from input dataframe
    annotations_df = input_df.filter(items=list_of_fields_for_annotaion+['patient_id'])

    annotations_df = annotations_df.set_index(['patient_id']).T
    annotations = {}
    list_of_cmaps=['viridis','cool']
    cmap_index = 0
    ann_order = 0
    for anncol in list_of_fields_for_annotaion:
        cmap = plt.get_cmap(list_of_cmaps[cmap_index])
        f = anncol
        if anncol not in annotations_df.index:
            continue
        andf = annotations_df[annotations_df.index == f]
        number_of_unique_values = len(pd.unique(andf.values.ravel()))
        if number_of_unique_values > 10:
            andf = andf.fillna('0').astype(float)
            annotations[f] = {
                'annotations': andf,
                'colors': {'age': 'blue'},
                'order': ann_order
            }
        else:
            andf = andf.fillna('0').astype(str)
            andf = annotation_replace_rules(f, andf)

            uniq_annots = sorted(pd.unique(andf.values.ravel()))
            annotations[f] = {'annotations': andf,
            'colors': {k: v for k, v in zip(uniq_annots, cmap(np.linspace(0, 1, len(uniq_annots), dtype=float)))},
            'order': 0}
        ann_order += 1
        cmap_index += 1
        if cmap_index >= len(list_of_cmaps):
            cmap_index = 0

    #create oncoplot
    op = pyoncoprint.OncoPrint(mutation_df)
    mutation_markers = {
        "all types of mutations": dict(
            marker="fill",
            color="black",
            zindex=0
        )}
    if args.output_file.endswith('.pdf'):
        pp = PdfPages(args.output_file)
    if args.nosortgenes:
        sortmethod = 'unsorted'
    else:
        sortmethod = 'default'
    print(sortmethod)
    op.oncoprint(markers=mutation_markers,gene_sort_method=sortmethod,annotations=annotations, topplot=True, rightplot=True,legend=True,
                 title=args.title)
    if args.output_file.endswith('.png'):
        plt.savefig(args.output_file)
    if args.show:
        plt.show()
    if args.output_file.endswith('.pdf'):
        pp.savefig()
        pp.close()

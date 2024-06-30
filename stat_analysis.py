

import argparse
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
from survival_analysis_initial_report import plot_kaplan_meier2
from sksurv.tree import SurvivalTree
from sksurv.preprocessing import OneHotEncoder

def nonezero_counts_plot(df,min_cases,perfix='gene_',title='',ax=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        #set current axe to ax
        plt.sca(ax)
        fig = plt.gcf()
    df = df.filter(regex=perfix)
    df = df.applymap(lambda x: 1 if x > 0 else 0)
    df = df.sum(axis=0) / len(df)
    df = df[df > min_cases]
    df = df.sort_values(ascending=False)
    df.plot(kind='bar',title=title,ylabel='Percent cases with mutation')
    return fig,ax

def nonezero_diff_counts_plot(df1,df2,min_cases,perfix='gene_',title='',ax=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        # set current axe to ax
        plt.sca(ax)
        fig = plt.gcf()
    df1 = df1.filter(regex=perfix)
    df1 = df1.applymap(lambda x: 1 if x > 0 else 0)
    df1 = df1.sum(axis=0)
    df2 = df2.filter(regex=perfix)
    df2 = df2.applymap(lambda x: 1 if x > 0 else 0)
    df2 = df2.sum(axis=0)
    #plot df1 and df2 on the same plot with different colors
    df1 = df1[df1 > min_cases] / (len(df1)+len(df2))
    df2 = df2[df2 > min_cases] / (len(df1)+len(df2))
    df1 = df1.sort_values(ascending=False)
    df2 = df2.sort_values(ascending=False)
    df1.plot(kind='bar',title=title,ylabel='Percent cases with mutation',color='b')
    df2.plot(kind='bar',ylabel='Percent cases with mutation',color='r')

    return fig,ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform data from initial csv to csv suitable for survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("-input_delimiter", help="Delimiter for input file", type=str, default=",")
    args = parser.parse_args()
    input_csv = args.input_csv
    input_delimiter = args.input_delimiter
    df = pd.read_csv(input_csv, delimiter=input_delimiter)
    #
    # Sum of true values in columns started from prefix gene_
    df['number_of_mutation'] = df.filter(regex='gene_').sum(axis=1)
    # convert columns treatment_timeN to one column treatment_time with list of values
    treatment_time_columns = df.filter(regex='treatment_time').columns
    df['treatment_time'] = df[treatment_time_columns].values.tolist()
    # convert columns treatment_typeN to one column treatment_type with list of values
    treatment_type_columns = df.filter(regex='treatment_type').columns
    df['treatment_type'] = df[treatment_type_columns].values.tolist()
    response_columns = df.filter(regex='response_').columns
    df['response_'] = df[response_columns].values.tolist()
    for i in range(len(df['treatment_type'])):
        end_index = None
        if 'none' in df['treatment_type'][i]:
            end_index = df['treatment_type'][i].index('none')
        if 'none' in df['response_'][i]:
            end_index = df['response_'][i].index('none')
        if end_index is not None:
            df['treatment_type'][i] = df['treatment_type'][i][:end_index]
            df['response_'][i] = df['response_'][i][:end_index]
            df['treatment_time'][i] = df['treatment_time'][i][:end_index]

    j = 0
    response_sums = {0:[0,0,0,0],1:[0,0,0,0],2:[0,0,0,0],3:[0,0,0,0],4:[0,0,0,0],5:[0,0,0,0]}
    disiase_free_times = {0:[],1:[],2:[],3:[],4:[],5:[]}
    treatment_type_attempts = {0: [0,0,0], 1: [0,0,0], 2: [0,0,0], 3: [0,0,0], 4: [0,0,0], 5: [0,0,0]}

    disiase_free_times_response = {0: [], 1: [], 2: [], 3: []}
    disiase_free_times_treatment = {0: [], 1: [], 2: []}
    response_last_response = {0: {}, 1: {}, 2: {}, 3: {}}
    response_last_treatment = {0: {}, 1: {}, 2: {}}
    for key, value in response_last_response.items():
        for i in range(4):
            value[i] = 0
    for key, value in response_last_treatment.items():
        for i in range(4):
            value[i] = 0
    transf_dataset = []
    list_of_lost_times={}
    list_of_lost_repsonse_treatment={}

    for row in df.iterrows():
        print(f"{row[1]['patient_id']} start. ", end=' ')
        j+=1
        if j > 500:
            plt.figure()
            j = 0
        d = row[1]['treatment_time'] #list of times of treatment
        #continue if list contains value less then -5000
        if any([x < -5000 for x in d]):
            list_of_lost_times[row[1]['patient_id']] = -1
            continue
        #continue if list contains None or NaN
        if d[0] is None or math.isnan(d[0]):
            print(f"Is None: {row[1]['patient_id']}")
            print(d)
            #continue


        if row[1]['status'] == 1.0:
            color = 'b'
        else:
            color = 'r'
        for i in range(1,len(d)):

            #if d[i-1] == None or NaN but d[i] not, write out i and patient_id to list_of_lost_times
            if (d[i-1] is None or math.isnan(d[i-1])) and not (d[i] is None or math.isnan(d[i])):
                list_of_lost_times[row[1]['patient_id']] = i-1
                break
            if d[i-1] is not None and not math.isnan(d[i-1]):
                try:
                    resp = float(row[1]['response_'][i - 1])
                except ValueError:
                    resp = row[1]['response_'][i - 1]
                try:
                    treat = float(row[1]['treatment_type'][i - 1])
                except ValueError:
                    treat = row[1]['treatment_type'][i - 1]
                if isinstance(resp,str) or resp is None:
                    list_of_lost_repsonse_treatment[row[1]['patient_id']] = -3 * 10 - (i - 1)
                if isinstance(treat,str) or treat is None:
                    list_of_lost_repsonse_treatment[row[1]['patient_id']] = -2 * 10 - (i - 1)

            resp = row[1]['response_'][i - 1]
            treat = row[1]['treatment_type'][i-1]
            if i < len(d):
                if isinstance(row[1]['treatment_type'][i],str):
                    if row[1]['treatment_type'][i] == 'none':
                        continue
                if isinstance(row[1]['response_'][i], str):
                    if row[1]['response_'][i] == 'none':
                        continue

                row[1]['treatment_type'][i] = float(row[1]['treatment_type'][i])
                row[1]['response_'][i] = float(row[1]['response_'][i])

                if row[1]['treatment_type'][i] is not None and not math.isnan(row[1]['treatment_type'][i]):
                    treat_status = 1
                else:
                    treat_status = row[1]['status']
            else:
                treat_status = row[1]['status']

            if isinstance(resp,str):
                try:
                    resp = float(resp)
                except ValueError:
                    pass
            if resp is not None and not isinstance(resp,str) and not math.isnan(resp):
                linet = ['-', '--', '-.', ':'][int(resp)]
                response_sums[i-1][int(resp)] += 1
                if i > 2:
                    last_resp = row[1]['response_'][i - 2]
                    last_treat = row[1]['treatment_type'][i - 2]
                    if last_resp is not None and not isinstance(last_resp, str) and not math.isnan(last_resp):
                        response_last_response[int(last_resp)][resp] += 1
                    if last_treat is not None and not isinstance(last_treat, str) and not math.isnan(last_treat):
                        response_last_treatment[int(last_treat)-1][resp] += 1

            else:
                linet = '-'

            if isinstance(treat,str):
                try:
                    treat = float(treat)
                except ValueError:
                    pass
            if treat is not None and not isinstance(treat,str) and not math.isnan(treat):
                symb = ['o', '*', 'x'][int(treat)-1]
                treatment_type_attempts[i-1][int(treat)-1] += 1

            else:
                symb = 'x'
            dft = d[i]-d[i-1]
            if dft is not None and dft > 0 and not math.isnan(dft) and row[1]['status'] == 1.0:
                disiase_free_times[i-1].append(dft)
                if resp is not None and not isinstance(resp, str) and not math.isnan(resp):
                    disiase_free_times_response[int(resp)].append(dft)
                if treat is not None and not isinstance(treat, str) and not math.isnan(treat):
                    disiase_free_times_treatment[int(treat)-1].append(dft)
            #Calculate number of mutations as sum of all gene_ columns for this row
            number_of_mutation = row[1].filter(regex='gene_').sum()

            new_row_tdf = {"tnum": i, "treatment_time": d[i - 1], "response": resp, "treatment_type": treat,
                           "status": treat_status, "disease_free_time": dft,"patient_id":row[1]['patient_id'],
                           "cancer_stage":row[1]['cancer_stage'], "cancer_type":row[1]['cancer_type'],
                           "smoking":row[1]['smoking'],"alcohol":row[1]['alcohol'],"drugs":row[1]['drugs'],
                           "age_level":row[1]['age_level'], "number_of_mutation":number_of_mutation,"sex": row[1]['sex'],
                           "p16":row[1]['p16'],'race':row[1]['race'],'patient_id':row[1]['patient_id'],'age':row[1]['age']}

            #add data from columns that start from gene_ perfix to dcitionary
            for col in df.filter(regex='gene_').columns:
                new_row_tdf[col] = row[1][col]
            transf_dataset.append(new_row_tdf)
            segment = np.array(row[1]['treatment_time'][i-1:i+1]) # - row[1]['treatment_time'][0]
            #print without new line
            print(f"{i} seg.plot",end=' ')
            plt.plot(segment,[j,j], linet+color)
            plt.plot(segment[0], [j], symb + color, markersize=10)
        print(f"finish")
    plt.grid()
    plt.tight_layout()
    print(list_of_lost_times)
    print(len(list_of_lost_times.keys()))
    print(list_of_lost_repsonse_treatment)
    print(len(list_of_lost_repsonse_treatment.keys()))



    plt.title('Treatment histories of patients (Swimer plot)')
    plt.xlabel('Disease free time (days)')
    plt.ylabel('Patient')
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Complete response'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Partial response'),
        Line2D([0], [0], color='blue', lw=2, linestyle='-.', label='Stable disease'),
        Line2D([0], [0], color='blue', lw=2, linestyle=':', label='Progression'),
        Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='o', label='Chemo'),
        Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='*', label='Imuno'),
        Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='x', label='Other'),
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Pass away'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Alive/no info')

    ]

    # Add the legend to the plot
    plt.legend(custom_lines, [line.get_label() for line in custom_lines])


    plt.show()
    plt.figure()
    for i in range(5):
        plt.bar([1+i/10,2+i/10,3+i/10],treatment_type_attempts[i],label=f'Attempt {i+1}',width=0.1)
    plt.title('Type of treatment on different attempts')
    plt.xlabel('Type of treatment')
    plt.ylabel('Number of patients with this type of treatment')
    #Add x-ticks 1=Chemo, 2=Imuno, 3=Other
    plt.xticks([1,2,3],['Chemo','Imuno','Radio\Surgical'])
    plt.legend()
    
    plt.figure()
    for i in range(5):
        plt.bar([0+i/10,1+i/10,2+i/10,3+i/10],response_sums[i],label=f'Attempt {i+1}',width=0.1)
    plt.title('Response on different attempts')
    plt.xlabel('Response to treatment')
    plt.ylabel('Number of patients with this response to treatment')
    plt.xticks([0, 1, 2, 3], ["Complete\n response ","Partial \n response", 'Stable disease', 'Progression'])
    plt.legend()
    #normalize response_last_response
    for key, value in response_last_response.items():
        s = sum(value.values())
        for k in value.keys():
            if s > 0:
                value[k] = value[k]/s
            else:
                value[k] = 0
    '''
    plt.figure()
    for i in range(4):
        plt.bar([0+i/10,1+i/10,2+i/10,3+i/10],response_last_response[i].values(),label=f'Previouse response {i}',width=0.1)
    plt.title('Response on previouse response')
    plt.xlabel('Previouse response ')
    plt.ylabel('Percent of cases with this response')
    plt.legend()
    plt.figure()
    #normalize response_last_treatment
    for key, value in response_last_treatment.items():
        s = sum(value.values())
        for k in value.keys():
            value[k] = value[k]/s
    '''

    '''
    for i in range(3):
        plt.bar([0 + i / 10, 1 + i / 10, 2 + i / 10, 3 + i / 10], response_last_treatment[i].values(), label=f'Previouse Treatment {i + 1}',
                width=0.1)
    plt.title('Response on different previouse treatment')
    plt.xlabel('Response to treatment')
    plt.ylabel('Percent of cases with this response')
    plt.legend()
    '''
    plt.figure()
    disiase_free_times_lst = [disiase_free_times[i] for i in range(5)]
    plt.boxplot(disiase_free_times_lst)
    kwht_dft = stats.kruskal(*disiase_free_times_lst)
    plt.title(f'Disease free times after different attempts of treatment.\n Kruskal-Wallis H Test p-value: {kwht_dft[1]:.4E}')
    plt.ylabel('Disease free time (days)')
    plt.xlabel('Attempt of treatment')

    plt.figure()
    plt.boxplot(disiase_free_times_response.values())
    kwht_dftr = stats.kruskal(*disiase_free_times_response.values())
    plt.title(f'Disease free times with diffrent response to treatment.\n Kruskal-Wallis H Test p-value: {kwht_dftr[1]:.4E}')
    plt.ylabel('Disease free time (days)')
    plt.xlabel('Response to treatment')
    plt.xticks([1, 2, 3, 4], ["Complete\n response ","Partial \n response", 'Stable disease', 'Progression'])

    plt.figure()
    plt.boxplot(disiase_free_times_treatment.values())
    kwht_dftt = stats.kruskal(*disiase_free_times_treatment.values())
    plt.title(f'Disease free times with diffrent type of treatment.\n Kruskal-Wallis H Test p-value: {kwht_dftt[1]:.4E}')
    plt.ylabel('Disease free time (days)')
    plt.xlabel('Type of treatment')
    plt.xticks([1,2,3],['Chemo','Imuno','Radio\Surgical'])


    tdf = pd.DataFrame(transf_dataset)
    #remove rows with NaN in response or treatment_type
    tdf = tdf.dropna(subset=['response','treatment_type'])
    # if disease_free_time is NaN, set it to abs(treatment_time)
    tdf['disease_free_time'] = tdf['disease_free_time'].fillna(abs(tdf['treatment_time']))
    #if disease_free_time is NaN, drop
    tdf = tdf.dropna(subset=['disease_free_time'])
    #if disease_free_time is less than 0 drop
    tdf = tdf[tdf['disease_free_time'] >= 0]
    # if response is cntains string values replace with int(2)
    tdf['response'] = tdf['response'].apply(lambda x: 2 if isinstance(x,str) else x)
    # response is NaN set to 2
    tdf['response'] = tdf['response'].fillna(2)
    tdf['binary_response'] = tdf.apply(lambda x: 1 if x['response'] < 2.0 or (x['response'] == 2.0 and x['disease_free_time'] < 180) else 0, axis=1)


    tdf.to_csv('tdf.csv',index=False)

    factors = [] # ['response', 'treatment_type','tnum', 'cancer_stage', 'cancer_type', 'smoking', 'alcohol', 'drugs', 'age_level','p16','race']
    for f in factors:
        fig = plot_kaplan_meier2(tdf, column_name=f, status_column='status',
                             survival_in_days='disease_free_time')



    #Create column binary_response. binary_response = 1 if response < 2.0 or response = 2.0 and disease_free_time < 180 else 0


    fig = plot_kaplan_meier2(tdf, column_name='binary_response', status_column='status',
                             survival_in_days='disease_free_time')

    from scipy.stats import fisher_exact

    for therapy in [1.0,2.0,3.0,0.0]:
        if therapy == 0.0:
            tdf_good_response = tdf[(tdf['binary_response'] >= 1.0) ]
            tdf_bad_response = tdf[(tdf['binary_response'] < 1.0) ]
        else:
            tdf_good_response = tdf[(tdf['binary_response'] >= 1.0) & (tdf['treatment_type'] == therapy)]
            tdf_bad_response = tdf[(tdf['binary_response'] < 1.0) & (tdf['treatment_type'] == therapy)]
        #for each columns that starts with gene_ calculate percent of cases with mutation
        #remove records with the same patient_id
        tdf_good_response.sort_values(by=['patient_id','tnum'],inplace=True)
        tdf_bad_response.sort_values(by=['patient_id','tnum'],inplace=True)
        tdf_good_response_u = tdf_good_response.drop_duplicates(subset=['patient_id','treatment_type','binary_response'],keep='first')
        tdf_bad_response_u = tdf_bad_response.drop_duplicates(subset=['patient_id','treatment_type','binary_response'],keep='first')

        perfix = 'gene_'
        tdf_good_response = tdf_good_response.filter(regex=perfix)
        tdf_bad_response = tdf_bad_response.filter(regex=perfix)
        tdf_good_response_u = tdf_good_response_u.filter(regex=perfix)
        tdf_bad_response_u = tdf_bad_response_u.filter(regex=perfix)

        tdf_bad_response = tdf_bad_response_u
        tdf_good_response = tdf_good_response_u
        good_response_gene_true = {}
        bad_response_gene_true = {}
        good_response_gene_false = {}
        bad_response_gene_false = {}
        fisher_results = {}
        p_value_threshold = 0.1
        p_value_threshold2 = 0.05


        for gene_name in tdf_good_response.columns:
            good_response_gene_true[gene_name] = tdf_good_response[gene_name].sum()
            good_response_gene_false[gene_name] = len(tdf_good_response) - good_response_gene_true[gene_name]
            bad_response_gene_true[gene_name] = tdf_bad_response[gene_name].sum()
            bad_response_gene_false[gene_name] = len(tdf_bad_response) - bad_response_gene_true[gene_name]
            #if good_response_gene_true[gene_name] + bad_response_gene_false[gene_name] < 10:
            #    continue
            ftable = [[good_response_gene_true[gene_name], good_response_gene_false[gene_name]] ,
                      [bad_response_gene_true[gene_name], bad_response_gene_false[gene_name]]]
            oddsratio, pvalue = fisher_exact(ftable)
            fisher_results[gene_name] = (oddsratio,pvalue)
            if pvalue < p_value_threshold:
                print(f"Therapy {therapy} gene {gene_name} oddsratio {oddsratio:.4f} pvalue {pvalue:.4f} ")

        #plot fisher results as scatter plot
        fig,ax = plt.subplots()
        pvalue = [x[1] for x in fisher_results.values()]
        oddsratio = [x[0] for x in fisher_results.values()]
        #replace inf in ods ratio with 100
        oddsratio = [10 if x == float('inf') else x for x in oddsratio]
        genes = [x for x in fisher_results.keys()]
        #remove perfix genes_ from gene names
        genes = [x.replace('gene_','') for x in genes]
        ax.scatter(np.log2([x[0] for x in fisher_results.values()]),-np.log10([x[1] for x in fisher_results.values()]))
        ax.set_xlabel('Log2(Odds ratio)')
        ax.set_ylabel('-Log10(P-value)')
        ax.grid()
        #select pvalue  < 0.05 and plot them in red
        significant = pd.DataFrame({'log2(OddsRatio)':np.log2(oddsratio),'-log10(p-value)':-np.log10(pvalue),'name':genes},index=genes)
        significant = significant[significant['-log10(p-value)'] > -np.log10(p_value_threshold)]
        plt.scatter(significant['log2(OddsRatio)'], significant['-log10(p-value)'], color='red')
        #add gene names to the plot
        for i, txt in enumerate(significant.index):
            ax.annotate("  "+txt, (significant['log2(OddsRatio)'][i], significant['-log10(p-value)'][i]),rotation=30*int(i) % 360,fontsize=8)
        #plot horizontal line at pvalue = p_value_threshold
        ax.axhline(-np.log10(p_value_threshold), color='r', linestyle='--')
        #plot text near line with p_value_threshold
        ax.text(0.1,-np.log10(p_value_threshold)+0.02,f'p-value = {p_value_threshold}',rotation=0,fontsize=12)
        ax.axhline(-np.log10(p_value_threshold2), color='r', linestyle='-.')
        # plot text near line with p_value_threshold
        ax.text(0.1, -np.log10(p_value_threshold2) + 0.02, f'p-value = {p_value_threshold2}', rotation=0, fontsize=12)

        # and vertical line at log2(oddsratio) = 0
        ax.axvline(0, color='k', linestyle='-',linewidth=1)
        therapy_lables = ['any', 'Chemo', 'Imuno', 'Radio/surgery']

        plt.title(f'Fisher tests of relation between mutations and response for {therapy_lables[int(therapy)]} therapy')

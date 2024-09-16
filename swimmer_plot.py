import argparse
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create swimmer plot for survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV file", type=str, required=True)
    parser.add_argument("-input_delimiter", help="Delimiter for input file", type=str, default=",")
    parser.add_argument("-survival_column", help="column_with_survival_time", type=str, default="disease_free_time")
    parser.add_argument("-treatment_id", help="ID of treatment", type=str, default="")
    parser.add_argument("-treatment_column", help="column_with_survival_time", type=str, default="")
    parser.add_argument("-response_column", help="column_with_survival_time", type=str, default="")
    parser.add_argument("-status_column", help="column_with_survival_status", type=str, default="status")
    parser.add_argument("-patient_id_column", help="ID of patient", type=str, default="patient_id")
    parser.add_argument("-output_file", help="Output file with swimmer plot", type=str, required=True)
    parser.add_argument("-max_days", help="Maximum days", type=int, required=False, default=365*5)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')

    args = parser.parse_args()
    input_csv = args.input_csv
    input_delimiter = args.input_delimiter
    df = pd.read_csv(input_csv, delimiter=input_delimiter)
    survival_column = args.survival_column
    status_column = args.status_column
    patient_id_column = args.patient_id_column
    treatment_id = args.treatment_id
    treatment_column = args.treatment_column
    response_column = args.response_column
    all_columns = [survival_column, status_column, patient_id_column, treatment_id, treatment_column, response_column]
    for column in all_columns:
        if len(column) > 0 and column not in df.columns:
            raise RuntimeError(f"Column {column} not found in input CSV file")
    if (len(treatment_column) > 0 or len(response_column)>0 ) and len(treatment_id) == 0:
        raise RuntimeError("Treatment ID is required")
    #column overall status of patient. Check all rows with same patient_id and if one of status is 1 overall status = 1
    df['overall_status'] = df.groupby(patient_id_column)[status_column].transform('max')
    #column overall survival - check all rows with same patient_id and select value with greates survival time
    df['overall_survival'] = df.groupby(patient_id_column)[survival_column].transform('sum')


    #todo implement variant without treatment_id
    # sort by patinet_id and treatment_id
    df = df.sort_values(by=['overall_status','overall_survival',patient_id_column, treatment_id])
    y = 0
    x = 0
    x1 = 0
    patinet_id = None
    max_days = args.max_days
    next_patient = False
    are_zero_treatment = False
    are_other_treatment = False
    is_treatment_column = len(treatment_column) > 0
    is_response_column = len(response_column) > 0
    pp = PdfPages(args.output_file)
    fig = plt.figure(figsize=(20, 10))
    for index,value in df.iterrows():
        if patinet_id is None:
            patinet_id = value[patient_id_column]
        elif next_patient and patinet_id == value[patient_id_column]:
            continue
        elif patinet_id != value[patient_id_column]:
            y += 1
            x = 0
            patinet_id = value[patient_id_column]
            next_patient = False
        else:
            x += x1
        x1 = value[survival_column]
        if x+x1 > max_days:
            x1 = max_days - x
            next_patient = True
        color = 'red'
        if value['overall_status'] == 1:
            color = 'blue'
        if len(treatment_column) > 0:
            treat = value[treatment_column]
            if treat in [1,2,3]:
                symb = ['o', '*', 'x'][int(treat) - 1]
            if treat <= 0:
                symb = '+'
                are_zero_treatment = True
            if treat > 3:
                symb = 'X'
                are_other_treatment = True
        else:
            symb = '*'
        if len(response_column) > 0:
            resp = value[response_column]
            if int(resp)<4 and int(resp)>=0:
                linet = ['-', '--', '-.', ':'][int(resp)]
        else:
            linet = '-'

        plot = plt.plot([x, x + x1], [y, y],linet, color=color)
        plot = plt.plot([x], [y],symb,color=color)
        plt.xlabel('Survival time (days)')
        plt.ylabel('Patient')
        plt.title('Swimmer plot')
    custom_lines = []
    if is_treatment_column:
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='o', label='Chemotherapy'))
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='*', label='Imunotherapy'))
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='x', label='Surgical\Radio'))
        if are_zero_treatment:
            custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='+', label='No treatment'))
        if are_other_treatment:
            custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', marker='X', label='Other treatment'))
    if is_response_column:
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Complete response'))
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='--',label='Partial response'))
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-.',label='Stable disease'))
        custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle=':', label='Progressive disease'))
    custom_lines.append(Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Pass away'))
    custom_lines.append(Line2D([0], [0], color='red', lw=2, linestyle='-', label='Alive/no info'))
    plt.legend(custom_lines, [line.get_label() for line in custom_lines])
    pp.savefig(fig)
    pp.close()
    if args.show:
        plt.show()
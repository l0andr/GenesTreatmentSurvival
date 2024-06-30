import pandas as pd
import numpy as np
from typing import List,Optional
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=PerformanceWarning)
def convert_int(x):
    try:
        x = int(x)
    except:
        x = None
    return x

def expand_values_for_patients(initial_df:pd.DataFrame,list_of_expand_marks:List[str],patients_id_columns:str):
    cols_with_dot_or_dash = initial_df.columns[initial_df.isin(list_of_expand_marks).any()].tolist()
    initial_df.replace(list_of_expand_marks, np.nan, inplace=True)
    for col in cols_with_dot_or_dash:
        initial_df[col] = initial_df[col].fillna(initial_df.groupby(patients_id_columns)[col].transform('first'))
    return initial_df

def find_date_columns(df):
    """Identify columns in a DataFrame that contain dates in the Month/Day/Year format."""
    date_cols = []

    # Regular expression pattern for Month/Day/Year format
    pattern = r"^(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})$"

    for col in df.columns:
        # Check if the column is of object type (string)
        t = ""
        try:
            t = df[col].dtype
        except Exception as e:
            continue
        if t == 'object':
            # Check if any entry in the column matches the date pattern
            if df[col].str.match(pattern).any():
                date_cols.append(col)
        elif t == 'datetime64[ns]':
            date_cols.append(col)
    return date_cols

def data_preprocessing(df,last_date_columns:List[str],initial_date_columns:List[str],date_columns_spec:Optional[List[str]]=None,max_survival_length:float=365*5):
    # Remove columns with same values for all patients and all genes

    df.columns = df.columns.str.replace(r'\(.*?\)', '').str.strip()
    unique_counts = df[df.columns].nunique()
    unique_counts = unique_counts[unique_counts <= 1]
    treatment_columns = ['Initial Treatment Part 1', 'Treatment for Recurrence 1', 'Treatment for Recurrence 2',
                         'Treatment for Recurrence 3', 'Treatment for Recurrence 4', 'Treatment for Recurrence 5']
    treatment_response_columns = ['Response to Initial Treatment', 'Best response to therapy 1',
                                  'Best response to therapy 2',
                                  'Best response to therapy 3', 'Best response to therapy 4',
                                  'Best response to therapy 5']
    treatment_dates_columns = ['Date of Initial Tx', 'Recurrence/progression #1 Date',
                               'Recurrence #2 Date', 'Recurrence #3 Date', 'Recurrence #4 Date', 'Recurrence #5 Date']
    to_drop = []
    for c in unique_counts.keys().tolist():
        if c not in treatment_columns and c not in treatment_response_columns and c not in treatment_dates_columns:
            to_drop.append(c)

    df = df.drop(columns=to_drop)
    # rename columns
    df.columns = df.columns.str.replace(r'\s*[\[\(\{].*?[\]\)\}]\s*', '', regex=True)

    warnings.simplefilter('ignore')
    if date_columns_spec is None:
        date_columns_spec = find_date_columns(df)
    date_columns = []
    print(f"1Number of patients {len(df['patient_name'].unique())}")
    for col in date_columns_spec:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                date_columns.append(col)
            except Exception as e:
                continue
    warnings.resetwarnings()
    #print(date_columns)

    df.loc[:, 'max_date'] = df[date_columns].apply(
        lambda row: max([date for date in row if pd.notna(date)], default=np.nan), axis=1)
    date_columns_minus_dob = list(set(date_columns) - set(['dob']))
    df.loc[:, 'min_date'] = df[date_columns_minus_dob].apply(
        lambda row: min([date for date in row if pd.notna(date)], default=np.nan), axis=1)

    df.loc[:,'last_report_date'] = pd.NaT
    last_date_columns.append('max_date')
    print(f"2Number of patients {len(df['patient_name'].unique())}")
    for c in last_date_columns:
        df.loc[df['last_report_date'].isna(), 'last_report_date'] = df[df['last_report_date'].isna()][c]

    df.loc[:,'initial_report_date'] = pd.NaT
    initial_date_columns.append('min_date')
    for c in initial_date_columns:
        df.loc[df['initial_report_date'].isna(), 'initial_report_date'] = df[df['initial_report_date'].isna()][c]

    df.loc[:, 'initial_date'] = pd.NaT
    df.loc[:,'initial_date'] = df['min_date']

    c2 = 'initial_report_date'
    c1 = 'dob'
    df.loc[:, 'Age'] = np.where(df[c1].notna() & df[c2].notna(), (df[c2] - df[c1]).dt.days / 365.25,
                                   np.nan)
    c2 = 'last_report_date'
    c1 = 'initial_date'
    df.loc[:, 'Survival_in_days'] = np.where(df[c1].notna() & df[c2].notna(), (df[c2] - df[c1]).dt.days,
                                                np.nan)
    df.loc[:, 'Status'] = df['SurvivalUPDATED'] == 'N'
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


    c2 = 'First IMTX course end date OR date of progression'
    c1 = 'First IMTX course start date'
    #if df[c2] == current date then immunotherapy is ongoing and date should be set = last_report_date
    df.loc[df[c2] == 'current',c2] = df[df[c2] == 'current']['last_report_date']
    df[c1] = pd.to_datetime(df[c1], errors='coerce')
    df[c2] = pd.to_datetime(df[c2], errors='coerce')
    df.loc[:, 'Length of Immunotherapy'] = np.where(df[c1].notna() & df[c2].notna(), (df[c2] - df[c1]).dt.days + 1,
                                             np.nan)
    # if 'Length of Immunotherapy' is nan then should set 0
    df.loc[df['Length of Immunotherapy'].isna(),'Length of Immunotherapy'] = 0
    # if 'Length of Immunotherapy' more than median then columns ImunoDurationLess = 'long' and 'short' otherwise
    imunno_median = 63 #df.drop_duplicates(subset='patient_name')['Length of Immunotherapy'].median()
    imdur_column_name = f'imuno_duration_levels'
    df.loc[df['Length of Immunotherapy'] > imunno_median, imdur_column_name] = 2
    df.loc[df['Length of Immunotherapy'] <= imunno_median,  imdur_column_name ] = 1
    df.loc[df['Length of Immunotherapy'] < 2, imdur_column_name] = 0
    tmb_percentile_median = df.drop_duplicates(subset='patient_name')['tmb_percentile'].median()
    tmb_percent_column_name = f'tmb_percentile_levels'
    df.loc[df['Length of Immunotherapy'] > tmb_percentile_median, tmb_percent_column_name] = f'over{int(tmb_percentile_median)}'
    df.loc[df['Length of Immunotherapy'] <= tmb_percentile_median, tmb_percent_column_name] = f'less{int(tmb_percentile_median)}'
    print(f"3Number of patients {len(df['patient_name'].unique())}")

    for index,row in df.iterrows():
        #Calculate number of Treatments: find first nan value in treatment_dates_columns
        for i in range(len(treatment_dates_columns)):
            if pd.isna(row[treatment_dates_columns[i]]):
                break   #i is number of treatments
        df.loc[index,'Number of Treatments'] = i



        min_time_interval_days = None
        treatment_date = None
        treatment_type = None
        for j in range(i):
            c2 = 'initial_report_date'
            c1 = treatment_dates_columns[j]
            d2 = row[c2]
            if (row[c1] is None) or (row[c2] is None):
                continue
            if isinstance(row[c1],pd.Timestamp):
                d1 = row[c1]
            else:
                d1 = pd.to_datetime(row[c1],errors='coerce')
            current_time_interval = np.abs((d2 - d1).days)
            if min_time_interval_days is None or current_time_interval < min_time_interval_days:
                min_time_interval_days = current_time_interval
                treatment_date = d1
                treatment_type = row[treatment_columns[j]]
                treatment_response = row[treatment_response_columns[j]]
        df.loc[index, 'sample_treatment_time'] = min_time_interval_days
        df.loc[index, 'treatment_date'] = treatment_date
        df.loc[index, 'treatment_type'] = treatment_type
        df.loc[index, 'treatment_response'] = treatment_response

        #convert treatment_response columns (containing floats 0.0 - 3.0 and None) to int


        recurrence_date = pd.NaT
        recurrence_time_interval = None
        for j in range(i):
            c1 = treatment_dates_columns[j]
            d2 = treatment_date
            if (d2 is None) or (row[c1] is None):
                continue
            if isinstance(row[c1], pd.Timestamp):
                d1 = row[c1]
            else:
                d1 = pd.to_datetime(row[c1], errors='coerce')
            current_time_interval = (d1 - d2).days
            if current_time_interval <= 0:
                continue
            if recurrence_time_interval is None or recurrence_time_interval > current_time_interval:
                recurrence_time_interval= current_time_interval
                recurrence_date = d1

        df.loc[index, 'recurrence_date'] = recurrence_date

    df.loc['treatment_response'] = df['treatment_response'].apply(lambda x: convert_int(x))
    df.loc['Number of Treatments'] = df['Number of Treatments'].apply(lambda x: convert_int(x))
    df.loc['Anatomic stage'] = df['Anatomic stage'].apply(lambda x: convert_int(x))
    #remove all rows with empty patient_name
    # replace None in P16 column by 0
    #convert P16 column to string
    print(f"4Number of patients {len(df['patient_name'].unique())}")
    df.loc['P16+'] = df['P16+'].apply(lambda x: str(x))
    df.loc[df['P16+'].isna(), 'P16+'] = 'N'
    # replace 0 in P16 column by 'N'
    df.loc[df['P16+'] == '0', 'P16+'] = 'N'
    # replace 1 in P16 column by 'Y'
    df.loc[df['P16+'] == '1', 'P16+'] = 'Y'


    df.loc[:, 'recurrence_status'] = True
    df.loc[df['recurrence_date'].isna() & df['Date of Death'].isna(), 'recurrence_status'] = False

    df.loc[df['recurrence_date'].isna() & ~df['Date of Death'].isna(), 'recurrence_date'] = \
        df[df['recurrence_date'].isna() & ~df['Date of Death'].isna()]['Date of Death']
    df.loc[df['recurrence_date'].isna(), 'recurrence_date'] = df[df['recurrence_date'].isna()]['last_report_date']
    c2 = 'treatment_date'
    c1 = 'recurrence_date'
    df.loc[:, 'recurrence_in_days'] = np.where(df[c1].notna() & df[c2].notna(), (df[c1] - df[c2]).dt.days,np.nan)
    df.loc['Prior cancer?'] = df['Prior cancer?'].apply(lambda x: x == 'Y')

    #age upper median
    age_median = df.drop_duplicates(subset='patient_name')['Age'].median()
    age_quantile_75 = df.drop_duplicates(subset='patient_name')['Age'].quantile(0.75)
    df.loc[df['Age'] > age_median, 'age_level'] = 1
    df.loc[df['Age'] <= age_median, 'age_level'] = 0
    df.loc[df['Age'] >= age_quantile_75, 'age_level'] = 2
    #total number of mutation as sum of all columns that start from genes_ prefix
    #if ENE? have value not from list ['Y','N'] then set to None
    df.loc[df['ENE?'].apply(lambda x: x not in ['Y','N']), 'ENE?'] = None
    df.loc[df['PNI?'].apply(lambda x: x not in ['Y', 'N']), 'PNI?'] = None
    df.loc[df['LVSI?'].apply(lambda x: x not in ['Y', 'N']), 'LVSI?'] = None

    columns_rename = {"patient_name": "patient_id",
                      "Age": "age",
                      "age_level":"age_level",
                      "sex":"sex",
                      "Smoking hx?": "smoking",
                      "Alcohol use": "alcohol",
                      "Drug use hx?": "drugs",
                      "Cancer Type. Simple": "cancer_type",
                      "Anatomic stage": "anatomic_stage",
                      "treatment_type": "treatment",
                      "treatment_response": "response",
                      "Status":"status",
                      "Survival_in_days": "survival_in_days",
                      "recurrence_status": "disease-free-status",
                      "recurrence_in_days":"disease-free-time",
                      "Length of Immunotherapy":"immunotherapy_in_days",
                      imdur_column_name:imdur_column_name,
                      tmb_percent_column_name:tmb_percent_column_name,
                      "initial_report_date":"initial_report_date",
                      "initial_date": "initial_date",
                      "Initial Treatment Part 1": "initial_treatment",
                      "min_date":"min_date",
                      "last_report_date":"last_date",
                      "Date of Death":"death_date",
                      "Prior cancer?":"prior_cancer",
                      "treatment_date":"treatment_date",
                      "recurrence_date":"recurrence_date",
                      "Number of Treatments":"number_of_treatments",
                      "cohort":"cohort",
                      "gene":"gene",
                      "P16+": "p16",
                      "race":"race",
                      "ENE?":"ene",
                      "PNI?":"pni",
                      "LVSI?":"lvi",
                      "PDL1 Expression":"pdl1",
                      "PDL-1 Category": "pdl1_category",
                      "Smoking  pack-years": "smoking_packs",
                      "tmb_value":"tmb_value",
                      "msi_status":"msi_status",
                      "tmb_percentile":"tmb_percentile",
                      "tmb":"tmb_level"}

    i = 0
    for tfield in treatment_columns:
        rfield = treatment_response_columns[i]
        tdfield = treatment_dates_columns[i]
        columns_rename[tfield] = f'treatment_type{i}'
        columns_rename[rfield] = f'response_{i}'
        c2 = tdfield
        c1 = 'initial_date'
        #convert column c2 to datetime
        df[c2] = pd.to_datetime(df[c2], errors='coerce')
        df.loc[:, f'treatment_time{i}'] = np.where(df[c1].notna() & df[c2].notna(), (df[c2] - df[c1]).dt.days,
                                                 np.nan)
        columns_rename[f'treatment_time{i}'] = f'treatment_time{i}'
        i += 1
    #in df[race] replace values: 0=white, 1=black, 2=other, 3=unknown
    df = df[df['patient_name'].notna()]
    df_clean = df[list(columns_rename.keys())]
    df_clean = df_clean.rename(columns=columns_rename)
    df_clean.drop_duplicates(subset=['patient_id','gene'],inplace=True)
    replacement_dict = {0: 'white', 1: 'black', 2: 'other', 3: 'unknown'}
    df_clean.loc[:,'race'] = df_clean['race'].replace(replacement_dict)
    replacement_dict = {'0': 'N', '1': 'Y', '': 'unknown'}
    df_clean.loc[:, 'pdl1'] = df_clean['pdl1'].replace(replacement_dict)
    df_pivot = df_clean[['patient_id','gene']].pivot_table(index='patient_id', columns='gene', aggfunc=len, fill_value=0)
    df_pivot.columns = ['gene_' + col for col in df_pivot.columns]
    df_clean.drop_duplicates(subset='patient_id',inplace=True)
    df_clean.drop(columns=['gene'],inplace=True)
    df_clean = df_clean.merge(df_pivot,how='left',on='patient_id')
    df_clean['total_mutations'] = df_clean.filter(like='gene_').sum(axis=1)
    survival_time_col = 'survival_in_days'
    status_col = 'status'
    df_clean.loc[df_clean[survival_time_col] > max_survival_length, status_col] = 0
    df_clean.loc[df_clean[survival_time_col] > max_survival_length, survival_time_col] = max_survival_length

    return df,df_clean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform data from initial csv to csv suitable for survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("-output_csv", help="Output CSV with preprocesed table", type=str, required=True)
    parser.add_argument("-input_delimiter", help="Delimiter for input file", type=str, default=",")
    parser.add_argument("--max_survival_length", help="Maximum consider time interval in Kaplan-meier plots", type=float,
                        default=365*5)

    args = parser.parse_args()
    genes_prefix = 'gene_'
    duplication_symbols = ['-', '.', 'duplicate']
    patient_id_column = 'patient_name'
    min_number_of_cases_of_mutations = 5
    initial_df = pd.read_csv(args.input_csv,delimiter=args.input_delimiter)
    df = expand_values_for_patients(initial_df,list_of_expand_marks=duplication_symbols,patients_id_columns=patient_id_column)
    df,df_clean = data_preprocessing(df, last_date_columns=['Date of Death','Last known f/u'], initial_date_columns=['tumor_sample_collected_date'],max_survival_length=args.max_survival_length)
    df_clean.to_csv(args.output_csv,index=False)

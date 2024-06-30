import argparse
import pandas as pd
import numpy as np
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Transform data from TCGA csv to csv suitable for survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_patient_csv", help="Clinical patient data", type=str, required=True)
    parser.add_argument("-input_genes_csv", help="Genetical patient data", type=str, required=True)
    parser.add_argument("-input_delimiter", help="Delimiter for input file", type=str, default=",")
    parser.add_argument("-list_of_genes", help="List of genes for analysis", type=str, default="")
    parser.add_argument("-output_csv", help="Output CSV file", type=str, default="tcga_data2.csv")
    args = parser.parse_args()
    input_patient_csv = args.input_patient_csv
    input_delimiter = args.input_delimiter
    df = pd.read_csv(input_patient_csv, delimiter=input_delimiter)
    df_genes = pd.read_csv(args.input_genes_csv)
    diagnosis = "Head & Neck Squamous Cell Carcinoma"
    colnames = ['bcr_patient_barcode', 'histologic_diagnosis', 'anatomic_organ_subdivision', 'gender',
       'birth_days_to', 'race', 'margin_status', 'vital_status', 'last_contact_days_to', 'race',
       'death_days_to', 'hpv_status_p16', 'tobacco_smoking_history_indicator',
        'alcohol_history_documented','alcohol_consumption_frequency',
       'age_at_initial_pathologic_diagnosis.x', 'clinical_M', 'clinical_N',
       'clinical_T', 'clinical_stage.x',
       'days_to_initial_pathologic_diagnosis' ]
    list_of_genes = args.list_of_genes.split(',')
    df_filtered = df[df['histologic_diagnosis'] == diagnosis]
    #drop column histological_diagnosis
    df_filtered = df_filtered[colnames]
    df_filtered = df_filtered.drop(columns=['histologic_diagnosis'])
    #fill hpv_16_status with 0 if nan
    df_filtered['p16'] = df_filtered['hpv_status_p16'].fillna(False)
    df_filtered['p16'] = df_filtered['p16'].apply(lambda x: True if x=='Positive' else False)


    #fill alcohol_consumption_frequency with 0 if nan
    df_filtered['alcohol_consumption_frequency'] = df_filtered['alcohol_consumption_frequency'].fillna(0)
    #fill death_days_to with last_contact_days_to if nan
    df_filtered['survival_in_days'] = df_filtered['death_days_to'].fillna(df_filtered['last_contact_days_to'])
    df_filtered.loc[df_filtered['survival_in_days'] == '0','survival_in_days'] = '1'
    #drop last_contact_days_to
    df_filtered = df_filtered.drop(columns=['last_contact_days_to'])
    df_filtered['status'] = df_filtered['vital_status'].apply(lambda x : True if x == 'Dead' else False)

    df_filtered.rename(columns={'gender':'sex'},inplace=True)

    #df_filtered = df_filtered.dropna(axis=0)
    diagnosis_groups_dict = {'oral cavity':['Oral Tongue','Floor of mouth','Buccal Mucosa',
                                            'Alveolar Ridge','Hard Palate','Oral Cavity','Lip'],
                             'oropharynx':['Base of tongue','Tonsil','Oropharynx'],
                             'hypopharynx':['Hypopharynx'],
                             'larynx':['Larynx']}
    def diagnosis_group(x):
        for key,value in diagnosis_groups_dict.items():
            if x in value:
                return key

    df_filtered['diagnosis'] = df_filtered['anatomic_organ_subdivision'].apply(lambda x:diagnosis_group(x))
    df_filtered['smoking'] = df_filtered['tobacco_smoking_history_indicator'].apply(lambda x:True if x == 'smoker' else False)
    df_filtered['alcohol'] = df_filtered['alcohol_consumption_frequency'].apply(lambda x:True if x > 0 else False)
    for index,row in df_filtered.iterrows():
        if row['alcohol_history_documented'] == 'NO':
            df_filtered.loc[index,'alcohol'] = np.nan
    df_filtered['age'] = df_filtered['birth_days_to'].apply(lambda x: np.abs(x)/365.25)
    for gene in list_of_genes:
        df_filtered["gene_"+gene]=False
    #cycle through all rows of table
    for index,row in df_filtered.iterrows():
        patient_id = row['bcr_patient_barcode']
        #filter gene data. Columns Tumor_Sample_Barcode should start from patient_id
        df_genes_filtered = df_genes[df_genes['Tumor_Sample_Barcode'].str.startswith(patient_id)]

        #cycle through all genes
        for gene in list_of_genes:
            #check if this gene is for this patient or no
            df_filtered.loc[index,"gene_"+gene] = gene in df_genes_filtered['Hugo_Symbol'].tolist()
    #compute number of nan values in each column


    col_to_drop = ['anatomic_organ_subdivision','birth_days_to','margin_status',
                   'tobacco_smoking_history_indicator' ,'alcohol_history_documented',
                   'alcohol_consumption_frequency', 'age_at_initial_pathologic_diagnosis.x',
                   'days_to_initial_pathologic_diagnosis','hpv_status_p16','vital_status','death_days_to',
                   'clinical_stage.x','clinical_M','clinical_N','clinical_T','alcohol']
    #drop rows where survival_in_days <=0, convert survival days to int before
    def toint(x):
        try:
            return int(x)
        except:
            return -1
    df_filtered['survival_in_days'] = df_filtered['survival_in_days'].apply(lambda x: toint(x))
    df_filtered = df_filtered[df_filtered['survival_in_days'] > 0]
    df_filtered.drop(columns=col_to_drop,inplace=True)
    print(df_filtered.isnull().sum())
    df_filtered = df_filtered.dropna(axis=0)

    #drop rows with missing values

    #print(pd.unique(df_filtered['anatomic_organ_subdivision']))
    df_filtered.rename(columns={'bcr_patient_barcode':'patient_id'},inplace=True)
    df_filtered.to_csv(args.output_csv, index=False)
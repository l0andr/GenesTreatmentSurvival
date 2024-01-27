#!/bin/bash

# Only one script argument - raw csv file
if [ $# -ne 1 ]; then
    echo "Usage: $0 <raw csv file>"
    exit 1
fi

# Check if input file exists
if [ ! -f $1 ]; then
    echo "Input file $1 does not exist!"
    exit 1
fi

drop_factors='cancer_type,number_of_treatments,response,tmb_value,prior_cancer,smoking,gene_FGF3,gene_FGF4,gene_TERT,gene_KMT2D,immunotherapy_in_days,tmb_percentile'
python prepare_input_data.py -input_csv $1 -output_csv transform1.csv
python EDA_report.py -input_csv transform1.csv -output_csv transform2.csv -output_pdf EDA_report.pdf --verbose 0
python table_transform.py --input transform2.csv --output transform_2_all.csv --delete_columns $drop_factors
python table_transform.py --input transform2.csv --output transform_2_1.csv --delete_columns $drop_factors --filter_column 'treatment' --filter_value '1.0'
python table_transform.py --input transform2.csv --output transform_2_2.csv --delete_columns $drop_factors --filter_column 'treatment' --filter_value '2.0'
python table_transform.py --input transform2.csv --output transform_2_3.csv --delete_columns $drop_factors --filter_column 'treatment' --filter_value '3.0'



pen=0.2
l1ratio=0.1
min_cases=7
python survival_analysis_cox_model.py -input_csv transform_2_all.csv --model_report cox_model_hazards_overall.pdf --status_col status --survival_time_col survival_in_days --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases
python survival_analysis_cox_model.py -input_csv transform_2_all.csv --model_report cox_model_hazards_rec.pdf --status_col disease-free-status --survival_time_col disease-free-time --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases

treat=1
python survival_analysis_cox_model.py -input_csv transform_2_$treat.csv --model_report cox_model_hazards_overall_t$treat.pdf --status_col status --survival_time_col survival_in_days --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases
python survival_analysis_cox_model.py -input_csv transform_2_$treat.csv --model_report cox_model_hazards_rec_t$treat.pdf --status_col disease-free-status --survival_time_col disease-free-time --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases
treat=2
python survival_analysis_cox_model.py -input_csv transform_2_$treat.csv --model_report cox_model_hazards_overall_t$treat.pdf --status_col status --survival_time_col survival_in_days --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases
python survival_analysis_cox_model.py -input_csv transform_2_$treat.csv --model_report cox_model_hazards_rec_t$treat.pdf --status_col disease-free-status --survival_time_col disease-free-time --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases
treat=3
python survival_analysis_cox_model.py -input_csv transform_2_$treat.csv --model_report cox_model_hazards_overall_t$treat.pdf --status_col status --survival_time_col survival_in_days --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases
python survival_analysis_cox_model.py -input_csv transform_2_$treat.csv --model_report cox_model_hazards_rec_t$treat.pdf --status_col disease-free-status --survival_time_col disease-free-time --verbose 0 --l1ratio $l1ratio --penalizer $pen --min_cases $min_cases



python survival_analysis_initial_report.py -input_csv transform_2_all.csv --genes "TP53,CDKN2A,PIK3CA" --factors "p16,alcohol,drugs,sex,cancer_stage,imuno_duration_level,tmb_percentile_levels" -output_pdf kaplan_meier_all.pdf
python survival_analysis_initial_report.py -input_csv transform_2_1.csv --genes "TP53,CDKN2A,PIK3CA" --factors "p16,alcohol,drugs,sex,cancer_stage,imuno_duration_level,tmb_percentile_levels" -output_pdf kaplan_meier_1.pdf
python survival_analysis_initial_report.py -input_csv transform_2_2.csv --genes "TP53,CDKN2A,PIK3CA" --factors "p16,alcohol,drugs,sex,cancer_stage,imuno_duration_level,tmb_percentile_levels" -output_pdf kaplan_meier_2.pdf
python survival_analysis_initial_report.py -input_csv transform_2_3.csv --genes "TP53,CDKN2A,PIK3CA" --factors "p16,alcohol,drugs,sex,cancer_stage,imuno_duration_level,tmb_percentile_levels" -output_pdf kaplan_meier_3.pdf


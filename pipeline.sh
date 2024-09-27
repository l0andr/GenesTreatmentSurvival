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

#python prepare_input_data.py -input_csv $1 -output_csv 2024_transformed.csv
python prepare_tcga_data.py -input_patient_csv TCGA_HNSC/clinical_patient_transformed.csv -input_genes_csv TCGA_HNSC/mutationsTCGA_hnscc.csv -list_of_genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC
python table_transform.py --input 2024_transformed.csv --input2 tcga_data2.csv --cohort_labels UMD,TCGA --output umb_tcga_join.csv
python prepare_treatment_data.py -input_csv 2024_transformed.csv -output_csv tdf.csv
#here should be call of R scripts with tables - TODO add them.
#here should be preparation of opendata - TODO add them.

python swimmer_plot.py -input_csv tdf.csv -output_file figure_1_swimmer_plot.png -treatment_id tnum -treatment_column treatment_type -response_column response --show
python oncoplot.py -input_mutation 2024_transformed.csv -output_file UMB_oncoplot.pdf -list_of_factors sex,smoking,p16,cancer_type,prior_cancer --show --number_of_genes 30 --verbose 3 --title "Oncoplot UMB cohort"
python oncoplot.py -input_mutation tcga_data2.csv -output_file TCGA_oncoplot.pdf -list_of_factors sex,smoking,cancer_type --show --number_of_genes 30 -list_of_genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,B2M,ARID2,ASXL1,CIC --verbose 1 --nosortgenes --title "Oncoplot TCGA cohort"
python cox_analysis.py -input_csv 2024_transformed.csv --genes "" --factors sex,age,p16,smoking,alcohol,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0,alcohol_history --penalizer 0.01 --l1ratio 0.01 --univar "" model_report "cox_overall_factors.pdf"
python cox_analysis.py -input_csv 2024_transformed.csv --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC --factors "" --penalizer 0.01 --l1ratio 0.01 --univar "" model_report "cox_overall_genes.pdf"
python cox_analysis.py -input_csv 2024_transformed.csv --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC --factors sex,age,p16,smoking,alcohol,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0 --penalizer 0.01 --l1ratio 0.01 --univar sex,age,response_0,anatomic_stage model_report "cox_overall_univar.pdf"
python cox_analysis.py -input_csv tcga_data2.csv --genes LRP1B,MTAP,CCND1,FGF4,TP53,PIK3CA,ARID1A --factors age,cancer_type,sex --penalizer 0.01 --l1ratio 0.01 --univar sex,age,cancer_type model_report "cox_overall_tcga.pdf"
python survplots.py --input_csv 2024_transformed.csv --plot kaplan_meier --max_survival_length 2000 --columns gene_CASP8,gene_LRP1B,gene_TP53,gene_MTAP,gene_PIK3CA,gene_ARID1A,response_0,sex,age_level,anatomic_stage,pdl1_category,cancer_type --output_pdf overall_survival_kaplan_meier.pdf --min_size_of_group 0.03
python survplots.py --input_csv tcga_data2.csv --plot kaplan_meier --max_survival_length 2000 --columns gene_CASP8,gene_LRP1B,gene_TP53,gene_MTAP,gene_PIK3CA,gene_ARID1A,sex,anatomic_stage,cancer_type --output_pdf overall_survival_kaplan_meier_tcga.pdf --min_size_of_group 0.03
python cox_analysis.py -input_csv tdf.csv --survival_time_col disease_free_time --status_col status --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,B2M,ARID2,ASXL1,CIC --factors "" --penalizer 0.01 --l1ratio 0.01 --show --univar ""
python cox_analysis.py -input_csv tdf.csv --survival_time_col disease_free_time --status_col status --genes "" --factors sex,age,p16,smoking,alcohol,race,cancer_type,drugs,treatment_type,anatomic_stage,response,tnum --penalizer 0.01 --l1ratio 0.01 --show --univar ""
python cox_analysis.py -input_csv tdf.csv --survival_time_col disease_free_time --status_col status --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,B2M,ARID2,ASXL1,CIC --factors sex,age,p16,smoking,alcohol,race,cancer_type,drugs,treatment_type,anatomic_stage,response,tnum --penalizer 0.01 --l1ratio 0.01 --show --univar "sex,age,tnum,response,cancer_type"

python survplots.py --input_csv tdf.csv --survival_time_col disease_free_time --plot kaplan_meier --max_survival_length 2000 --columns tnum,response,sex,cancer_type,alcohol,drugs,anatomic_stage,gene_FGF4,gene_CDKN2A,gene_MYL1,gene_ARID2 --output_pdf dfs_kaplan_meier.pdf --min_size_of_group 0.03


sex,age,p16,smoking,alcohol,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0

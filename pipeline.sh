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

python prepare_input_data.py -input_csv $1 -output_csv 2024-05-27_transformed.csv
python prepare_tcga_data.py -input_patient_csv TCGA_HNSC/clinical_patient_transformed.csv -input_genes_csv TCGA_HNSC/mutationsTCGA_hnscc.csv -list_of_genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC
python table_transform.py --input 2024-05-27_transformed.csv --input2 tcga_data2.csv --cohort_labels UMD,TCGA --output umb_tcga_join.csv
python prepare_treatment_data.py -input_csv 2024-05-27_transformed.csv -output_csv tdf.csv
python swimmer_plot.py -input_csv tdf.csv -output_file figure_2_swimmer_plot.png -treatment_id tnum -treatment_column treatment_type -response_column response

python cox_analysis.py -input_csv 2024-05-27_transformed.csv --genes TP53,CDKN2A,TERT,FAT1,ZNF750,NOTCH1,LRP1B,ARID1A,FLCN,MYL1,KMT2C,FGFR3,ASXL1,PIK3CA,CCND1 --factors sex,age,p16,smoking,alcohol,race,cancer_type,prior_cancer,drugs,response_0,response_1,treatment_type0,treatment_type1,total_mutations,cancer_stage --penalizer 0.0001 --l1ratio 0.0001 --univar sex,age
python cox_analysis.py -input_csv tdf.csv --show --genes TP53,CDKN2A,TERT,FAT1,ZNF750,NOTCH1,LRP1B,ARID1A,FLCN,MYL1,KMT2C,FGFR3,ASXL1,PIK3CA,CCND1 --factors sex,age,p16,smoking,alcohol,cancer_type,drugs,response,treatment_type,cancer_stage,tnum,race --penalizer 0.01 --l1ratio 0.0001 --univar sex,age,tnum --survival_time_col disease_free_time
python survplots.py --input_csv 2024-05-27_transformed.csv --plot kaplan_meier --max_survival_length 2000 --columns gene_CASP8,gene_LRP1B,gene_TP53,gene_MTAP,gene_PIK3CA,response_0,sex,age,anatomic_stage,pdl1_category --output_pdf overall_survival_kaplan_meier.pdf --min_size_of_group
0.03
python survplots.py --input_csv tdf.csv --survival_time_col disease_free_time --plot kaplan_meier --max_survival_length 2000 --columns tnum,response,sex,gene_TERT,tnum,gene_MYL1 --output_pdf dfs_kaplan_meier.pdf --min_size_of_group 0.03


sex,age,p16,smoking,alcohol,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0

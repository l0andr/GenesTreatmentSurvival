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
output_dir="output"
mkdir -p $output_dir
show=""
#Data preparation
python prepare_input_data.py -input_csv $1 -output_csv 2024_transformed.csv
python prepare_tcga_data.py -input_patient_csv TCGA_HNSC/clinical_patient_transformed.csv -input_genes_csv TCGA_HNSC/mutationsTCGA_hnscc.csv -list_of_genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC
python table_transform.py --input 2024_transformed.csv --input2 tcga_data2.csv --cohort_labels UMD,TCGA --output umb_tcga_join.csv
python prepare_treatment_data.py -input_csv 2024_transformed.csv -output_csv tdf.csv
python table_transform.py --input tdf.csv --output tdf_chemo.csv --filter_column treatment_type --filter_value 1.0
python table_transform.py --input tdf.csv --output tdf_imuno.csv --filter_column treatment_type --filter_value 2.0
python table_transform.py --input tdf.csv --output tdf_radio.csv --filter_column treatment_type --filter_value 3.0
python table_transform.py --input tdf.csv --output tdf_recurrence.csv --filter_column tnum --filter_value "2,3,4,5,6,7"
python table_transform.py --input tdf.csv --output tdf_initial.csv --filter_column tnum --filter_value "1"

#here should be call of R scripts with tables - TODO add them.
#here should be preparation of opendata - TODO add them.

python swimmer_plot.py -input_csv tdf.csv -output_file $output_dir/figure_1_swimmer_plot.pdf -treatment_id tnum -treatment_column treatment_type -response_column response $show -survival_right_labels_column overall_survival -on_therapy_column current_treatment --tiff
python oncoplot.py -input_mutation 2024_transformed.csv -output_file $output_dir/figure_2_UMB_oncoplot.pdf -list_of_factors sex,smoking,p16,cancer_type,prior_cancer $show --number_of_genes 30 --verbose 3 --title "Oncoplot UMB cohort" --tiff
python oncoplot.py -input_mutation tcga_data2.csv -output_file $output_dir/figure_2_TCGA_oncoplot.pdf -list_of_factors sex,smoking,cancer_type $show --number_of_genes 30 -list_of_genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,B2M,ARID2,ASXL1,CIC --verbose 1 --nosortgenes --title "Oncoplot TCGA cohort" --tiff
python cox_analysis.py -input_csv 2024_transformed.csv --genes "" --factors sex,age,p16,smoking,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0,alcohol_history --penalizer 0.01 --l1ratio 0.01 --univar "" $show --model_report $output_dir/figure_3_overall_survival_cox_univariant_factors.pdf --title "overall survival time " --tiff
python cox_analysis.py -input_csv 2024_transformed.csv --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC --factors "" --penalizer 0.01 --l1ratio 0.01 --univar "" $show --model_report $output_dir/figure_3_overall_survival_cox_univariant_genes.pdf --title "overall survival time " --tiff
python cox_analysis.py -input_csv 2024_transformed.csv --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,ARID2,ASXL1,B2M,CIC --factors sex,age,p16,smoking,alcohol_history,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0 --penalizer 0.01 --l1ratio 0.01 --univar sex,age,response_0,anatomic_stage $show --model_report $output_dir/figure_3_overall_survival_cox_fix_univariant.pdf --title "overall survival time " --tiff
python cox_analysis.py -input_csv tcga_data2.csv --genes LRP1B,MTAP,CASP8,PIK3CA,ARID1A --factors age,cancer_type,sex --penalizer 0.02 --l1ratio 0.01 --model_report $figure_3_overall_survival_tcga.pdf --title "TCGA. overall survival time" $show --tiff
python survplots.py --input_csv 2024_transformed.csv --plot kaplan_meier --max_survival_length 2000 --columns gene_CASP8,gene_LRP1B,gene_TP53,gene_MTAP,gene_PIK3CA,gene_ARID1A,response_0,sex,age_level,anatomic_stage,pdl1_category,cancer_type --output_pdf $output_dir/figure_4_overall_survival_kaplan_meier.pdf --min_size_of_group 0.03 --custom_legend km_legend.json --tiff
python survplots.py --input_csv tcga_data2.csv --plot kaplan_meier --max_survival_length 2000 --columns gene_CASP8,gene_LRP1B,gene_TP53,gene_MTAP,gene_PIK3CA,gene_ARID1A,sex,anatomic_stage,cancer_type --output_pdf $output_dir/figure_4_overall_survival_kaplan_meier_tcga.pdf --min_size_of_group 0.01 --custom_legend km_legend.json --tiff
python cox_analysis.py -input_csv tdf.csv --survival_time_col disease_free_time --status_col status --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,B2M,ARID2,ASXL1,CIC --factors "" --penalizer 0.01 --l1ratio 0.01 $show --univar "" --filter_nan --filter_nan_columns treatment_type,response --model_report $output_dir/figure_5_disease_free_survival_cox_univariant_genes.pdf --title "disease free survival time " --tiff
python cox_analysis.py -input_csv tdf.csv --survival_time_col disease_free_time --status_col status --genes "" --factors sex,age,p16,smoking,alcohol_history,race,cancer_type,drugs,treatment_type,anatomic_stage,response,tnum --penalizer 0.01 --l1ratio 0.01 $show --univar "" --filter_nan --filter_nan_columns treatment_type,response --model_report $output_dir/figure_5_disease_free_survival_cox_univariant_factors.pdf --title "disease free survival time " --tiff
python cox_analysis.py -input_csv tdf.csv --survival_time_col disease_free_time --status_col status --genes TP53,CDKN2A,TERT,FAT1,KMT2D,PIK3CA,FGF3,NOTCH1,FGF4,ZNF750,ARID1A,CCND1,LRP1B,CDKN2B,EGFR,KMT2C,CASP8,NFE2L2,CYLD,FBXW7,FLCN,MTAP,MYL1,NOTCH3,SMAD4,SOX2,B2M,ARID2,ASXL1,CIC --factors sex,age,p16,smoking,alcohol_history,race,cancer_type,drugs,treatment_type,anatomic_stage,response,tnum --penalizer 0.01 --l1ratio 0.01 $show --univar "sex,age,tnum,response,cancer_type" --model_report $output_dir/figure_5_disease_free_survival_cox_fix_univariant.pdf --title "disease free survival time " --filter_nan --filter_nan_columns treatment_type,response --tiff
python survplots.py --input_csv tdf.csv --survival_time_col disease_free_time --plot kaplan_meier --max_survival_length 2000 --columns tnum,treatment_group,response,sex,cancer_type,alcohol_history,drugs,anatomic_stage,gene_FGF4,gene_CDKN2A,gene_MYL1,gene_ARID2 --output_pdf $output_dir/figure_6_disease_free_time_kaplan_meier.pdf --min_size_of_group 0.01 --custom_legend km_legend.json --filter_nan_columns treatment_type,response --tiff
python survplots.py --input_csv tdf.csv --status_col binary_response --plot fisher_exact_test --columns gene_* --output_pdf $output_dir/figure_8_all_treatments_fisher_test.pdf --min_size_of_group 0.03 --title "Response-Genes relation. All type of treatments." $show --tiff
python survplots.py --input_csv tdf_chemo.csv --status_col binary_response --plot fisher_exact_test --columns gene_* --output_pdf $output_dir/figure_8_chemo_fisher_test.pdf --min_size_of_group 0.03 --title "Response-Genes relation. Chemotherapy." $show --tiff
python survplots.py --input_csv tdf_imuno.csv --status_col binary_response --plot fisher_exact_test --columns gene_* --output_pdf $output_dir/figure_8_imuno_fisher_test.pdf --min_size_of_group 0.03 --title "Response-Genes relation. Imunotherapy." $show --tiff
python survplots.py --input_csv tdf_radio.csv --status_col binary_response --plot fisher_exact_test --columns gene_* --output_pdf $output_dir/figure_8_radio_fisher_test.pdf --min_size_of_group 0.03 --title "Response-Genes relation. Surgical\Radiotherapy." $show --tiff
python survplots.py --input_csv tdf.csv --survival_time_col disease_free_time --plot kruskal_wallis_test --max_survival_length 2000 --columns tnum,response,treatment_type --output_pdf $output_dir/figure_7_kruskal_wallis_test.pdf --min_size_of_group 0.03 --custom_legend km_legend.json --tiff --filter_nan_columns treatment_type,response
python adaptree.py -input_csv tdf.csv -ycolumn binary_response --xcolumns gene_TP53,gene_ASXL1,gene_AXIN1,gene_MYL1,gene_SOX2,gene_FGF4,sex,cancer_type,gene_ARID2,gene_CDKN2A,gene_PIK3CA,tnum,anatomic_stage,gene_MTAP,gene_CCND1,gene_TERT --verbose 1 --sort_columns patient_id,tnum --unique_record patient_id,treatment_type,binary_response --output_model response_dt_6_max_mean_purity.model --criteria entropy --model response_dt_6_max_mean_purity.model --steps_of_optimization 200 --class_names Bad,Good --filter_nan_columns treatment_type,response --min_samples_leaf 6 --max_depth 6 --plot_type dtreeviz --random_seed 2 --custom_legend km_legend.json --tiff $output_dir/figure_9_disease_free_survival_decision_tree.tiff
python survplots.py --input_csv tdf_initial.csv --survival_time_col disease_free_time --plot kaplan_meier --max_survival_length 2000 --columns tnum,treatment_group,response,sex,cancer_type,alcohol_history,drugs,anatomic_stage,gene_FGF4,gene_CDKN2A,gene_MYL1,gene_ARID2 --output_pdf $output_dir/suppliment_figure_11_disease_free_time_kaplan_meier_initial_treatment_all.pdf --min_size_of_group 0.01 --custom_legend km_legend.json --filter_nan_columns treatment_type,response --tiff
python survplots.py --input_csv tdf_recurrence.csv --survival_time_col disease_free_time --plot kaplan_meier --max_survival_length 2000 --columns tnum,treatment_group,response,sex,cancer_type,alcohol_history,drugs,anatomic_stage,gene_FGF4,gene_CDKN2A,gene_MYL1,gene_ARID2 --output_pdf $output_dir/suppliment_figure_12_disease_free_time_kaplan_meier_recurrence_treatment_all.pdf --min_size_of_group 0.01 --custom_legend km_legend.json --filter_nan_columns treatment_type,response --tiff


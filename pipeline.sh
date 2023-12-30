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

python prepare_input_data.py -input_csv $1 -output_csv transform1.csv
python EDA_report.py -input_csv transform1.csv -output_csv transform2.csv -output_pdf EDA_report.pdf
python survival_analysis_initial_report -input_csv transform2.csv --genes "TP53,CDKN2A" --factors "sex,alcohol" -output_pdf survival_analysis_initial_report.pdf

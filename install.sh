#!/bin/bash
pip install -r requirements.txt

git clone https://github.com/l0andr/pysurvtools.git
pip install -r pysurvtools/requirements.txt
cp pysurvtools/oncoplot.py .
cp pysurvtools/survplots.py .
cp pysurvtools/cox_analysis.py .
cp pysurvtools/adaptree.py .

git clone https://github.com/l0andr/pyswimplot.git
pip install -r pyswimplot/requirements.txt
cp pyswimplot/swimmer_plot.py .

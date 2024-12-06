## Tumor genomics and the association with survival in recurrent/metastatic head and neck cancer patients 

This repository conatins set of scripts for data preparation, exploratory 
data analysis and survival analysis of head and neck cancer patients. 

### Installation

1. Clone repository to your local machine
    ```
    git clone https://github.com/l0andr/HNSCC-Genes-Treatment-Survival-Analysis.git
    cd HNSCC-Genes-Treatment-Survival-Analysis
   
2. Install requirements
    ```
    pip install -r requirements.txt

    git clone https://github.com/l0andr/pysurvtools.git
    pip install -r pysurvtools/requirements.txt
    cp pysurvtools/oncoplot.py .
    cp pysurvtools/survplots.py .
    cp pysurvtools/cox_analysis.py .
    
    git clone https://github.com/l0andr/pyswimplot.git
    pip install -r pyswimplot/requirements.txt
    cp pyswimplot/swimmer_plot.py .
    
    ```
    
### Processing graph

<img src="img/processing_graph_12_05_2024.png" width="1200">

   
### Content

## Script Descriptions

1. **`pipeline.sh`**  
   This script orchestrates the entire data analysis workflow, from data preparation to advanced statistical analysis and visualization, leveraging tools from multiple repositories. The pipeline begins with the transformation of raw input data and sequentially performs data integration, mutation analysis, and survival analysis, producing comprehensive reports and visualizations. Key steps include:

   - **Data Preparation**: Prepares data with `prepare_input_data.py`, `prepare_tcga_data.py`, and `table_transform.py`, converting and merging data from various sources, including clinical and mutation data.
   - **Mutation Analysis**: Generates oncoplots with `oncoplot.py` to visualize mutations across selected genes. This step utilizes tools from the [pysurvtools repository](https://github.com/l0andr/pysurvtools) to create gene mutation summaries for specified cohorts.
   - **Survival Analysis**: Performs survival analysis with `cox_analysis.py` and `survplots.py`, both sourced from the [pysurvtools repository](https://github.com/l0andr/pysurvtools). These scripts generate Kaplan-Meier plots and Cox proportional hazards models to evaluate the impact of clinical and genetic factors on survival outcomes.
   - **Decision Tree Analysis** Perform creating optimal Deciosion tree with adaptree.py from  [pysurvtools repository](https://github.com/l0andr/pysurvtools). This script perform Bayesian optimization of hyperparameters of decision tree model with goal to maximise local and average global purity of resulted tree leafs with constraints for minimal samples supported each leaf and maximal depth of tree.     
   - **Treatment Visualization**: Creates swimmer plots using `swimmer_plot.py` from the [pyswimplot repository](https://github.com/l0andr/pyswimplot), visualizing individual patient treatment timelines, including response types and overall survival.

   The pipeline produces visual reports, including PDF and TIFF outputs of oncoplots, Kaplan-Meier survival curves, and Cox regression analyses, all saved in the specified output directory. 

2. **`prepare_input_data.py`**  
   This script processes raw input data, converting it into a structured format suitable for downstream analyses. It performs data cleaning, validation, and formatting, ensuring that columns and values align with the expected schema for analysis scripts in the pipeline. The output is a standardized CSV file, ready for integration with other data sources.

3. **`prepare_tcga_data.py`**  
   This script prepares mutation and clinical data from The Cancer Genome Atlas (TCGA) to ensure compatibility with the analysis pipeline. It allows users to specify genes of interest, filters the TCGA data to include these genes, and outputs a formatted dataset. This curated dataset is then used in subsequent mutation analysis steps, allowing for detailed insights into gene-specific mutations across a patient cohort.

4. **`prepare_treatment_data.py`**  
   This script organizes treatment-related data, such as disease-free survival time and treatment types, for inclusion in the survival analysis. It processes columns related to treatment response and outcomes, preparing the data for statistical analysis and visualization. The output is a refined dataset that supports comprehensive treatment and survival analysis.

5. **`table_transform.py`**  
   This versatile script performs table transformations, including filtering and joining datasets. It allows for row filtering based on specific column values, merging multiple data tables, and creating combined datasets for analysis. This script is essential for adapting and refining input data, ensuring that only relevant information is passed to the analysis scripts.

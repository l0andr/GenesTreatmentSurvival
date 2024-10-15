## Paper: Tumor genomics and the association with survival in recurrent/metastatic head and neck cancer patients 

This repository conatins set of scripts for data preparation, exploratory 
data analysis and survival analysis of head and neck cancer patients. 

### Installation

1. Clone repository to your local machine
    ```
    git clone
   
2. Install requirements
    ```
    pip install -r requirements.txt
    ```
### Processing graph

Image below shows the processing graph of the analysis pipeline.
<img src="images/processing_graph_10_15_2024.png" width="1200">

   
### Content

pipeline.sh - script for running all analysis steps

prepare_input_data.py - script for convertion raw data to expected data format<br>
prepare_tcga_data.py - script for convertion TCGA data to expected data format<br>
prepare_treatment_data.py - script for preparation disease free time data <br> 
table_transform.py - script for simple tables transformation<br>



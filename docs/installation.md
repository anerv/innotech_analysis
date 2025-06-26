# Installation

````bash
git clone -b main --single-branch https://github.com/anerv/innotech_analysis --depth 1
cd innotech_analysis
````

````bash
conda create -n innotech_analysis geopandas pyyaml pyarrow contextily scikit-learn h3-py seaborn python-duckdb ipykernel 
````


````bash
conda activate innotech
pip install matplotlib-scalebar
pip install --use-pep517 -e .
````
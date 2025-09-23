# Installation

````bash
git clone -b main --single-branch https://github.com/anerv/innotech_analysis --depth 1
cd innotech_analysis
````

````bash
conda create -n innotech_analysis geopandas pyyaml pyarrow contextily scikit-learn h3-py seaborn pysal python-duckdb ipykernel 
````


````bash
conda activate innotech
pip install matplotlib-scalebar
pip install --use-pep517 -e .
pip install generativepy
````


```bash
python setup_folders.py
```

Must be run using wsl!

````bash
chmod +x copy_input_data.sh
bash copy_input_data.sh /mnt/c/Users/anerv/repositories/innotech
````
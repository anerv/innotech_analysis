# %%

import numpy as np
import pandas as pd
import geopandas as gpd

import yaml
from pathlib import Path

from src.helper_functions import (
    combine_columns_from_tables,
    create_hex_grid,
)

import duckdb

# %%
# Define the path to the config.yml file
script_path = Path(__file__).resolve()
root_path = script_path.parent.parent
data_path = root_path / "data"
results_path = root_path / "results"
config_analysis_path = root_path / "config_analysis.yml"
config_model_path = root_path / "config_model.yml"

# Read and parse the YAML file
with open(config_analysis_path, "r") as file:
    config_analysis = yaml.safe_load(file)

    crs = config_analysis["crs"]

with open(config_model_path, "r") as file:
    config_model = yaml.safe_load(file)


otp_db_fp = data_path / "otp_results.db"
duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con.execute("INSTALL spatial;")
duck_db_con.execute("LOAD spatial;")

services = config_model["services"]

# %%

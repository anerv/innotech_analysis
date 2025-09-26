# %%

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path

from src.helper_functions import (
    spatial_weights_combined,
    compute_spatial_autocorrelation,
    compute_lisa,
)
from matplotlib.colors import ListedColormap
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt


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


# %% which columns/variables to use?

# TODO: which data to include?

exec(open(root_path / "src" / "read_analysis_data.py").read())

gdf = hex_travel_times_gdf

id_column = "grid_id"
k_value = 6

columns = [c for c in gdf.columns if "duration_min" in c]
fps_morans = [results_path / "plots" / f"morans_{c}.png" for c in columns]
fps_lisa = [results_path / "maps" / f"lisa_{c}.png" for c in columns]

gdf.dropna(subset=columns, inplace=True)

w = spatial_weights_combined(gdf, id_column, k_value)


morans_results = compute_spatial_autocorrelation(
    columns, columns, gdf, w, fps_morans, show_plot=True
)

lisa_results = compute_lisa(
    columns,
    columns,
    gdf,
    w,
    fps_lisa,
    show_plot=True,
)
# %%

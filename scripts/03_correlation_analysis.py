# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import robust_scale
import geopandas as gpd
import seaborn as sns
import yaml
from pathlib import Path

from IPython.display import display
import duckdb


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
exec(open(root_path / "src" / "read_analysis_data.py").read())

# %%
#### EXAMINE CORRELATION BETWEEN VARIABLES ####
###############################################

analysis_gdf = hex_travel_times_gdf

# %%
analysis_gdf["total_waiting_time"] = analysis_gdf[
    [col for col in analysis_gdf.columns if col.startswith("wait_time_dest_min")]
].sum(axis=1)

analysis_gdf["total_travel_time"] = analysis_gdf[
    [col for col in analysis_gdf.columns if col.startswith("duration_min")]
].sum(axis=1)

analysis_gdf["total_time"] = analysis_gdf[
    [
        col
        for col in analysis_gdf.columns
        if col.startswith("wait_time_dest_min") or col.startswith("duration_min")
    ]
].sum(axis=1)

# %%

analysis_gdf.describe()

# %%
corr_columns = [
    c for c in analysis_gdf.columns if c not in ["source_id", "geometry", "grid_id"]
]
# %%
analysis_gdf[corr_columns].corr().round(2).to_csv(
    results_path / "data/travel_time_correlations.csv", index=True
)

analysis_gdf[corr_columns].corr().round(2).style.background_gradient(
    cmap="RdBu_r", vmin=-1, vmax=1
)

# %%

services = config_model["services"]

for service_dict in services:
    service = service_dict["service_type"]
    service_cols = [col for col in analysis_gdf.columns if service in col]

    display(
        analysis_gdf[service_cols]
        .corr()
        .round(2)
        .style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
    )

# %%
travel_time_columns = ["duration_min", "wait_time_dest_min", "no_connection_count"]
for col in travel_time_columns:
    corr_cols = [c for c in analysis_gdf.columns if col in c]
    display(
        analysis_gdf[corr_cols].corr().round(2).style.background_gradient(cmap="Reds")
    )

# %%

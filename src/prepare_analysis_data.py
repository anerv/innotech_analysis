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

from src.helper_functions import (
    combine_columns_from_tables,
    create_hex_grid,
)
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

# TODO: Use config to choose aggregated or non aggregated data?
# Use when exporting to not overwrite results
# TODO: find ways to drop islands
# TODO: option to drop locations with long wait times
# TODO: config for how to handle no results

####### GET DATA ##########

table_names = [s["service_type"] + "_1" for s in services]

# TODO: INCLUDE TRAVEL MODES??
travel_time_columns = [
    # "waitingTime",
    "walkDistance",
    "abs_dist",
    "duration_min",
    "wait_time_dest_min",
    "total_time_min",
    # "transfers",
]

all_travel_times_gdf = combine_columns_from_tables(
    travel_time_columns,
    duck_db_con,
    table_names,
    "source_id",
    "geometry",
    crs,
)

# %%
# Aggregate by hex grid

# Create hex grid for the study area
study_area = gpd.read_file(config_analysis["study_area_fp"])

hex_grid = create_hex_grid(study_area, 7, crs, 200)

hex_travel_times = gpd.sjoin(
    hex_grid,
    all_travel_times_gdf,
    how="inner",
    predicate="intersects",
    rsuffix="travel",
    lsuffix="hex",
)

hex_travel_times.drop(columns=["index_travel"], inplace=True)

hex_id_col = "grid_id"

cols_to_average = [
    col
    for col in hex_travel_times.columns
    if col.startswith("wait_time_dest_min")
    or col.startswith("duration_min")
    or col.startswith("walkDistance")
    or col.startswith("abs_dist")
    or col.startswith("total_time_min")
]

hex_avg_travel_times = (
    hex_travel_times.groupby(hex_id_col)[cols_to_average].mean().reset_index()
)

hex_avg_travel_times_gdf = hex_grid.merge(
    hex_avg_travel_times, on="grid_id", how="left"
)

# %%
# drop hexagons with no results in them
hex_avg_travel_times_gdf.dropna(subset=["abs_dist_doctor_1"], inplace=True)

# %%

# COUNT NO RESULTS FOR EACH DESTINATION IN EACH HEXAGON

cols_to_include = [
    col for col in hex_travel_times.columns if col.startswith("total_time_min")
]
nan_counts_per_hex = (
    hex_travel_times.groupby(hex_id_col)[cols_to_include]
    .apply(lambda group: group.isna().sum(), include_groups=False)
    .reset_index()
)

nan_counts_per_hex.columns = [hex_id_col] + [
    item.split("total_time_min_")[1] + "_nan_count"
    for item in nan_counts_per_hex.columns[1:]
]

sum_cols = nan_counts_per_hex.columns[1:]
nan_counts_per_hex["total_no_results"] = nan_counts_per_hex[sum_cols].sum(axis=1)

hex_avg_travel_times_gdf = hex_avg_travel_times_gdf.merge(
    nan_counts_per_hex, on=hex_id_col, how="left"
)

# %%

analysis_gdf = hex_avg_travel_times_gdf  # all_travel_times_gdf.copy()

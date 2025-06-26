# %%

# PROCESS RESULTS FROM SERVICE ACCESS ANALYSIS

import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path
import os
import sys
from src.helper_functions import (
    highlight_max_traveltime,
    highlight_min_traveltime,
    unpack_modes_from_json,
    transfers_from_json,
    create_hex_grid,
    compute_weighted_time,
    combine_results,
)


os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)


# Define the path to the config.yml file
script_path = Path(__file__).resolve()
root_path = script_path.parent.parent
data_path = root_path / "data/input"
results_path = root_path / "results"
config_analysis_path = root_path / "config_analysis.yml"
config_model_path = root_path / "config_model.yml"

# Read and parse the YAML file
with open(config_analysis_path, "r") as file:
    config_analysis = yaml.safe_load(file)

    crs = config_analysis["crs"]


with open(config_model_path, "r") as file:
    config_model = yaml.safe_load(file)


# %%
walkspeed_min = config_model["walk_speed"] * 60  # convert to minutes

# Load address data for correct geometries
address_points = gpd.read_parquet(config_model["addresses_fp_all"])

# Load results
services = config_model["services"]

# %%


# Compute weighted travel times based on service importance

# TODO: move to config file
weight_dictionary = {
    "doctor": 0.05,
    "dentist": 0.05,
    "pharmacy": 0.05,
    "kindergarten-nursery": 5,
    "school": 5,
    "supermarket": 5,
    "library": 1,
    "train_station": 5,
    "sports_facility": 1,
}

weight_cols = ["duration_min", "total_time_min"]

for w in weight_cols:

    weighted_travel_times = compute_weighted_time(
        services, 1, results_path, weight_dictionary, w
    )

    weighted_travel_times.to_parquet(
        results_path / f"data/weighted_{w}_otp_geo.parquet",
        index=False,
        engine="pyarrow",
    )


# %%

travel_time_columns = [
    # "waitingTime",
    "walkDistance",
    "abs_dist",
    "duration_min",
    "wait_time_dest_min",
    "total_time_min",
    # "transfers",
]


all_travel_times_gdf = combine_results(
    config_analysis["services"],
    results_path,
    travel_time_columns,
    n_neighbors=1,
)

all_travel_times_gdf["total_time"] = all_travel_times_gdf[
    [col for col in all_travel_times_gdf.columns if col.endswith("total_time_min")]
].sum(axis=1)

# %%

# compute average travel time per hex bin

study_area = gpd.read_file(
    config_analysis["study_area_config"]["regions"]["outputpath"]
)

hex_grid = create_hex_grid(study_area, 8, crs, 200)

hex_travel_times = gpd.sjoin(
    hex_grid,
    all_travel_times_gdf,
    how="inner",
    predicate="intersects",
    rsuffix="travel",
    lsuffix="hex",
)

hex_id_col = "grid_id"

cols_to_average = [
    col for col in hex_travel_times.columns if col.endswith("_total_time_min")
]
cols_to_average.extend(["total_time"])


hex_avg_travel_times = (
    hex_travel_times.groupby(hex_id_col)[cols_to_average].mean().reset_index()
)

hex_avg_travel_times_gdf = hex_grid.merge(
    hex_avg_travel_times, on="grid_id", how="left"
)

hex_avg_travel_times_gdf.to_parquet(
    results_path / "data/hex_avg_travel_times_otp.parquet",
)


# %%

municipalities = gpd.read_parquet(
    config_analysis["study_area_config"]["municipalities"]["outputpath"]
)

id_column = config_analysis["study_area_config"]["municipalities"]["id_column"]
municipalities = municipalities[["geometry", id_column]]

regional_travel_times = gpd.sjoin(
    municipalities,
    all_travel_times_gdf,
    how="inner",
    predicate="intersects",
    rsuffix="travel",
    lsuffix="region",
)


cols_to_average = [
    col for col in regional_travel_times.columns if col.endswith("_total_time_min")
]
cols_to_average.extend(["total_time"])

municipal_avg_travel_times = (
    regional_travel_times.groupby(id_column)[cols_to_average].mean().reset_index()
)

municipal_avg_travel_times_gdf = municipalities.merge(
    municipal_avg_travel_times, on=id_column, how="left"
)

municipal_avg_travel_times_gdf.to_parquet(
    results_path / "data/municipal_avg_travel_times_otp.parquet",
)


# %%

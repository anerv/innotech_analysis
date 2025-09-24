# %%

import numpy as np
import pandas as pd
import geopandas as gpd
import duckdb
import yaml
from pathlib import Path

from src.helper_functions import (
    safe_wkb_load,
)

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
    drop_islands = config_analysis.get("drop_islands", False)

    islands_fp = config_analysis.get("islands_fp", None)

with open(config_model_path, "r") as file:
    config_model = yaml.safe_load(file)


otp_db_fp = data_path / "otp_results.db"
duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con.execute("INSTALL spatial;")
duck_db_con.execute("LOAD spatial;")

services = config_model["services"]

# %%

hex_travel_times = duck_db_con.execute("SELECT * FROM hex_travel_times;").fetchdf()

hex_travel_times["geometry"] = hex_travel_times["geom_wkb"].apply(safe_wkb_load)

hex_travel_times = hex_travel_times.drop(columns=["geom_wkb"])

hex_travel_times_gdf = gpd.GeoDataFrame(hex_travel_times, geometry="geometry", crs=crs)

# %%
# Select columns
keep_values = [
    "duration_min",
    "wait_time_dest_min",
    "total_time_min",
    "abs_dist",
    "no_connection_count",
]
keep_columns = [
    c for c in hex_travel_times_gdf.columns if any(k in c for k in keep_values)
]

keep_columns.extend(["geometry", "grid_id", f"total_observations"])

hex_travel_times_gdf = hex_travel_times_gdf[keep_columns]

# drop rows with no values

hex_travel_times_gdf = hex_travel_times_gdf[
    hex_travel_times_gdf["total_observations"].notna()
]

# %%
hex_grid_total = duck_db_con.execute(
    "SELECT *, ST_AsWKB(geometry) as geom_wkb FROM hex_total_travel_times;"
).fetchdf()
hex_grid_total["geometry"] = hex_grid_total["geom_wkb"].apply(safe_wkb_load)
hex_grid_total = hex_grid_total.drop(columns=["geom_wkb"])
hex_grid_total_gdf = gpd.GeoDataFrame(hex_grid_total, geometry="geometry", crs=crs)

# %%
if drop_islands:

    islands = gpd.read_parquet(islands_fp)

    assert islands.crs == crs, "CRS of islands does not match the analysis CRS"

    intersection = hex_travel_times_gdf.sjoin(
        islands, how="inner", predicate="intersects"
    )

    # drop the rows from hex_travel_times_gdf that DO have a match in islands
    hex_travel_times_gdf = hex_travel_times_gdf[
        ~hex_travel_times_gdf["grid_id"].isin(intersection["grid_id"])
    ]

    intersection2 = hex_grid_total_gdf.sjoin(
        islands, how="inner", predicate="intersects"
    )

    # drop the rows from hex_grid_total_gdf that DO have a match in islands
    hex_grid_total_gdf = hex_grid_total_gdf[
        ~hex_grid_total_gdf["grid_id"].isin(intersection2["grid_id"])
    ]


print(
    "Analysis data read successfully, ready as hex_travel_times_gdf and hex_grid_total_gdf"
)
# %%

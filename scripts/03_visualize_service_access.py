# %%
from src.helper_functions import (
    plot_traveltime_results,
    plot_no_connection,
)
import yaml
import sys
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)

# Define the path to the config.yml file
script_path = Path(__file__).resolve()
root_path = script_path.parent.parent
data_path = root_path / "data/processed/destinations"
results_path = root_path / "results"
config_path = root_path / "config.yml"

# Read and parse the YAML file
with open(config_path, "r") as file:
    config_model = yaml.safe_load(file)

    crs = config_model["crs"]

# %%

# load study area for plotting

study_area = gpd.read_file(config_model["study_area_fp"])

services = config_model["services"]

for service in services:

    for i in range(1, int(service["n_neighbors"]) + 1):
        dataset = f"{service['service_type']}_{i}"

        gdf = gpd.read_parquet(
            results_path / f"data/{dataset}_addresses_otp_geo.parquet"
        )
        # Process each dataset

        plot_columns = [
            "duration_min",
            "wait_time_dest_min",
            "total_time_min",
        ]

        labels = ["Travel time (min)", "Wait time (min)", "Total duration (min)"]

        attribution_text = "KDS, OpenStreetMap"
        font_size = 10

        for i, plot_col in enumerate(plot_columns):
            fp = results_path / f"maps/{dataset}_{plot_col}.png"

            title = f"{labels[i]} to {dataset.split("_")[-1]} nearest {dataset.split("_")[0]} by public transport"

            plot_traveltime_results(
                gdf,
                plot_col,
                attribution_text,
                font_size,
                title,
                fp,
            )

        no_results = gdf[(gdf["duration"].isna()) & (gdf.abs_dist > 0)].copy()
        if not no_results.empty:
            fp_no_results = results_path / f"maps/{dataset}_no_results.png"
            title_no_results = f"Locations with no results for {dataset.split('_')[-1]} nearest {dataset.split('_')[0]} by public transport"

            plot_no_connection(
                no_results,
                study_area,
                attribution_text,
                font_size,
                title_no_results,
                fp_no_results,
            )

# %%

# TODO: PLOT HEX travel time results

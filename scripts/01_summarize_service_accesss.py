# %%

# PROCESS RESULTS FROM SERVICE ACCESS ANALYSIS

import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path
import os
import sys
import duckdb

from src.helper_functions import (
    highlight_max_traveltime,
    highlight_min_traveltime,
    create_hex_grid,
    compute_weighted_time,
    export_gdf_to_duckdb_spatial,
    combine_columns_from_tables,
    export_gdf_to_duckdb_spatial,
)


os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
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

    service_weights = config_analysis["service_weights"]


with open(config_model_path, "r") as file:
    config_model = yaml.safe_load(file)


walkspeed_min = config_model["walk_speed"] * 60  # convert to minutes

# Load results
otp_db_fp = data_path / "otp_results.db"

services = config_model["services"]

for service in services:
    service_type = service["service_type"]
    if service_type not in service_weights:
        raise ValueError(
            f"Service type '{service_type}' not found in service_weights. Please check the configuration."
        )


# %%
# DubckDB connection

# Open a persistent DuckDB database file Data is stored in Duckdb and then exported to parquet
duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con.execute("INSTALL spatial;")
duck_db_con.execute("LOAD spatial;")

# %%

# summarizing results for each service

summaries = []

for service in services:

    for i in range(1, int(service["n_neighbors"]) + 1):
        dataset = f"{service['service_type']}_{i}"
        # Process each dataset

        print(f"Processing result dataset: {dataset}")
        fp = data_path / f"input/{dataset}_otp_geo.parquet"
        if not fp.exists():
            print(f"File {fp} does not exist. Skipping.")
            continue
        gdf = gpd.read_parquet(fp)
        print(f"Loaded {len(gdf)} rows from {fp}")

        result_count = gdf[gdf["duration_min"].notna()].shape[0]
        print(f"{result_count} solutions found in {dataset} with {len(gdf)} rows.")

        # Count sources with no results
        no_results_count = gdf[gdf["duration_min"].isna()].shape[0]
        if no_results_count > 0:
            print(
                f"{no_results_count} sources have no results in {dataset}. This may indicate that the search window was too small or that no transit solution is available."
            )

        ave_duration = gdf["duration_min"].mean()
        print(f"Average trip duration for {dataset}: {ave_duration:.2f} minutes")

        ave_wait_time = gdf["wait_time_dest_min"].mean()
        print(
            f"Average wait time at destination for {dataset}: {ave_wait_time:.2f} minutes"
        )

        export_gdf_to_duckdb_spatial(gdf, duck_db_con, dataset)

        print("\n")

        # Export min, mean, max, and median duration and wait time
        summary = {
            "dataset": dataset,
            "min_duration": float(f"{gdf['duration_min'].min():.2f}"),
            "mean_duration": float(f"{gdf['duration_min'].mean():.2f}"),
            "max_duration": float(f"{gdf['duration_min'].max():.2f}"),
            "median_duration": float(f"{gdf['duration_min'].median():.2f}"),
            "min_wait_time": float(f"{gdf['wait_time_dest_min'].min():.2f}"),
            "mean_wait_time": float(f"{gdf['wait_time_dest_min'].mean():.2f}"),
            "max_wait_time": float(f"{gdf['wait_time_dest_min'].max():.2f}"),
            "median_wait_time": float(f"{gdf['wait_time_dest_min'].median():.2f}"),
            "median_transfers": int(gdf["transfers"].median()),
            "max_transfers": int(gdf["transfers"].max()),
        }

        summaries.append(summary)

# Convert summaries to DataFrame
summary_df = pd.DataFrame(summaries)
summary_df.set_index("dataset", inplace=True)

rows_to_style = [
    "mean_duration",
    "max_duration",
    "median_duration",
    "mean_wait_time",
    "max_wait_time",
    "median_wait_time",
    "median_transfers",
    "max_transfers",
]


styled_table = (
    summary_df.T.style.apply(
        highlight_max_traveltime,
        axis=1,
        subset=(
            rows_to_style,
            summary_df.T.columns,
        ),  # style only these rows for all columns
    )
    .apply(
        highlight_min_traveltime,
        axis=1,
        subset=(
            rows_to_style,
            summary_df.T.columns,
        ),  # style only these rows for all columns
    )
    .format("{:.2f}")
    .set_table_styles(
        [
            {"selector": "th", "props": [("font-weight", "bold")]},
        ]
    )
    .set_properties(**{"text-align": "left", "font-size": "12px", "width": "100px"})
    .set_caption("Travel times and wait times for public transport to nearest services")
    .set_table_attributes('style="width: 50%; border-collapse: collapse;"')
)


summary_df.to_csv(
    results_path / "data/service_access_summary.csv", index=True, float_format="%.2f"
)

styled_table.to_html(
    results_path / "data/service_access_summary.html",
    table_attributes='style="width: 50%; border-collapse: collapse;"',
)

styled_table

# %%
# Compute weighted travel times based on service importance

weight_cols = ["duration_min", "total_time_min"]

for w in weight_cols:

    weighted_travel_times = compute_weighted_time(
        services, 1, data_path / "input", service_weights, w
    )

    weighted_travel_times.to_parquet(
        results_path / f"data/weighted_{w}_otp_geo.parquet",
        index=False,
        engine="pyarrow",
    )


# TODO: Visualize  weighted travel times on map
# TODO: make histogram of weighted travel times
# %%

# Combine results from all services into a single GeoDataFrame

# TODO: Use duckdb and separate script to combine results

# column_names = ["duration_min", "wait_time_dest_min", "total_time_min"]
# table_names = [s["service_type"] + "_1" for s in services]


# combined_gdf = combine_columns_from_tables(
#     column_names,
#     conn=duck_db_con,
#     table_names=table_names,
#     common_id_column="source_id",
# )


# total_col = "travel_time_total_min"
# sum_cols = [col for col in combined_gdf.columns if col.startswith("total_time_min")]
# combined_gdf[total_col] = combined_gdf[sum_cols].sum(axis=1)

# %%

# compute average travel time per hex bin

# TODO: HOW TO HANDLE LOCATIONS WITH NO RESULTS?

study_area = gpd.read_file(config_analysis["study_area_fp"])

# TODO: load aggregated data from helper script
# TODO: visualize aggregated travel and wait times on map
# TODO: also include only walk, modes, no solutions, etc


# aggregation_type = "ave_travel_times"

# hex_grid = create_hex_grid(study_area, 6, crs, 200)

# hex_travel_times = gpd.sjoin(
#     hex_grid,
#     combined_gdf,
#     how="inner",
#     predicate="intersects",
#     rsuffix="travel",
#     lsuffix="hex",
# )

# hex_id_col = "grid_id"

# cols_to_average = [
#     col for col in hex_travel_times.columns if col.startswith("total_time_min")
# ]
# cols_to_average.extend([total_col])


# hex_avg_travel_times = (
#     hex_travel_times.groupby(hex_id_col)[cols_to_average].mean().reset_index()
# )

# hex_avg_travel_times_gdf = hex_grid.merge(
#     hex_avg_travel_times, on="grid_id", how="left"
# )

# hex_avg_travel_times_gdf.to_parquet(
#     results_path / f"data/hex_{aggregation_type}_otp.parquet",
# )

# %%
# # count no results per hex bin
# cols_to_include = [
#     col for col in hex_travel_times.columns if col.startswith("total_time_min")
# ]
# nan_counts_per_hex = (
#     hex_travel_times.groupby(hex_id_col)[cols_to_include]
#     .apply(lambda group: group.isna().sum(), include_groups=False)
#     .reset_index()
# )

# nan_counts_per_hex.columns = [hex_id_col] + [
#     item.split("total_time_min_")[1] + "_nan_count"
#     for item in nan_counts_per_hex.columns[1:]
# ]

# sum_cols = nan_counts_per_hex.columns[1:]
# nan_counts_per_hex["total_no_results"] = nan_counts_per_hex[sum_cols].sum(axis=1)

# nan_counts_per_hex_gdf = hex_grid.merge(nan_counts_per_hex, on=hex_id_col, how="left")

# nan_counts_per_hex_gdf.to_parquet(
#     results_path / "data/hex_nan_counts_per_service.parquet"
# )


# %%

# TODO: HOW TO HANDLE LOCATIONS WITH NO RESULTS?

# TODO: load aggregated data from helper script
# TODO: visualize aggregated travel and wait times on map
# TODO: also include only walk, modes, no solutions, etc

municipalities = gpd.read_parquet(config_analysis["municipalities_fp"])

muni_id_col = config_model["study_area_config"]["municipalities"]["id_column"]
municipalities = municipalities[["geometry", muni_id_col]]

municipal_travel_times = gpd.sjoin(
    municipalities,
    combined_gdf,
    how="inner",
    predicate="intersects",
    rsuffix="travel",
    lsuffix="region",
)

cols_to_average = [
    col for col in municipal_travel_times.columns if col.startswith("total_time_min")
]
cols_to_average.extend([total_col])

municipal_avg_travel_times = (
    municipal_travel_times.groupby(muni_id_col)[cols_to_average].mean().reset_index()
)

municipal_avg_travel_times_gdf = municipalities.merge(
    municipal_avg_travel_times, on=muni_id_col, how="left"
)

municipal_avg_travel_times_gdf.to_parquet(
    results_path / f"data/municipal_{aggregation_type}_otp.parquet",
)

# %%
# count no results per municipality
cols_to_average = [
    col for col in municipal_travel_times.columns if col.startswith("total_time_min")
]

nan_counts_per_muni = (
    municipal_travel_times.groupby(muni_id_col)[cols_to_average]
    .apply(lambda group: group.isna().sum(), include_groups=False)
    .reset_index()
)


nan_counts_per_muni.columns = [muni_id_col] + [
    item.split("total_time_min_")[1] + "_nan_count"
    for item in nan_counts_per_muni.columns[1:]
]

sum_cols = nan_counts_per_muni.columns[1:]
nan_counts_per_muni["total_no_results"] = nan_counts_per_muni[sum_cols].sum(axis=1)

nan_counts_per_muni_gdf = municipalities[[muni_id_col, "geometry"]].merge(
    nan_counts_per_muni, on=muni_id_col, how="left"
)

nan_counts_per_muni_gdf.to_parquet(
    results_path / "data/muni_nan_counts_per_service.parquet"
)

# %%

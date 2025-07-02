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
    find_k_elbow_method,
    test_kmeans_with_different_seeds,
    visualize_clusters,
    compare_clusterings,
    run_kmeans,
    examine_cluster_results,
    style_cluster_means,
    create_hex_grid,
    plot_silhouette_scores,
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

# %%
#### EXAMINE CORRELATION BETWEEN VARIABLES ####
###############################################

analysis_gdf["total_waiting_time"] = analysis_gdf[
    [col for col in analysis_gdf.columns if col.startswith("wait_time_dest_min")]
].sum(axis=1)

analysis_gdf["total_travel_time"] = analysis_gdf[
    [col for col in analysis_gdf.columns if col.startswith("duration_min")]
].sum(axis=1)

analysis_gdf["total_time"] = analysis_gdf[
    [col for col in analysis_gdf.columns if col.startswith("total_time_min")]
].sum(axis=1)

# %%

analysis_gdf.describe()

# %%
corr_columns = [
    c for c in analysis_gdf.columns if c not in ["source_id", "geometry", "grid_id"]
]

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

for col in travel_time_columns:
    corr_cols = [c for c in analysis_gdf.columns if col in c]
    display(
        analysis_gdf[corr_cols].corr().round(2).style.background_gradient(cmap="Reds")
    )

# %%

#################### CLUSTERING ####################
###################################################
cluster_cols = [
    c
    for c in analysis_gdf.columns
    if any(
        substring in c
        for substring in [
            # "abs_dist",
            # "walkDistance",
            "duration_min",
            "wait_time_dest",
            "nan_count",
        ]
    )
]

# TODO: HOW TO HANDLE NAN VALUES?

kmeans_data = analysis_gdf.copy()
kmeans_data.dropna(subset=cluster_cols, inplace=True)

scaled_data = robust_scale(kmeans_data[cluster_cols])

# %%
### Find appropriate number of clusters

# Elbow method
m1, m2 = find_k_elbow_method(scaled_data, min_k=1, max_k=30)

for key, val in m1.items():
    print(f"{key} : {val:.2f}")

# Silhouette score
# NOTE: Set sample size for large data sets
plot_silhouette_scores(
    scaled_data, k_values=range(6, 12), iterations=10, sample_size=None, global_seed=42
)

# NOTE: DEFINE K HERE
k = 8

# %%
### Test effect of different seed points
seeds = [13, 27, 42, 99]
seeds_test = test_kmeans_with_different_seeds(k, scaled_data, seeds=seeds, n_runs=10)

# Visualize the clusters
visualize_clusters(scaled_data, seeds_test, seeds)

# Compare the clusterings
compare_clusterings(seeds_test, seeds)

# NOTE: Define seed here - ensure that seed effect is negligible
seed = 13

# %%

##### RUNNING K-Means #######

cluster_col = "kmeans"
fp_kde = "../results/plots/kde_cluster_distributions.png"
fp_map = "../results/maps/kmeans_clusters_map.png"
fp_size = "../results/plots/kmeans_clusters_size.png"

k_labels = run_kmeans(k, scaled_data)

kmeans_data[cluster_col] = k_labels

cluster_means = examine_cluster_results(
    kmeans_data,
    cluster_col,
    cluster_cols,
    fp_kde=fp_kde,
    fp_map=fp_map,
    fp_size=fp_size,
    cmap="tab20",
    palette="tab20",
)

style_cluster_means(cluster_means)

cluster_means.to_csv(
    results_path / "data/kmeans_cluster_means.csv", index=True, float_format="%.2f"
)

# %%

# TODO: Look into PCA

# TODO: Look into hbscan and other methods not sentitive to NaN values

# TODO: correlate clusters and/or travel times with urban areas


# %%


# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import robust_scale
import geopandas as gpd
from scipy.spatial.distance import cdist
import seaborn as sns
import yaml
from pathlib import Path
import os
import sys
from src.helper_functions import combine_results
from IPython.display import display

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
#  k-means clustering

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
    config_model["services"],
    results_path,
    travel_time_columns,
    n_neighbors=1,
)


all_travel_times_gdf["total_waiting_time"] = all_travel_times_gdf[
    [col for col in all_travel_times_gdf.columns if col.endswith("_wait_time_dest_min")]
].sum(axis=1)

all_travel_times_gdf["total_travel_time"] = all_travel_times_gdf[
    [col for col in all_travel_times_gdf.columns if col.endswith("_duration_min")]
].sum(axis=1)

all_travel_times_gdf["total_time"] = all_travel_times_gdf[
    [col for col in all_travel_times_gdf.columns if col.endswith("_total_time_min")]
].sum(axis=1)
# %%

#### EXAMINE CORRELATION BETWEEN VARIABLES ####

all_travel_times_gdf.describe()

corr_columns = [
    c for c in all_travel_times_gdf.columns if c not in ["source_id", "geometry"]
]

all_travel_times_gdf[corr_columns].corr().round(2).style.background_gradient(
    cmap="coolwarm"
)

# %%

services = config_model["services"]

for service_dict in services:
    service = service_dict["service_type"]
    service_cols = [col for col in all_travel_times_gdf.columns if service in col]

    display(
        all_travel_times_gdf[service_cols]
        .corr()
        .round(2)
        .style.background_gradient(cmap="coolwarm")
    )

# %%

for col in travel_time_columns:
    corr_cols = [c for c in all_travel_times_gdf.columns if col in c]
    display(
        all_travel_times_gdf[corr_cols]
        .corr()
        .round(2)
        .style.background_gradient(cmap="Reds")
    )

# %%


def run_kmeans(k, scaled_data, seed=13):

    np.random.seed(seed)

    kmeans = KMeans(n_clusters=k)
    k_class = kmeans.fit(scaled_data)

    return k_class.labels_


def get_mean_cluster_variables(gdf, cluster_col, cluster_variables):

    cluster_means = gdf.groupby(cluster_col)[cluster_variables].mean()

    cluster_means = cluster_means.T.round(3)

    return cluster_means


# TODO: implement function for silhouette coefficient to evaluate clustering quality: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html


def find_k_elbow_method(input_data, min_k=1, max_k=10):
    # Based on https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(min_k, max_k)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(input_data)

        distortions.append(
            sum(
                np.min(
                    cdist(input_data, kmeanModel.cluster_centers_, "euclidean"),
                    axis=1,
                )
            )
            / input_data.shape[0]
        )
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = (
            sum(
                np.min(
                    cdist(input_data, kmeanModel.cluster_centers_, "euclidean"),
                    axis=1,
                )
            )
            / input_data.shape[0]
        )
        mapping2[k] = kmeanModel.inertia_

    plt.plot(K, distortions, "bx-")
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.title("Elbow Method: Find best K")
    plt.show()

    return mapping1, mapping2


def plot_cluster_variable_distributions(
    gdf,
    cluster_col,
    cluster_variables,
    fp,
    palette="Set1",
):
    """
    Plot the distributions of cluster variables using kernel density estimation (KDE).

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing the data.
    cluster_col (str): The name of the column representing the clusters.
    cluster_variables (list): The list of variables to plot.
    fp (str): The file path to save the plot.
    palette (str, optional): The color palette to use for the plot.".

    Returns:
    None
    """
    tidy_df = gdf.set_index(cluster_col)
    tidy_df = tidy_df[cluster_variables]
    tidy_df = tidy_df.stack()
    tidy_df = tidy_df.reset_index()
    tidy_df = tidy_df.rename(columns={"level_1": "Attribute", 0: "Values"})

    sns.set_theme(font_scale=1.5, style="white")

    facets = sns.FacetGrid(
        data=tidy_df,
        col="Attribute",
        hue=cluster_col,
        sharey=False,
        sharex=False,
        aspect=2,
        col_wrap=3,
        palette=palette,
    )
    fig = facets.map(
        sns.kdeplot,
        "Values",
        fill=False,
        warn_singular=False,
        multiple="stack",
    )

    fig.add_legend(title="Cluster")
    fig.savefig(fp)


# %%

# TODO: scale/standardize variables?

# TODO: use elbow method to determine optimal number of clusters

# TODO: cluster data using k-means

# TODO: plot clusters and KDE of cluster distribution

# Define K!
k = 7

##### K-Means #######

gdf = None
cluster_vars = None
cluster_col = "kmeans"
palette = "Set1"
fp_kde = "../results/plots/kde_cluster_distributions.png"

# Use robust_scale to norm cluster variables
scaled_data = robust_scale(gdf[cluster_vars])

# Find appropriate number of clusters
m1, m2 = find_k_elbow_method(scaled_data, min_k=1, max_k=20)

for key, val in m1.items():
    print(f"{key} : {val:.2f}")


k_labels = run_kmeans(k, scaled_data)

gdf[cluster_col] = k_labels


cluster_means = get_mean_cluster_variables(gdf, cluster_col, cluster_vars)

plot_cluster_variable_distributions(
    gdf, cluster_col, cluster_vars, fp_kde, palette=palette
)

# %%

# TODO: correlate clusters and/or travel times with urban areas

# %%

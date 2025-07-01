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
#### EXAMINE CORRELATION BETWEEN VARIABLES ####
###############################################

all_travel_times_gdf["total_waiting_time"] = all_travel_times_gdf[
    [
        col
        for col in all_travel_times_gdf.columns
        if col.startswith("wait_time_dest_min")
    ]
].sum(axis=1)

all_travel_times_gdf["total_travel_time"] = all_travel_times_gdf[
    [col for col in all_travel_times_gdf.columns if col.startswith("duration_min")]
].sum(axis=1)

all_travel_times_gdf["total_time"] = all_travel_times_gdf[
    [col for col in all_travel_times_gdf.columns if col.startswith("total_time_min")]
].sum(axis=1)

# %%

all_travel_times_gdf.describe()

# %%
corr_columns = [
    c for c in all_travel_times_gdf.columns if c not in ["source_id", "geometry"]
]

all_travel_times_gdf[corr_columns].corr().round(2).style.background_gradient(
    cmap="RdBu_r", vmin=-1, vmax=1
)

# %%
all_travel_times_gdf[corr_columns].corr().round(2).to_csv(
    results_path / "data/travel_time_correlations.csv", index=True
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
        .style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
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
#################### CLUSTERING ####################
###################################################
cluster_cols = [
    c
    for c in all_travel_times_gdf.columns
    if any(
        substring in c
        for substring in ["abs_dist", "walkDistance", "duration_min", "wait_time_dest"]
    )
]

# TODO: HOW TO HANDLE NAN VALUES?

kmeans_data = all_travel_times_gdf.copy()
kmeans_data.dropna(subset=cluster_cols, inplace=True)

scaled_data = robust_scale(kmeans_data[cluster_cols])
# %%

# Find appropriate number of clusters
m1, m2 = find_k_elbow_method(scaled_data, min_k=1, max_k=30)

for key, val in m1.items():
    print(f"{key} : {val:.2f}")

# %%
# NOTE: DEFINE K HERE
k = 12
# %%
seeds = [13, 27, 42, 99]
seeds_test = test_kmeans_with_different_seeds(k, scaled_data, seeds=seeds, n_runs=10)

# Visualize the clusters
visualize_clusters(scaled_data, seeds_test, seeds)

# Compare the clusterings
compare_clusterings(seeds_test, seeds)

# %%
# NOTE: Define seed here - ensure that seed effect is negligible
seed = 13
# %%


def run_kmeans(k, scaled_data, seed=13):

    np.random.seed(seed)

    kmeans = KMeans(n_clusters=k)
    k_class = kmeans.fit(scaled_data)

    return k_class.labels_


##### RUNNING K-Means #######

cluster_col = "kmeans"
palette = "Set1"
fp_kde = "../results/plots/kde_cluster_distributions.png"

k_labels = run_kmeans(k, scaled_data)

kmeans_data[cluster_col] = k_labels


# %%


def get_mean_cluster_variables(gdf, cluster_col, cluster_variables):

    cluster_means = gdf.groupby(cluster_col)[cluster_variables].mean()

    cluster_means = cluster_means.T.round(3)

    return cluster_means


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


cluster_means = get_mean_cluster_variables(kmeans_data, cluster_col, cluster_cols)

plot_cluster_variable_distributions(
    kmeans_data, cluster_col, cluster_cols, fp_kde, palette=palette
)

# TODO: plot cluster means


# %%

# TODO: Look into PCA

# TODO: Look into hbscan and other methods not sentitive to NaN values


# %%

# TODO: correlate clusters and/or travel times with urban areas


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def plot_silhouette_scores(data, k_values, iterations=5):
    """
    Compute and plot silhouette scores for different k-values over multiple iterations of k-means.

    Parameters:
    - data: numpy array, the input data to cluster.
    - k_values: list of int, different numbers of clusters to try.
    - iterations: int, number of iterations for each k-value.
    """
    silhouette_scores = np.zeros((len(k_values), iterations))

    for i, k in enumerate(k_values):
        for j in range(iterations):
            kmeans = KMeans(n_clusters=k, random_state=j)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores[i, j] = silhouette_avg

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot(silhouette_scores.T, labels=k_values)
    plt.title("Silhouette Scores for Different k-values")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


plot_silhouette_scores(scaled_data, k_values=range(10, 15))
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed


def compute_silhouette_score(data, k, iteration):
    kmeans = KMeans(n_clusters=k, random_state=iteration)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg


def plot_silhouette_scores(data, k_values, iterations=5):
    silhouette_scores = np.zeros((len(k_values), iterations))

    # Use parallel processing to compute silhouette scores
    def process_k(k, index):
        scores = Parallel(n_jobs=-1)(
            delayed(compute_silhouette_score)(data, k, j) for j in range(iterations)
        )
        silhouette_scores[index] = scores

    Parallel(n_jobs=-1)(delayed(process_k)(k, i) for i, k in enumerate(k_values))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot(silhouette_scores.T, labels=k_values)
    plt.title("Silhouette Scores for Different k-values")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


plot_silhouette_scores(scaled_data, k_values=range(10, 15))

# %%
from sklearn.preprocessing import StandardScaler

x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)  # normalizing the features

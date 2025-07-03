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

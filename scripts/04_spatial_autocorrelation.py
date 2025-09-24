# %%

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path

from src.helper_functions import (
    combine_columns_from_tables,
    create_hex_grid,
)
from matplotlib.colors import ListedColormap
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt

from pysal.explore import esda
from pysal.lib import weights
from splot.esda import lisa_cluster
from IPython.display import display
import math

# %%
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
def compute_spatial_weights(
    gdf, na_columns, w_type, dist=1000, k=6, silence_warnings=False
):
    """
    Wrapper function for computing the spatial weights for the analysis/results grids.
    ...

    Arguments:
        gdf (geodataframe): geodataframe with polygons
        na_columns (list): list of columns used to drop NA rows. If no rows should be dropped, just use e.g. 'grid_index'
        w_type (str): type of spatial weight to compute. Only supports KNN, distance band or queens.
        dist (numeric): distance to use if distance band
        k (int): number of nearest neighbors if using KNN

    Returns:
        w (pysal spatial weight object): the spatial weight object
    """
    gdf.dropna(subset=na_columns, inplace=True)

    # convert to centroids
    cents = gdf.centroid

    # Extract coordinates into an array
    pts = pd.DataFrame({"X": cents.x, "Y": cents.y}).values

    if w_type == "dist":
        w = weights.distance.DistanceBand.from_array(
            pts, dist, binary=False, silence_warnings=silence_warnings
        )

    elif w_type == "knn":
        w = weights.distance.KNN.from_array(pts, k=k, silence_warnings=silence_warnings)

    elif w_type == "queen":
        w = weights.contiguity.Queen.from_dataframe(
            gdf, use_index=False, silence_warnings=silence_warnings
        )

    else:
        print("no valid type defined")

        pass

    # row standardize
    w.transform = "R"

    return w


def spatial_weights_combined(gdf, id_column, k=3, silence_warnings=True):

    w_queen = compute_spatial_weights(
        gdf, id_column, "queen", silence_warnings=silence_warnings
    )
    w_knn = compute_spatial_weights(
        gdf, id_column, w_type="knn", k=k, silence_warnings=silence_warnings
    )
    w = weights.set_operations.w_union(
        w_queen, w_knn, silence_warnings=silence_warnings
    )

    assert len(w.islands) == 0

    return w


def compute_lisa(
    col_names, variable_names, gdf, spatial_weights, filepaths, p=0.05, show_plot=True
):
    # based on https://geographicdata.science/book/notebooks/07_local_autocorrelation.html

    """
    Wrapper function for computing and plotting local spatial autocorrelation.
    ...

    Arguments:
        col_names (list of str): names of cols for which local spatial autocorrelation should be computed
        variable_names (list of str): name of variables (only to avoid using potentially long or confusing column names for print statements etc.)
        gdf (geodataframe): geodataframe with polygon data set
        spatial_weights (pysal spatial weight object): the spatial weight object used in the computation
        filepaths (list or str): list of filepaths for storing the plots
        p (float): the desired pseudo p-value

    Returns:
        lisas (dict): dictionary with pysal lisas objects for all columns/variables
    """

    lisas = {}

    significance_labels = {}

    for i, c in enumerate(col_names):
        v = variable_names[i]

        lisa = esda.moran.Moran_Local(gdf[c], spatial_weights)

        lisas[v] = lisa

        sig = 1 * (lisa.p_sim < p)

        spots = lisa.q * sig

        # Mapping from value to name (as a dict)
        spots_labels = {
            0: "Non-Significant",
            1: "HH",
            2: "LH",
            3: "LL",
            4: "HL",
        }
        gdf[f"{v}_q"] = pd.Series(spots, index=gdf.index).map(spots_labels)

        f, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axs = axs.flatten()

        ax = axs[0]

        gdf.assign(Is=lisa.Is).plot(
            column="Is",
            cmap="plasma",
            scheme="quantiles",
            k=2,
            edgecolor="white",
            linewidth=0.1,
            alpha=0.75,
            legend=True,
            ax=ax,
        )

        ax = axs[1]

        lisa_cluster(lisa, gdf, p=1, ax=ax)

        ax = axs[2]
        labels = pd.Series(1 * (lisa.p_sim < p), index=gdf.index).map(
            {1: "Significant", 0: "Non-Significant"}
        )
        gdf.assign(cl=labels).plot(
            column="cl",
            categorical=True,
            k=2,
            cmap="Paired",
            linewidth=0.1,
            edgecolor="white",
            legend=True,
            ax=ax,
        )

        significance_labels[v] = labels

        ax = axs[3]
        lisa_cluster(lisa, gdf, p=p, ax=ax)

        for z, ax in enumerate(axs.flatten()):
            ax.set_axis_off()
            ax.set_title(
                [
                    "Local Statistics",
                    "Scatterplot Quadrant",
                    "Statistical Significance",
                    "Moran Cluster Map",
                ][z],
                y=0,
            )

        f.suptitle(
            f"Local Spatial Autocorrelation for differences in: {v}", fontsize=16
        )

        f.tight_layout()

        f.savefig(filepaths[i])

        if show_plot:
            plt.show()

        plt.close()

    return lisas


def compute_spatial_autocorrelation(
    col_names,
    variable_names,
    df,
    spatial_weights,
    filepaths,
    show_plot=True,
    print_results=True,
):
    """
    Wrapper function for computing and plotting global spatial autocorrelation (Moran's I)
    ...

    Arguments:
        col_names (list of str): names of cols for which local spatial autocorrelation should be computed
        variable_names (list of str): name of variables (only to avoid using potentially long or confusing column names for print statements etc.)
        df (dataframe/geodataframe): dataframe or geodataframe with data
        spatial_weights (pysal spatial weight object): the spatial weight object used in the computation
        filepaths (list or str): list of filepaths for storing the plots

    Returns:
        morans (dict): dictionary with pysal morans objects for all columns/variables
    """
    morans = {}

    for i, c in enumerate(col_names):
        v = variable_names[i]

        # compute spatial lag
        df[f"{v}_lag"] = weights.spatial_lag.lag_spatial(spatial_weights, df[c])

        fig, ax = plt.subplots(1, figsize=(5, 5))

        sns.regplot(
            x=c,
            y=f"{v}_lag",
            ci=None,
            data=df,
            line_kws={"color": "r"},
            scatter_kws={"alpha": 0.4},
            color="black",
        )

        ax.axvline(0, c="k", alpha=0.5)
        ax.axhline(0, c="k", alpha=0.5)
        ax.set_title(f"Moran Plot - {v}")

        if show_plot:
            plt.show()

        moran = esda.moran.Moran(df[c], spatial_weights)

        if print_results:
            print(
                f"With significance {moran.p_sim}, the Moran's I value for {v} is {moran.I:.2f}"
            )

        morans[v] = moran

        fig.savefig(filepaths[i])

        plt.close()

    return morans


# def compare_lisa_results(fp, metric, rename_dict, format_style):
#     """
#     Compare LISA (Local Indicators of Spatial Association) results.

#     Parameters:
#     - fp (str): Filepath of the data file.
#     - metric (str): The metric to compare.
#     - aggregation_level (str): The level of aggregation.
#     - rename_dict (dict): A dictionary to rename the columns.
#     - format_style (function): A function to format the output style.

#     Returns:
#     None
#     """

#     summary = {}

#     gdf = gpd.read_parquet(fp)

#     cols = [
#         c for c in gdf.columns if c not in ["geometry", "hex_id", "municipality", "id"]
#     ]

#     for c in cols:
#         summary[c] = gdf[c].value_counts().to_dict()

#     dfs = []
#     for c in summary.keys():
#         df = pd.DataFrame.from_dict(summary[c], orient="index", columns=[c])

#         dfs.append(df)

#     joined_df = pd.concat(dfs, axis=1)

#     new_col_names = [c.strip("_q") for c in joined_df.columns]
#     new_columns_dict = {}
#     for z, c in enumerate(joined_df.columns):
#         new_columns_dict[c] = new_col_names[z]
#     joined_df.rename(columns=new_columns_dict, inplace=True)

#     joined_df.rename(columns=rename_dict, inplace=True)

#     print(f"LISA summary for {metric}")

#     display(joined_df.style.pipe(format_style))

#     long_df = joined_df.reset_index().melt(
#         id_vars="index", var_name="Metric", value_name="Count"
#     )

#     # Create the stacked bar chart
#     # TODO: replace plotly with matplotlib or seaborn
#     fig = px.bar(
#         long_df,
#         x="Metric",
#         y="Count",
#         color="index",
#         title=f"LISA for {metric}",
#         labels={"index": "LISA Type", "Count": "Count", "Metric": "Metric"},
#         hover_data=["Metric", "index", "Count"],
#         color_discrete_map={
#             "Non-Significant": "#d3d3d3",
#             "HH": "#d62728",
#             "HL": "#e6bbad",
#             "LH": "#add8e6",
#             "LL": "#1f77b4",
#         },
#     )

#     # Show the figure
#     fig.show()


# def process_plot_moransi(fp, metric, aggregation_level, rename_dict):
#     """
#     Process and plot Moran's I for a given metric at a specified aggregation level.

#     Args:
#         fp (str): The file path of the JSON file containing the data.
#         metric (str): The name of the metric.
#         aggregation_level (str): The aggregation level.
#         rename_dict (dict): A dictionary to rename the index of the DataFrame.

#     Returns:
#         pandas.DataFrame: The processed DataFrame.

#     """
#     df = pd.read_json(
#         fp,
#         orient="index",
#     )

#     df.rename(columns={0: f"morans I: {aggregation_level}"}, inplace=True)

#     df.rename(
#         index=rename_dict,
#         inplace=True,
#     )

#     # TODO: replace plotly with matplotlib or seaborn
#     fig = px.bar(
#         df.reset_index(),
#         x="index",
#         y=f"morans I: {aggregation_level}",
#         title=f"Moran's I for {metric} at {aggregation_level}",
#         labels={
#             "index": "Metric type",
#         },
#     )

#     fig.show()

#     plt.close()

#     return df


def color_list_to_cmap(color_list):
    colors = {i: color_list[i] for i in range(len(color_list))}
    return ListedColormap([t[1] for t in sorted(colors.items())])


def plot_significant_lisa_clusters_all(
    gdf,
    plot_columns,
    titles,
    figsize=(12, 8),
    fontsize=12,
    colors=["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6", "lightgrey"],
    fp=None,
    dpi=300,
    legend_pos=(0.95, 0.95),
):
    custom_cmap = color_list_to_cmap(colors)

    row_num = math.ceil(len(plot_columns) / 2)

    fig, axes = plt.subplots(nrows=row_num, ncols=2, figsize=figsize)

    axes = axes.flatten()

    if len(plot_columns) % 2 != 0:
        fig.delaxes(axes[-1])

    for i, p in enumerate(plot_columns):

        ax = axes[i]

        gdf.plot(
            column=p,
            categorical=True,
            legend=True,
            linewidth=0.0,
            ax=ax,
            edgecolor="none",
            legend_kwds={
                "frameon": False,
                "loc": "upper right",
                "bbox_to_anchor": legend_pos,
                "fontsize": fontsize,
            },
            cmap=custom_cmap,
        )

        ax.set_axis_off()
        ax.set_title(titles[i], fontsize=fontsize)

    fig.tight_layout()

    if fp:
        fig.savefig(fp, bbox_inches="tight", dpi=dpi)


# %% which columns/variables to use?

# TODO: which data to include?
# TODO: which columns/variables to use?

exec(open(root_path / "src" / "read_analysis_data.py").read())

gdf = hex_travel_times_gdf

id_column = "grid_id"
k_value = 6

w = spatial_weights_combined(gdf, id_column, k_value)
# %%

columns = None
fps_morans = None
fps_lisa = None

morans_results = compute_spatial_autocorrelation(
    columns, columns, gdf, w, fps_morans, show_plot=False
)

lisa_results = compute_lisa(
    columns,
    columns,
    gdf,
    w,
    fps_lisa,
    show_plot=False,
)

# compare_lisa_results(
#     fp,
#     metric="network reach",
#     aggregation_level=aggregation_levels[-1],
#     rename_dict=rename_dicts[2],
#     format_style=format_style_index,
# )

plot_significant_lisa_clusters_all(
    gdf_density,
    plot_columns=plot_columns,
    titles=titles,
    fp=fp,
    legend_pos=(0.95, 0.95),
)

df = process_plot_moransi(fp, metric, a, rename_dicts[i])

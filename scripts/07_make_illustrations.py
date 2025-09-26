# %%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import math
import itertools
import duckdb
from matplotlib.colors import rgb2hex
from src.helper_functions import (
    plot_traveltime_results,
    plot_no_connection,
    plot_histogram,
    map_results_user_defined,
    load_gdf_from_duckdb,
    export_gdf_to_duckdb_spatial,
    normalize_data,
    make_bivariate_choropleth_map,
    hex_to_color,
    create_color_grid,
)
import yaml
import sys
import os
from pathlib import Path

# %%

# Define the path to the config.yml file
script_path = Path(__file__).resolve()
root_path = script_path.parent.parent
data_path = root_path / "data"
results_path = root_path / "results"
config_path = root_path / "config_model.yml"
config_analysis_path = root_path / "config_analysis.yml"

with open(config_analysis_path, "r") as file:
    config_analysis = yaml.safe_load(file)

    crs = config_analysis["crs"]
    drop_islands = config_analysis.get("drop_islands", False)

    islands_fp = config_analysis.get("islands_fp", None)

# Read and parse the YAML file
with open(config_path, "r") as file:
    config_model = yaml.safe_load(file)

    crs = config_model["crs"]

otp_db_fp = root_path / "data" / "otp_results.db"
duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con.execute("INSTALL spatial;")
duck_db_con.execute("LOAD spatial;")
# %%

# load study area for plotting

study_area = gpd.read_file(config_analysis["study_area_fp"])

services = config_model["services"]

for service in services:

    for i in range(1, int(service["n_neighbors"]) + 1):
        dataset = f"{service['service_type']}_{i}"

        gdf = gpd.read_parquet(data_path / f"input/{dataset}_addresses_otp_geo.parquet")

        # TODO: DROP ISL
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
# plot maps of total travel times on hex grid
fontsize = 12

hex_tt = load_gdf_from_duckdb(duck_db_con, f"hex_total_travel_times", crs)

if drop_islands:
    islands = gpd.read_parquet(islands_fp)

    assert islands.crs == crs, "CRS of islands does not match the analysis CRS"

    intersection = hex_tt.sjoin(islands, how="inner", predicate="intersects")

    # drop the rows from hex_travel_times_gdf that DO have a match in islands
    hex_tt = hex_tt[~hex_tt["grid_id"].isin(intersection["grid_id"])]

columns = ["avg_total_duration", "avg_total_wait_time", "avg_total_time"]

for c in columns:

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="3.5%", pad="1%")
    cax.tick_params(labelsize=fontsize)

    hex_tt.plot(
        column=c,
        legend=True,
        cmap="viridis",
        ax=ax,
        cax=cax,
    )

    for spine in cax.spines.values():
        spine.set_visible(False)

    ax.set_axis_off()

    ax.add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower left",
            box_alpha=0,
            font_properties={"size": fontsize},
        )
    )

    ax.set_title(f"{c.replace("_"," ").title()} (min.)", fontsize=fontsize + 2)
    plt.tight_layout()
    plt.savefig(
        results_path / f"maps/hex_map_{c}.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
# plot weighted travel times based on service importance for hex grid

weight_cols = ["duration_min", "total_time_min"]
labels = ["Travel Time", "Total Time (incl. wait time)"]
fontsize = 12

for w in weight_cols:
    hex_w = load_gdf_from_duckdb(duck_db_con, f"hex_weighted_{w}", crs)

    if drop_islands:
        islands = gpd.read_parquet(islands_fp)

        assert islands.crs == crs, "CRS of islands does not match the analysis CRS"

        intersection = hex_w.sjoin(islands, how="inner", predicate="intersects")

        # drop the rows from hex_travel_times_gdf that DO have a match in islands
        hex_w = hex_w[~hex_w["grid_id"].isin(intersection["grid_id"])]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="3.5%", pad="1%")
    cax.tick_params(labelsize=fontsize)

    hex_w.plot(
        column="avg_total_weighted_time",
        legend=True,
        cmap="viridis",
        ax=ax,
        cax=cax,
    )

    for spine in cax.spines.values():
        spine.set_visible(False)

    ax.set_axis_off()

    ax.add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower left",
            box_alpha=0,
            font_properties={"size": fontsize},
        )
    )

    ax.set_title(
        f"Average Weighted {labels[weight_cols.index(w)]}", fontsize=fontsize + 2
    )
    plt.tight_layout()
    plt.savefig(
        results_path
        / f"maps/hex_weighted_map_{labels[weight_cols.index(w)].replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
# population density map

# Population
figsize_map = (15, 10)
dpi = 300
fontsize_legend = 12

pop_gdf = gpd.read_file("../data/processed/voting_areas.gpkg")

filepath = "../illustrations/socio_population_density"

fig, ax = plt.subplots(figsize=figsize_map)

pop_gdf.plot(
    ax=ax,
    scheme="quantiles",  # "quantiles",  # "natural_breaks",
    k=5,
    column="population_density",
    cmap="viridis",
    linewidth=0.0,
    edgecolor="none",
    legend=True,
    legend_kwds={
        # "fmt": "{:.0f}",
        "frameon": False,
        "fontsize": fontsize_legend,
    },
)

# Access the legend
legend = ax.get_legend()

new_labels = []
for text in legend.get_texts():
    label = text.get_text()  # Extract the label text

    label_split = label.split(",")

    first_val = label_split[0]
    second_val = label_split[1].strip(" ")

    new_labels.append(
        f"{int(round(float(first_val), -1))}"
        + ", "
        + f"{int(round(float(second_val), -1))}"
    )

# Update the legend text
for text, new_label in zip(legend.get_texts(), new_labels):
    text.set_text(new_label)

ax.set_axis_off()
# ax.set_title("Population density (people/km²)", fontsize=pdict["map_title_fs"])

ax.add_artist(
    ScaleBar(
        dx=1,
        units="m",
        dimension="si-length",
        length_fraction=0.15,
        width_fraction=0.002,
        location="lower left",
        box_alpha=0,
        font_properties={"size": fontsize_legend},
    )
)
cx.add_attribution(
    ax=ax,
    text="(C) " + "Statistics Denmark",
    font_size=fontsize_legend,
)
txt = ax.texts[-1]
txt.set_position([0.99, 0.01])
txt.set_ha("right")
txt.set_va("bottom")

plt.tight_layout()

fig.savefig(filepath + ".png", dpi=dpi)

# %%

socio_corr_variables = [
    "0-17 years (%)",
    "18-29 years (%)",
    "30-39 years (%)",
    "40-49 years (%)",
    "50-59 years (%)",
    "60-69 years (%)",
    "70+ years (%)",
    "Students (%)",
    "Income under 150k (%)",
    # "Income under 100k (%)",
    # "Income 100-150k (%)",
    "Income 150-200k (%)",
    "Income 200-300k (%)",
    "Income 300-400k (%)",
    "Income 400-500k (%)",
    "Income 500-750k (%)",
    "Income 750k+ (%)",
    "Household income 50th percentile",
    "Household income 80th percentile",
    "Households w car (%)",
    "Households 1 car (%)",
    "Households 2 cars (%)",
    "Households no car (%)",
    "Population density",
    "Urban area (%)",
]

socio_age_vars = socio_corr_variables[:7]
socio_income_vars = socio_corr_variables[7:17]
socio_car_vars = socio_corr_variables[17:21]
socio_urban_pop_vars = socio_corr_variables[21:]
socio_car_pop = socio_corr_variables[17:]

if "households_income_under_100k_pct" in pop_gdf.columns:
    pop_gdf["households_income_under_150k_pct"] = (
        pop_gdf["households_income_under_100k_pct"]
        + pop_gdf["households_income_100_150k_pct"]
    )

population_rename_dict = {
    "-17_pct": "0-17 years (%)",
    "18-29_pct": "18-29 years (%)",
    "30-39_pct": "30-39 years (%)",
    "40-49_pct": "40-49 years (%)",
    "50-59_pct": "50-59 years (%)",
    "60-69_pct": "60-69 years (%)",
    "70-_pct": "70+ years (%)",
    "student_pct": "Students (%)",
    "households_income_under_150k_pct": "Income under 150k (%)",
    "households_income_under_100k_pct": "Income under 100k (%)",
    "households_income_100_150k_pct": "Income 100-150k (%)",
    "households_income_150_200k_pct": "Income 150-200k (%)",
    "households_income_200_300k_pct": "Income 200-300k (%)",
    "households_income_300_400k_pct": "Income 300-400k (%)",
    "households_income_400_500k_pct": "Income 400-500k (%)",
    "households_income_500_750k_pct": "Income 500-750k (%)",
    "households_income_750k_pct": "Income 750k+ (%)",
    "households_with_car_pct": "Households w car (%)",
    "households_1car_pct": "Households 1 car (%)",
    "households_2cars_pct": "Households 2 cars (%)",
    "households_nocar_pct": "Households no car (%)",
    "households_income_50_percentile": "Household income 50th percentile",
    "households_income_80_percentile": "Household income 80th percentile",
    "population_density": "Population density",
    "urban_pct": "Urban area (%)",
}

pop_gdf.rename(columns=population_rename_dict, inplace=True)

fontsize_title = 12
fontsize_legend = 10

for label, plot_columns in zip(["income", "cars"], [socio_income_vars, socio_car_vars]):

    filepath = f"../illustrations/socio_vars_{label}"

    row_num = math.ceil(len(plot_columns) / 3)

    height = row_num * 3 + 1
    width = 10
    figsize = (width, height)

    fig, axes = plt.subplots(nrows=row_num, ncols=3, figsize=figsize)

    axes = axes.flatten()

    rmv_idx = len(plot_columns) - len(axes)
    for r in range(rmv_idx, 0):
        fig.delaxes(axes[r])

    for i, p in enumerate(plot_columns):

        ax = axes[i]

        pop_gdf.plot(
            column=p,
            legend=True,
            linewidth=0.0,
            ax=ax,
            edgecolor="none",
            scheme="natural_breaks",
            k=5,
            legend_kwds={
                "frameon": False,
                "loc": "upper right",
                # "bbox_to_anchor": legend_pos,
                "fontsize": fontsize_legend,
                "fmt": "{:.0f}",
            },
            cmap="cividis",
        )

        # # Update the legend text
        # for text, new_label in zip(legend.get_texts(), new_labels):
        #     text.set_text(new_label)

        ax.set_axis_off()
        ax.set_title(p, fontsize=fontsize_title)

    fig.tight_layout()

    fig.savefig(filepath, bbox_inches="tight", dpi=dpi)

# %%


analysis_vars = [
    "Household income 50th percentile",
    "Household income 80th percentile",
    "Household low income (%)",
    "Household medium income (%)",
    "Household high income (%)",
    "Households w car (%)",
    "Households 1 car (%)",
    "Households 2 cars (%)",
    "Households no car (%)",
    "Population density",
]

pop_gdf["Household low income (%)"] = (
    pop_gdf["Income under 150k (%)"]
    + pop_gdf["Income 150-200k (%)"]
    + pop_gdf["Income 200-300k (%)"]
)
pop_gdf["Household medium income (%)"] = (
    pop_gdf["Income 300-400k (%)"] + pop_gdf["Income 400-500k (%)"]
)
pop_gdf["Household high income (%)"] = (
    pop_gdf["Income 500-750k (%)"] + pop_gdf["Income 750k+ (%)"]
)

variable_combos = list(itertools.combinations(analysis_vars, 2))

normalize_data(pop_gdf, analysis_vars)

### bounds defining upper boundaries of color classes - assumes data normalized to [0,1]
class_bounds = [0.25, 0.50, 0.75, 1]

### get corner colors from https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/
c00 = hex_to_color("#e8e8e8")
c10 = hex_to_color("#be64ac")
c01 = hex_to_color("#5ac8c8")
c11 = hex_to_color("#3b4994")

colorlist = create_color_grid(class_bounds, c00, c10, c01, c11)
### convert back to hex color
colorlist = [rgb2hex([c.r, c.g, c.b]) for c in colorlist]

# %%
for v in variable_combos:

    v0 = v[0] + "_norm"
    v1 = v[1] + "_norm"

    make_bivariate_choropleth_map(
        pop_gdf,
        v0,
        v1,
        v[0],
        v[1],
        class_bounds,
        colorlist,
        fp=f"../illustrations/socio_bivariate_{v[0]}_{v[1]}.png",
        fs_labels=14,
        fs_tick=12,
    )

# %%


figsize_map = (15, 10)
dpi = 300
fontsize_legend = 12

filepath = "../illustrations/socio_income_50th_percentile_new"

fig, ax = plt.subplots(figsize=figsize_map)

pop_gdf.plot(
    ax=ax,
    scheme="quantiles",  # "quantiles",  # "natural_breaks",
    k=7,
    column="households_income_50_percentile",
    cmap="viridis",
    linewidth=0.0,
    edgecolor="none",
    legend=True,
    legend_kwds={
        "fmt": "{:,.0f}",
        "frameon": False,
        "fontsize": fontsize_legend,
    },
)

# # Access the legend
# legend = ax.get_legend()

# new_labels = []
# for text in legend.get_texts():
#     label = text.get_text()  # Extract the label text

#     label_split = label.split(",")

#     first_val = label_split[0]
#     second_val = label_split[1].strip(" ")

#     new_labels.append(
#         f"{int(round(float(first_val), -1))}"
#         + ", "
#         + f"{int(round(float(second_val), -1))}"
#     )

# Update the legend text
for text, new_label in zip(legend.get_texts(), new_labels):
    text.set_text(new_label)

ax.set_axis_off()

ax.add_artist(
    ScaleBar(
        dx=1,
        units="m",
        dimension="si-length",
        length_fraction=0.15,
        width_fraction=0.002,
        location="lower left",
        box_alpha=0,
        font_properties={"size": fontsize_legend},
    )
)
cx.add_attribution(
    ax=ax,
    text="(C) " + "Statistics Denmark",
    font_size=fontsize_legend,
)
txt = ax.texts[-1]
txt.set_position([0.99, 0.01])
txt.set_ha("right")
txt.set_va("bottom")

plt.tight_layout()

fig.savefig(filepath + ".png", dpi=dpi)

# %%

# %%

# PROCESS RESULTS FROM SERVICE ACCESS ANALYSIS

import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path
import os
import sys
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.helper_functions import (
    highlight_max_traveltime,
    highlight_min_traveltime,
    plot_histogram,
    map_results_user_defined,
    create_hex_grid,
    compute_weighted_time,
    export_gdf_to_duckdb_spatial,
    combine_columns_from_tables,
    load_gdf_from_duckdb,
    safe_wkb_load,
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
    max_wait_time = config_analysis.get("max_wait_time", None)
    max_duration = config_analysis.get("max_duration", None)


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

        if max_wait_time is not None:
            gdf.loc[
                gdf["wait_time_dest_min"] >= max_wait_time,
                ["duration_min", "wait_time_dest_min", "total_time_min"],
            ] = None

        if max_duration is not None:
            gdf.loc[
                gdf["duration_min"] >= max_duration,
                ["duration_min", "wait_time_dest_min", "total_time_min"],
            ] = None

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
# Aggregate travel time geometries to hex grid and municipalities

geoms = duck_db_con.execute(
    "SELECT source_id, ST_AsWKB(geometry) AS geom_wkb FROM dentist_1"
).fetchdf()

geoms["geometry"] = geoms["geom_wkb"].apply(safe_wkb_load)

geoms.drop(columns=["geom_wkb"], inplace=True)
geoms_gdf = gpd.GeoDataFrame(geoms, geometry="geometry", crs=crs)


study_area = gpd.read_file(config_analysis["study_area_fp"])

hex_grid = create_hex_grid(study_area, 7, crs, 300)


municipalities = gpd.read_parquet(config_analysis["municipalities_fp"])

muni_id_col = config_model["study_area_config"]["municipalities"]["id_column"]
municipalities = municipalities[["geometry", muni_id_col]]

# %%

joined_hex = geoms_gdf.sjoin(
    hex_grid,
    how="left",
    predicate="intersects",
)

joined_hex2 = joined_hex[joined_hex.grid_id.isna()][
    ["source_id", "geometry"]
].sjoin_nearest(hex_grid, how="left", max_distance=300, distance_col="dist_to_hex")


joined_hex2.drop(columns=["dist_to_hex"], inplace=True)
joined_hex = pd.concat([joined_hex[~joined_hex.grid_id.isna()], joined_hex2])
joined_hex.drop(columns=["index_right"], inplace=True)


assert len(joined_hex) == len(geoms_gdf)
assert joined_hex["grid_id"].isna().sum() == 0
assert len(joined_hex["source_id"].unique()) == len(geoms_gdf)

# %%
joined_muni = geoms_gdf.sjoin(municipalities, how="left", predicate="intersects")

joined_muni.drop_duplicates(subset=["source_id"], inplace=True)

joined_muni2 = joined_muni[joined_muni[muni_id_col].isna()][
    ["source_id", "geometry"]
].sjoin_nearest(
    municipalities, how="left", max_distance=300, distance_col="dist_to_muni"
)

joined_muni2.drop(columns=["dist_to_muni"], inplace=True)

joined_muni = pd.concat([joined_muni[~joined_muni[muni_id_col].isna()], joined_muni2])

joined_muni.drop(columns=["index_right"], inplace=True)

assert len(joined_muni) == len(geoms_gdf)
assert joined_muni[muni_id_col].isna().sum() == 0
assert len(joined_muni["source_id"].unique()) == len(geoms_gdf)

# %%
# combine hex and muni gdfs
joined_hex_muni = joined_hex.merge(
    joined_muni[[muni_id_col, "source_id"]], on="source_id", how="left"
)

assert len(joined_hex_muni) == len(geoms_gdf)
assert joined_hex_muni[muni_id_col].isna().sum() == 0
assert joined_hex_muni["grid_id"].isna().sum() == 0
assert len(joined_hex_muni["source_id"].unique()) == len(geoms_gdf)

# %%
# export to duckdb
export_gdf_to_duckdb_spatial(joined_hex_muni, duck_db_con, "source_hex_muni")

tables_df = duck_db_con.sql("SELECT table_name FROM duckdb_tables;").df()
assert "source_hex_muni" in tables_df["table_name"].values
# %%
export_gdf_to_duckdb_spatial(hex_grid, duck_db_con, "hex_grid")
export_gdf_to_duckdb_spatial(municipalities, duck_db_con, "municipalities")
# %%
# Compute weighted travel times based on service importance

weight_cols = ["duration_min", "total_time_min"]
labels = ["Travel Time", "Total Time (incl. wait time)"]

for i, w in enumerate(weight_cols):

    weighted_travel_times = compute_weighted_time(
        services, 1, data_path / "input", service_weights, w
    )

    weighted_travel_times.to_parquet(
        results_path / f"data/weighted_{w}_otp_geo.parquet",
        index=False,
        engine="pyarrow",
    )

    export_gdf_to_duckdb_spatial(weighted_travel_times, duck_db_con, f"weighted_{w}")

    plot_histogram(
        weighted_travel_times,
        "total_weighted_time",
        f"Histogram of {labels[i]}",
        f"Weighted {labels[i]} (min)",
        "Frequency",
        results_path / f"plots/weighted_histogram_{labels[i].replace(" ", "_")}.png",
    )

    # # Map of weighted travel times
    map_results_user_defined(
        weighted_travel_times,
        "total_weighted_time",
        "Total vægtet rejsetid (minutter)",
        results_path / f"maps/weighted_map_{labels[i].replace(" ", "_")}.png",
        bins=[1000, 2000, 3000, 4000, 5000],
    )


# %%
# aggregate to hex grid
for w in weight_cols:

    query = f"""CREATE OR REPLACE TABLE hex_weighted_{w} AS (SELECT 
            h.grid_id,
            hg.geometry,
            AVG(w.total_weighted_time) AS avg_total_weighted_time
        FROM weighted_{w} w
        JOIN source_hex_muni h 
            ON w.source_id = h.source_id
        JOIN hex_grid hg
            ON h.grid_id = hg.grid_id
        GROUP BY h.grid_id, hg.geometry);
        """
    duck_db_con.execute(query)

# plot maps of weighted travel times on hex grid
fontsize = 12
for w in weight_cols:
    weighted_hex = load_gdf_from_duckdb(duck_db_con, f"hex_weighted_{w}", crs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="3.5%", pad="1%")
    cax.tick_params(labelsize=fontsize)

    weighted_hex.plot(
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
# compute and plot total travel times
# NOTE: Services are hardcoded here - could be improved!
sum_query = """
DROP TABLE IF EXISTS  total_travel_times;
CREATE TABLE total_travel_times AS
SELECT 
    source_id, 
    SUM(duration_min) AS total_duration,
    SUM(wait_time_dest_min) AS total_wait_time
FROM (
    SELECT source_id, duration_min, wait_time_dest_min FROM dentist_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM doctor_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM kindergarten_nursery_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM library_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM pharmacy_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM school_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM sports_facility_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM supermarket_1
    UNION ALL
    SELECT source_id, duration_min, wait_time_dest_min FROM train_station_1
) AS all_data
GROUP BY source_id;
ALTER TABLE total_travel_times ADD COLUMN total_time_min DOUBLE;
UPDATE total_travel_times SET total_time_min = total_duration + total_wait_time;
"""
duck_db_con.execute(sum_query)


total_travel_times_df = duck_db_con.execute(
    """
SELECT tt.source_id, tt.total_duration, tt.total_wait_time, tt.total_time_min, ST_AsWKB(geometry) AS geom_wkb
FROM total_travel_times tt JOIN dentist_1 s ON tt.source_id = s.source_id
"""
).fetchdf()

total_travel_times_df["geometry"] = total_travel_times_df["geom_wkb"].apply(
    safe_wkb_load
)

# Drop the WKB helper column
total_travel_times_df = total_travel_times_df.drop(columns=["geom_wkb"])

total_travel_times_gdf = gpd.GeoDataFrame(
    total_travel_times_df, geometry="geometry", crs=crs
)


# plot histogram of total travel times
plot_histogram(
    total_travel_times_gdf,
    "total_time_min",
    "Histogram of Total Travel Times to All Services",
    "Total Travel Time to All Services (min)",
    "Frequency",
    results_path / "plots/total_travel_time_histogram.png",
)

# map of total travel times
map_results_user_defined(
    total_travel_times_gdf,
    "total_time_min",
    "Total rejsetid til alle servicefunktioner (minutter)",
    results_path / "maps/total_travel_time_map.png",
    bins=[
        200,
        400,
        600,
        800,
        1000,
        1200,
        1400,
        1600,
    ],
)

# to duckdb
export_gdf_to_duckdb_spatial(total_travel_times_gdf, duck_db_con, "total_travel_times")

# TODO: AGGREGATE TO HEX GRID
# PLOT MAPS
# %%
query = f"""CREATE OR REPLACE TABLE hex_total_travel_times AS (SELECT 
            h.grid_id,
            hg.geometry,
            AVG(t.total_duration) AS avg_total_duration,
            AVG(t.total_wait_time) AS avg_total_wait_time,
            AVG(t.total_time_min) AS avg_total_time
        FROM total_travel_times t
        JOIN source_hex_muni h 
            ON t.source_id = h.source_id
        JOIN hex_grid hg
            ON h.grid_id = hg.grid_id
        GROUP BY h.grid_id, hg.geometry);
        """
duck_db_con.execute(query)

# %%
# plot maps of weighted travel times on hex grid
fontsize = 12

hex_tt = load_gdf_from_duckdb(duck_db_con, f"hex_total_travel_times", crs)

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

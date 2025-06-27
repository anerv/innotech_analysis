# %%

# PROCESS RESULTS FROM SERVICE ACCESS ANALYSIS

import pandas as pd
import geopandas as gpd
import yaml
from pathlib import Path
import os
import sys
import duckdb
from shapely import wkb

from src.helper_functions import (
    highlight_max_traveltime,
    highlight_min_traveltime,
    create_hex_grid,
    compute_weighted_time,
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


with open(config_model_path, "r") as file:
    config_model = yaml.safe_load(file)


walkspeed_min = config_model["walk_speed"] * 60  # convert to minutes

# Load results
services = config_model["services"]

otp_db_fp = data_path / "otp_results.db"

# DubckDB connection
# con = duckdb.connect()
# %%
# Open a persistent DuckDB database file Data is stored in Duckdb and then exported to parquet
duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con.execute("INSTALL spatial;")
duck_db_con.execute("LOAD spatial;")

# %%


def safe_wkb_load(val):

    if isinstance(val, (bytes, memoryview, bytearray)):
        return wkb.loads(bytes(val))  # Convert to raw bytes
    return None


def load_gdf_from_duckdb(
    con: duckdb.DuckDBPyConnection, table_name: str, crs: str, limit: int
) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from a DuckDB table with spatial support.
    Converts WKB to Shapely geometries.
    """
    query = f'SELECT *, ST_AsWKB(geometry) AS geom_wkb FROM "{table_name}"'
    if limit:
        query += f" LIMIT {limit}"
    df = con.execute(query).fetchdf()

    df["geometry"] = df["geom_wkb"].apply(safe_wkb_load)

    # Drop the WKB helper column
    df = df.drop(columns=["geom_wkb"])

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    return gdf


def export_gdf_to_duckdb_spatial(
    gdf: gpd.GeoDataFrame, con: duckdb.DuckDBPyConnection, table_name: str
):
    """
    Export a GeoDataFrame to DuckDB using the spatial extension.
    Converts geometry to WKB and avoids GeoPandas warnings.
    """

    if gdf.empty:
        raise ValueError("GeoDataFrame is empty.")

    # Load DuckDB spatial extension
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    # Convert geometry to WKB BEFORE assigning back to a DataFrame
    geometry_wkb = gdf.geometry.apply(lambda geom: geom.wkb if geom else None)
    df = gdf.drop(
        columns=gdf.geometry.name
    ).copy()  # drop geometry column before assignment
    df["geometry"] = geometry_wkb  # now assign WKB as raw bytes

    # Register with DuckDB and create spatial table
    con.register("temp_gdf", df)

    column_names = [f'"{col}"' for col in df.columns if col != "geometry"]
    select_clause = ", ".join(column_names + ["ST_GeomFromWKB(geometry) AS geometry"])

    con.execute(
        f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT {select_clause} FROM temp_gdf;'
    )
    con.unregister("temp_gdf")

    print(f"GeoDataFrame with {len(df)} rows exported to DuckDB table '{table_name}'.")


def combine_columns_from_tables(
    column_names: list[str],
    conn: duckdb.DuckDBPyConnection,
    table_names: list[str],
    common_id_column: str,
    geometry_column: str = "geometry",
    crs: str = "EPSG:25832",
):
    """
    Combines specified columns from multiple tables into a single query result,
    including the geometry column from the first table, properly converted to shapely geometries.

    Parameters:
    - column_names (list of str): The names of the columns to combine from each table.
    - geometry_column (str): The name of the geometry column to include.
    - conn: DuckDB connection object.
    - table_names (list of str): List of table names to join.
    - common_id_column (str): The common ID column to join tables on.
    - crs (optional): CRS to assign to the output GeoDataFrame.
    """

    # Start building the SQL query
    select_clause = f"t1.{common_id_column}, ST_AsWKB(t1.{geometry_column}) AS geom_wkb"
    join_clause = ""

    # Add each table's columns to SELECT and JOIN clauses
    for i, table in enumerate(table_names):
        alias = f"t{i+1}"
        for column_name in column_names:  # Iterate over each column name
            select_clause += f", {alias}.{column_name} AS {column_name}_{table}"
        if i > 0:
            join_clause += f' LEFT JOIN "{table}" {alias} ON t1.{common_id_column} = {alias}.{common_id_column}'

    # Construct the full query
    query = f"""
    SELECT {select_clause}
    FROM \"{table_names[0]}\" t1
    {join_clause}
    """

    try:
        df = conn.execute(query).fetchdf()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    df[geometry_column] = df["geom_wkb"].apply(safe_wkb_load)
    df = df.drop(columns=["geom_wkb"])

    gdf = gpd.GeoDataFrame(df, geometry=geometry_column, crs=crs)

    return gdf


# %%
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

        # TODO: Export to duckdb
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


# TODO: move to config file
weight_dictionary = {
    "doctor": 0.05,
    "dentist": 0.05,
    "pharmacy": 0.05,
    "kindergarten-nursery": 5,
    "school": 5,
    "supermarket": 5,
    "library": 1,
    "train_station": 5,
    "sports_facility": 1,
}

weight_cols = ["duration_min", "total_time_min"]

for w in weight_cols:

    weighted_travel_times = compute_weighted_time(
        services, 1, data_path / "input", weight_dictionary, w
    )

    weighted_travel_times.to_parquet(
        results_path / f"data/weighted_{w}_otp_geo.parquet",
        index=False,
        engine="pyarrow",
    )


# %%

# Combine results from all services into a single GeoDataFrame

column_names = ["duration_min", "wait_time_dest_min", "total_time_min"]
table_names = [s["service_type"] + "_1" for s in services]


combined_gdf = combine_columns_from_tables(
    column_names,
    conn=duck_db_con,
    table_names=table_names,
    common_id_column="source_id",
)

# sum all columns that start with total_time_min

# %%
total_col = "travel_time_total_min"
sum_cols = [col for col in combined_gdf.columns if col.startswith("total_time_min")]
combined_gdf[total_col] = combined_gdf[sum_cols].sum(axis=1)

# %%

# compute average travel time per hex bin

aggregation_type = "ave_travel_times"

study_area = gpd.read_file(config_analysis["study_area_fp"])

hex_grid = create_hex_grid(study_area, 6, crs, 200)

hex_travel_times = gpd.sjoin(
    hex_grid,
    combined_gdf,
    how="inner",
    predicate="intersects",
    rsuffix="travel",
    lsuffix="hex",
)

hex_id_col = "grid_id"

cols_to_average = [
    col for col in hex_travel_times.columns if col.startswith("total_time_min")
]
cols_to_average.extend([total_col])


hex_avg_travel_times = (
    hex_travel_times.groupby(hex_id_col)[cols_to_average].mean().reset_index()
)

hex_avg_travel_times_gdf = hex_grid.merge(
    hex_avg_travel_times, on="grid_id", how="left"
)

hex_avg_travel_times_gdf.to_parquet(
    results_path / f"data/hex_{aggregation_type}_otp.parquet",
)


# %%

municipalities = gpd.read_parquet(config_analysis["municipalities_fp"])

id_column = config_model["study_area_config"]["municipalities"]["id_column"]
municipalities = municipalities[["geometry", id_column]]

regional_travel_times = gpd.sjoin(
    municipalities,
    combined_gdf,
    how="inner",
    predicate="intersects",
    rsuffix="travel",
    lsuffix="region",
)


cols_to_average = [
    col for col in regional_travel_times.columns if col.endswith("_total_time_min")
]
cols_to_average.extend([total_col])

municipal_avg_travel_times = (
    regional_travel_times.groupby(id_column)[cols_to_average].mean().reset_index()
)

municipal_avg_travel_times_gdf = municipalities.merge(
    municipal_avg_travel_times, on=id_column, how="left"
)

municipal_avg_travel_times_gdf.to_parquet(
    results_path / f"data/municipal_{aggregation_type}_otp.parquet",
)


# %%

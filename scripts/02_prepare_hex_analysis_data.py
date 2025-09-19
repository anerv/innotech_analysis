# %%

import numpy as np
import pandas as pd
import geopandas as gpd
import duckdb
import yaml
from pathlib import Path

from src.helper_functions import (
    safe_wkb_load,
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


otp_db_fp = data_path / "otp_results.db"
duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con = duckdb.connect(otp_db_fp)

duck_db_con.execute("INSTALL spatial;")
duck_db_con.execute("LOAD spatial;")

services = config_model["services"]

# %%
####### GET DATA ##########

table_names = [s["service_type"] + "_1" for s in services]

random_table = duck_db_con.execute(
    f"""
SELECT * FROM {table_names[0]} LIMIT 1
"""
).fetchdf()

id_col = "source_id"
geometry_col = "geometry"
columns = [
    "duration_min",
    "wait_time_dest_min",
    "total_time_min",
    "walkDistance",
    "abs_dist",
    "transfers",
]

columns.extend([c for c in random_table.columns if "_duration" in c])

first_col = columns[0]

# %%
output_table = "hex_travel_times"
geom_id_col = "grid_id"
geometry_table_name = "hex_grid"

# Build CTEs for travel time tables (join with source_hex to get geom_id_col)
ctes = []
for table in table_names:
    avg_expressions = [f"AVG({col}) AS {col}_{table}_ave" for col in columns]
    null_count_expr = f"SUM(CASE WHEN {first_col} IS NULL THEN 1 ELSE 0 END) AS no_connection_count_{table}"
    sum_observations_expr = f"COUNT(*) AS total_observations_{table}"

    cte = f"""{table}_agg AS (
        SELECT sh.{geom_id_col},
               {", ".join(avg_expressions)},
               {null_count_expr},
               {sum_observations_expr}
        FROM {table} t
        JOIN source_hex_muni sh ON t.{id_col} = sh.{id_col}
        GROUP BY sh.{geom_id_col}
    )"""
    ctes.append(cte)

# Geometry CTE from geometry table
geom_cte = f"""geom AS (
    SELECT {geom_id_col}, ST_AsWKB({geometry_col}) AS geom_wkb
    FROM {geometry_table_name}
)"""
ctes.insert(0, geom_cte)

# Build SELECT clause
select_cols = [
    f"geom.{geom_id_col}",
    f"geom.geom_wkb",
]
for table in table_names:
    for col in columns:
        select_cols.append(f"{table}_agg.{col}_{table}_ave")
    select_cols.append(f"{table}_agg.no_connection_count_{table}")
    select_cols.append(f"{table}_agg.total_observations_{table}")

# Final query with CREATE TABLE
query = f"""
CREATE OR REPLACE TABLE {output_table} AS
WITH
{",\n".join(ctes)}
SELECT
    {", ".join(select_cols)}
FROM geom
{" ".join([f"LEFT JOIN {table}_agg USING ({geom_id_col})" for table in table_names])};
"""

duck_db_con.execute(query)


# %%
# hex_travel_times = duck_db_con.execute(f"SELECT * FROM {output_table}").fetchdf()

# hex_travel_times["geometry"] = hex_travel_times["geom_wkb"].apply(safe_wkb_load)

# # Drop the WKB helper column
# hex_travel_times = hex_travel_times.drop(columns=["geom_wkb"])

# hex_travel_times_gdf = gpd.GeoDataFrame(hex_travel_times, geometry="geometry", crs=crs)

# hex_travel_times_gdf.to_parquet("hex_travel_times.parquet")

# %%

# REPEAT FOR MUNICIPALITIES

output_table = "municipal_travel_times"
geom_id_col = config_model["study_area_config"]["municipalities"]["id_column"]
geometry_table_name = "municipalities"

# Build CTEs for travel time tables (join with source_hex to get geom_id_col)
ctes = []
for table in table_names:
    avg_expressions = [f"AVG({col}) AS {col}_{table}_ave" for col in columns]
    null_count_expr = f"SUM(CASE WHEN {first_col} IS NULL THEN 1 ELSE 0 END) AS no_connection_count_{table}"
    sum_observations_expr = f"COUNT(*) AS total_observations_{table}"

    cte = f"""{table}_agg AS (
        SELECT sh.{geom_id_col},
               {", ".join(avg_expressions)},
               {null_count_expr},
               {sum_observations_expr}
        FROM {table} t
        JOIN source_hex_muni sh ON t.{id_col} = sh.{id_col}
        GROUP BY sh.{geom_id_col}
    )"""
    ctes.append(cte)

# Geometry CTE from geometry table
geom_cte = f"""geom AS (
    SELECT {geom_id_col}, ST_AsWKB({geometry_col}) AS geom_wkb
    FROM {geometry_table_name}
)"""
ctes.insert(0, geom_cte)

# Build SELECT clause
select_cols = [
    f"geom.{geom_id_col}",
    f"geom.geom_wkb",
]
for table in table_names:
    for col in columns:
        select_cols.append(f"{table}_agg.{col}_{table}_ave")
    select_cols.append(f"{table}_agg.no_connection_count_{table}")
    select_cols.append(f"{table}_agg.total_observations_{table}")

# Final query with CREATE TABLE
query = f"""
CREATE OR REPLACE TABLE {output_table} AS
WITH
{",\n".join(ctes)}
SELECT
    {", ".join(select_cols)}
FROM geom
{" ".join([f"LEFT JOIN {table}_agg USING ({geom_id_col})" for table in table_names])};
"""

duck_db_con.execute(query)


# %%

# muni_travel_times = duck_db_con.execute(f"SELECT * FROM {output_table}").fetchdf()

# muni_travel_times["geometry"] = muni_travel_times["geom_wkb"].apply(safe_wkb_load)

# # Drop the WKB helper column
# muni_travel_times = muni_travel_times.drop(columns=["geom_wkb"])

# muni_travel_times_gdf = gpd.GeoDataFrame(
#     muni_travel_times, geometry="geometry", crs=crs
# )

# muni_travel_times_gdf.to_parquet("muni_travel_times.parquet")


# %%

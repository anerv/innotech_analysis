import geopandas as gpd
import pandas as pd
import overpy
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar
import h3
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np
from shapely.ops import transform
import json
import os
import requests
import numpy as np
import duckdb
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

######################## OTP FUNCTIONS ########################


def convert_otp_time(millis, tz="Europe/Copenhagen"):
    if isinstance(millis, (int, float)) and millis > 0:
        try:
            return datetime.fromtimestamp(millis / 1000, tz=ZoneInfo(tz)).strftime(
                "%Y-%m-%d %H:%M"
            )
        except Exception as e:
            print(f"Failed to convert timestamp {millis}: {e}")
            return None
    return None


def get_travel_info(
    from_lat,
    from_lon,
    to_lat,
    to_lon,
    date,
    time,
    url,
    search_window,
    walk_speed=1.3,
    arrive_by="true",
):

    query = f"""
    {{
    plan(
        from: {{lat: {from_lat}, lon: {from_lon}}}
        to: {{lat: {to_lat}, lon: {to_lon}}}
        date: "{date}"
        time: "{time}"
        walkSpeed: {walk_speed}
        arriveBy: {arrive_by},
        searchWindow: {search_window}
        numItineraries: 1
    ) {{
        itineraries {{
        startTime
        waitingTime
        duration
        walkDistance
        legs {{
            mode
            duration
        }}
        }}
    }}
    }}
    """

    # print(f"Sending request to OTP API with query: {query}")
    response = requests.post(url, json={"query": query})
    # print(f"Error: {response.status_code} - {response.text}")

    # print(response.json())

    return response.json()


# Process the individual service types
def process_adresses(
    dataset,
    sampelsize,
    time,
    date,
    walk_speed,
    search_window,
    url,
    data_path,
    otp_con,
    con,
    chunk_size=1000,
    max_workers=16,
):
    filename = dataset + ".parquet"
    data = data_path / filename
    dataset = dataset.replace("-", "_")

    # Create target table

    otp_con.execute(f"DROP TABLE IF EXISTS {dataset};")
    otp_con.execute(
        f"""
        CREATE TABLE {dataset} (
            source_id TEXT,
            target_id TEXT,
            from_lat DOUBLE,
            from_lon DOUBLE,
            startTime TEXT,
            waitingTime DOUBLE,
            duration DOUBLE,
            walkDistance DOUBLE,
            abs_dist DOUBLE,
            mode_durations_json TEXT,
        )
    """
    )

    # Load data into a temporary table
    if sampelsize == 0:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE data_pairs AS 
            SELECT * 
            FROM '{data}'
        """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE data_pairs AS 
            SELECT * 
            FROM '{data}'
            USING SAMPLE {sampelsize} ROWS
        """
        )

    # Function to process a single row
    def process_row(
        row,
        date,
        time,
        search_window=search_window,
        walk_speed=walk_speed,
        url=url,
        arrive_by="true",
    ):

        try:
            travel_info = get_travel_info(
                row.source_lat,
                row.source_lon,
                row.dest_lat,
                row.dest_lon,
                date,
                time,
                url=url,
                search_window=search_window,
                walk_speed=walk_speed,
                arrive_by=arrive_by,
            )
            itinerary = travel_info["data"]["plan"]["itineraries"][0]

            # Duration per mode
            mode_durations = {}
            for leg in itinerary["legs"]:
                mode = leg["mode"]
                duration = leg["duration"]
                mode_durations[mode] = mode_durations.get(mode, 0) + duration

            mode_durations_json = json.dumps(mode_durations)

            return (
                row.source_address_id,
                row.dest_address_id,
                row.source_lat,
                row.source_lon,
                convert_otp_time(itinerary["startTime"]),
                itinerary["waitingTime"],
                itinerary["duration"],
                itinerary["walkDistance"],
                row.dest_distance,
                mode_durations_json,
            )
        except Exception:
            return (
                row.source_address_id,
                row.dest_address_id,
                row.source_lat,
                row.source_lon,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                row.dest_distance,
                json.dumps({}),  # Empty dict as JSON
            )

    # Process in chunks with parallel execution
    offset = 0
    while True:
        chunk = con.execute(
            f"""
            SELECT * FROM data_pairs 
            LIMIT {chunk_size} OFFSET {offset}
        """
        ).fetchdf()

        if chunk.empty:
            break

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_row, row, date, time, search_window)
                for row in chunk.itertuples(index=False)
            ]

            for future in as_completed(futures):
                result = future.result()
                otp_con.execute(
                    f"""
                    INSERT INTO {dataset} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,)
                """,
                    result,
                )

        offset += chunk_size


############################ RANDOM HELPER FUNCTIONS ############################


def combine_results(
    services,
    path,
    travel_time_columns,
    id_column="source_id",
    n_neighbors=1,
):

    all_travel_times = []

    for service in services:

        dataset = f"{service['service_type']}_{n_neighbors}"
        fp = path / f"data/{dataset}_otp_geo.parquet"
        if not fp.exists():
            print(f"File {fp} does not exist. Skipping.")
            continue
        df = pd.read_parquet(fp)

        df = df[[id_column] + travel_time_columns]

        rename_dict = {
            col: f"{dataset}_{col}" for col in travel_time_columns if col in df.columns
        }

        df.rename(columns=rename_dict, inplace=True)

        all_travel_times.append(df)

    all_travel_times_df = reduce(
        lambda left, right: pd.merge(left, right, on="source_id", how="outer"),
        all_travel_times,
    )

    # get geoms
    gdf = gpd.read_parquet(path / f"data/{dataset}_otp_geo.parquet")

    all_travel_times_gdf = pd.merge(
        gdf[["source_id", "geometry"]], all_travel_times_df, on="source_id", how="left"
    )

    assert len(all_travel_times_gdf) == len(
        all_travel_times_df
    ), "Mismatch in number of rows after merge."

    return all_travel_times_gdf


def compute_weighted_time(
    services,
    n_neighbors,
    path,
    weight_dictionary,
    travel_time_column="total_time_min",
):

    all_travel_times = []

    for service in services:

        dataset = f"{service['service_type']}_{n_neighbors}"
        fp = path / f"data/{dataset}_otp_geo.parquet"
        if not fp.exists():
            print(f"File {fp} does not exist. Skipping.")
            continue
        df = pd.read_parquet(fp)

        df = df[["source_id", travel_time_column]]

        df.rename(columns={travel_time_column: f"{dataset}_travel_time"}, inplace=True)

        df[f"{dataset}_weighted_time"] = df.apply(
            lambda row: row[f"{dataset}_travel_time"]
            * weight_dictionary.get(service["service_type"], 1),
            axis=1,
        )

        all_travel_times.append(df)

    all_travel_times_df = reduce(
        lambda left, right: pd.merge(left, right, on="source_id", how="outer"),
        all_travel_times,
    )

    gdf = gpd.read_parquet(path / f"data/{dataset}_otp_geo.parquet")
    all_travel_times_gdf = pd.merge(
        gdf[["source_id", "geometry"]], all_travel_times_df, on="source_id", how="left"
    )

    assert len(all_travel_times_gdf) == len(
        df
    ), "Mismatch in number of rows after merge."

    all_travel_times_gdf["total_weighted_time"] = all_travel_times_gdf[
        [col for col in all_travel_times_gdf.columns if col.endswith("_weighted_time")]
    ].sum(axis=1)

    return all_travel_times_gdf


def transfers_from_json(json_str):
    if pd.isna(json_str) or json_str == "":
        return pd.NA
    try:
        mode_dict = json.loads(json_str)
        non_walk_modes = [mode for mode in mode_dict.keys() if mode.upper() != "WALK"]
        n_modes = len(non_walk_modes)

        return max(0, n_modes - 1)
    except Exception:
        return pd.NA


def unpack_modes_from_json(df, json_column="mode_durations_json"):
    if json_column not in df.columns:
        raise ValueError(f"Column '{json_column}' does not exist in the DataFrame.")

    # Parse JSON strings into dictionaries (handle None or empty strings gracefully)
    def parse_json(x):
        if pd.isna(x) or x == "":
            return {}
        if isinstance(x, dict):
            return x
        try:
            return json.loads(x)
        except Exception:
            return {}

    # Apply parsing
    modes_dicts = df[json_column].apply(parse_json)

    # Get all unique mode keys across all rows
    all_modes = set()
    for d in modes_dicts:
        all_modes.update(d.keys())

    all_modes = list(all_modes)
    all_modes = [mode.lower() for mode in all_modes]  # Normalize to lowercase
    all_modes = [
        mode + "_duration" for mode in all_modes
    ]  # Append '_duration' to each mode
    all_modes = sorted(all_modes)  # Sort modes for consistent order

    # Create a DataFrame with one column per mode, filled with NaN initially
    modes_df = pd.DataFrame(index=df.index, columns=all_modes)

    # Fill the modes_df with durations from the dicts
    for mode in all_modes:
        modes_df[mode] = modes_dicts.apply(lambda d: d.get(mode, None))

    modes_df = modes_df.apply(pd.to_numeric, errors="coerce")

    modes_df = modes_df.fillna(0)  # Replace NaN with 0 for easier analysis

    # Combine original df with the expanded modes_df
    df = pd.concat([df, modes_df], axis=1)

    return df


def get_service_type(nace_code, nace_dict):
    for service_type, codes in nace_dict.items():
        if nace_code in codes:
            return service_type
    return None


def get_nace_code(service_type, nace_dict):
    for type, codes in nace_dict.items():
        if service_type == type:
            return codes[0]  # return the first code in the list
    return None


def remove_z(geometry):
    if geometry.has_z:
        return transform(lambda x, y, z=None: (x, y), geometry)
    return geometry


# Function to create a GeoDataFrame from nodes
def create_nodes_gdf(nodes):
    if not nodes:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    data = []
    for node in nodes:
        point = Point(node.lon, node.lat)
        tags = node.tags
        tags["id"] = node.id
        data.append({"geometry": point, **tags})
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# Function to create a GeoDataFrame from ways
def create_ways_gdf(ways):
    if not ways:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    data = []
    for way in ways:
        coords = [(node.lon, node.lat) for node in way.nodes]
        line = LineString(coords)
        tags = way.tags
        tags["id"] = way.id
        data.append({"geometry": line, **tags})
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# # Function to create a GeoDataFrame from relations
# def create_relations_gdf(overpass_result, relations):
#     if not relations:
#         return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

#     data = []
#     for relation in relations:
#         coords = []
#         for member in relation.members:
#             if isinstance(member, overpy.RelationWay):
#                 way = overpass_result.get_way(member.ref)
#                 coords.extend([(node.lon, node.lat) for node in way.nodes])
#             elif isinstance(member, overpy.RelationNode):
#                 node = overpass_result.get_node(member.ref)
#                 coords.append((node.lon, node.lat))
#         if len(coords) > 2:
#             poly = Polygon(coords)
#             tags = relation.tags
#             tags["id"] = relation.id
#             data.append({"geometry": poly, **tags})
#     return gpd.GeoDataFrame(data, crs="EPSG:4326")


def combine_points_within_distance(points_gdf, distance=200, inherit_columns=None):
    """
    Combines all point geometries within a specified distance into one point.

    Parameters:
    points_gdf (geopandas.GeoDataFrame): GeoDataFrame containing the point geometries.
    distance (float): The distance in the same unit as the CRS of the GeoDataFrame.
    inherit_columns (list of str): Columns whose values will be inherited from the first point within each buffer.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with combined points.
    """
    if inherit_columns is None:
        inherit_columns = []

    # Create a buffer around each point
    buffered_points = points_gdf.geometry.buffer(distance)

    # Create a list to store the combined points
    combined_points = []

    # Iterate through the buffered points
    for i, buffer in enumerate(buffered_points):
        # Find all points within the current buffer
        points_within = points_gdf[points_gdf.geometry.within(buffer)]

        # If there are points within the buffer, combine them into one point
        if not points_within.empty:
            # Calculate the centroid of the combined points
            combined_point = points_within.geometry.union_all().centroid

            # Create a new row with the combined point and inherited values
            combined_row = {
                "geometry": combined_point,
                **{column: points_within[column].iloc[0] for column in inherit_columns},
            }
            combined_points.append(combined_row)

            # Remove the combined points from the original GeoDataFrame
            points_gdf = points_gdf.drop(points_within.index)

    # Create a new GeoDataFrame with the combined points
    combined_points_gdf = gpd.GeoDataFrame(combined_points, crs=points_gdf.crs)

    return combined_points_gdf


def aggregate_points_by_distance(
    gdf,
    distance_threshold=50,
    destination_type_column="destination_type_main",
    inherit_columns=None,
):
    """
    Aggregates point geometries in a GeoDataFrame into one point if they are within a user-specified distance threshold
    and share the same value for the column "destination-type".

    Parameters:
    gdf (geopandas.GeoDataFrame): The input GeoDataFrame containing point geometries.
    distance_threshold (float): The distance threshold for aggregating points.
    destination_type_column (str): The column name for the destination type.
    inherit_columns (list of str): Columns whose values will be inherited from the first point within each buffer.

    Returns:
    geopandas.GeoDataFrame: The aggregated GeoDataFrame.
    """
    if inherit_columns is None:
        inherit_columns = []

    # Ensure the GeoDataFrame is in a projected coordinate system
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must be in a projected coordinate system.")

    aggregated_points = []

    # Group by destination-type
    for destination_type, group in gdf.groupby(destination_type_column):
        # Combine points within the distance threshold for each group
        combined_gdf = combine_points_within_distance(
            group, distance=distance_threshold, inherit_columns=inherit_columns
        )

        # Add the destination-type column to the combined points
        combined_gdf[destination_type_column] = destination_type

        # Append the combined points to the aggregated points list
        aggregated_points.append(combined_gdf)

    # Concatenate all aggregated points into a single GeoDataFrame
    aggregated_gdf = gpd.GeoDataFrame(
        pd.concat(aggregated_points, ignore_index=True), crs=gdf.crs
    )

    return aggregated_gdf


def create_hex_grid(polygon_gdf, hex_resolution, crs, buffer_dist):

    # Inspired by https://stackoverflow.com/questions/51159241/how-to-generate-shapefiles-for-h3-hexagons-in-a-particular-area

    poly_bounds = polygon_gdf.buffer(buffer_dist).to_crs("EPSG:4326").total_bounds

    latlng_poly = h3.LatLngPoly(
        [
            (poly_bounds[0], poly_bounds[1]),
            (poly_bounds[0], poly_bounds[3]),
            (poly_bounds[2], poly_bounds[3]),
            (poly_bounds[2], poly_bounds[1]),
        ]
    )

    hex_list = []
    hex_list.extend(h3.polygon_to_cells(latlng_poly, res=hex_resolution))

    # Create hexagon data frame
    hex_pd = pd.DataFrame(hex_list, columns=["hex_id"])

    # Create hexagon geometry and GeoDataFrame
    hex_pd["latlng_geometry"] = [
        h3.cells_to_h3shape([x], tight=True) for x in hex_pd["hex_id"]
    ]

    hex_pd["geometry"] = hex_pd["latlng_geometry"].apply(lambda x: Polygon(x.outer))

    grid = gpd.GeoDataFrame(hex_pd)

    grid.set_crs("4326", inplace=True).to_crs(crs, inplace=True)

    grid["grid_id"] = grid.hex_id

    grid = grid[["grid_id", "geometry"]]

    return grid


def count_destinations_in_hex_grid(gdf, hex_grid, destination_col):

    joined = gpd.sjoin(hex_grid, gdf, how="left", predicate="intersects")

    counts = (
        joined.groupby(["grid_id", destination_col])[destination_col]
        .count()
        .reset_index(name="count")
    )

    # Pivot the counts DataFrame to create a column for each destination type
    counts_pivot = counts.pivot(
        index="grid_id", columns=destination_col, values="count"
    ).fillna(0)

    # Merge the pivoted counts back into the hex grid
    hex_grid = hex_grid.merge(
        counts_pivot, left_on="grid_id", right_index=True, how="left"
    )

    # Fill NaN values with 0 for missing destination counts
    hex_grid = hex_grid.fillna(0)

    return hex_grid


def linestring_to_polygon(geom):
    # Only convert if LineString is closed
    if geom.is_ring:
        return Polygon(geom)
    else:
        return None  # or raise a warning / try to close it manually


def drop_contained_polygons(gdf, drop=True):
    """
    Find polygons that are fully contained by another polygon in the same GeoDataFrame.
    """
    contained_indices = set()

    for idx, geom in gdf.geometry.items():
        others = gdf.drop(idx)
        for other_idx, other_geom in others.geometry.items():
            if geom.within(other_geom):
                contained_indices.add(idx)
                break

    if drop:
        return gdf.drop(index=contained_indices)
    else:
        return list(contained_indices)


######################## PLOTTING FUNCTIONS ########################


def highlight_max_traveltime(s):
    """
    Highlight the maximum travel time in each row.
    """
    is_max = s == s.max()
    return ["color: red" if v else "" for v in is_max]


def highlight_min_traveltime(s):
    """
    Highlight the minimum travel time in each row.
    """
    is_min = s == s.min()
    return ["color: blue" if v else "" for v in is_min]


def plot_no_connection(
    gdf, study_area, attribution_text, font_size, title, fp=None, crs="EPSG:25832"
):

    assert study_area.crs == gdf.crs, "CRS mismatch between study area and GeoDataFrame"

    _, ax = plt.subplots(figsize=(10, 10))

    study_area.plot(
        ax=ax,
        color="none",
        edgecolor="black",
        alpha=0.5,
    )

    gdf.plot(
        ax=ax,
        legend=True,
        markersize=5,
        color="orange",
        edgecolor="orange",
        alpha=0.5,
        legend_kwds={"label": "No connection"},
    )

    ax.set_title(title, fontsize=font_size + 2, fontdict={"weight": "bold"})

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
            font_properties={"size": font_size},
        )
    )
    cx.add_attribution(ax=ax, text=attribution_text, font_size=font_size)
    txt = ax.texts[-1]
    txt.set_position([0.99, 0.01])
    txt.set_ha("right")
    txt.set_va("bottom")

    plt.tight_layout()

    if fp:

        plt.savefig(
            fp,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()


def plot_traveltime_results(
    gdf,
    plot_col,
    attribution_text,
    font_size,
    title,
    fp=None,
):
    """
    Plot the results on a map.
    """

    _, ax = plt.subplots(figsize=(10, 10))

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="3.5%", pad="1%")
    cax.tick_params(labelsize=font_size)

    gdf.plot(
        ax=ax,
        cax=cax,
        column=plot_col,
        cmap="viridis",
        legend=True,
        markersize=5,
    )

    for spine in cax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=font_size + 2, fontdict={"weight": "bold"})

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
            font_properties={"size": font_size},
        )
    )
    cx.add_attribution(ax=ax, text=attribution_text, font_size=font_size)
    txt = ax.texts[-1]
    txt.set_position([0.99, 0.01])
    txt.set_ha("right")
    txt.set_va("bottom")

    plt.tight_layout()

    if fp:

        plt.savefig(
            fp,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()


# Define the styling function for NaN values
def highlight_nan(x):
    return ["color: grey" if pd.isna(v) else "" for v in x]


def highlight_zero(x):
    return ["color: red" if v == 0 else "" for v in x]


# Define the styling function for max values in each row
def highlight_max(x):
    is_max = x == x.max()
    return ["background-color: yellow" if v else "" for v in is_max]


def replace_nan_with_dash(val):
    return "-" if pd.isna(val) else val


def highlight_next_max(row, color="lightyellow"):
    attr = f"background-color: {color}"
    sorted_values = row.sort_values(ascending=False)
    if len(sorted_values) > 1:
        second_highest_value = sorted_values.iloc[1]
        return [attr if val == second_highest_value else "" for val in row]
    return [""] * len(row)


def plot_hex_summaries(
    combined_grid,
    study_area,
    destination,
    fp,
    figsize=(20, 10),
    font_size=14,
    attribution_text="(C) OSM, CVR",
    titles=[
        "OSM",
        "CVR",
        "Difference (OSM - CVR)",
    ],
    cmaps=[
        "viridis",
        "viridis",
        "RdBu_r",
    ],
):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    axes = axes.flatten()

    suptitle = f"{destination.replace('_', ' ').title()}"

    viridis_norm = plt.Normalize(
        vmin=0,
        vmax=max(
            combined_grid[destination + "_osm"].max(),
            combined_grid[destination + "_cvr"].max(),
        ),
    )

    largest_abs_value = combined_grid[destination + "_diff"].abs().max()
    divnorm = colors.TwoSlopeNorm(
        vmin=-largest_abs_value,
        vcenter=0,
        vmax=largest_abs_value,
    )

    norms = [viridis_norm, viridis_norm, divnorm]

    for j, col in enumerate(["_osm", "_cvr", "_diff"]):

        ax = axes[j]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad="1%")
        cax.tick_params(labelsize=font_size)

        study_area.plot(ax=ax, color="white", edgecolor="black")

        grid_subset = combined_grid[
            (combined_grid[destination + "_osm"] > 0)
            | (combined_grid[destination + "_cvr"] > 0)
        ].copy()

        grid_subset[destination + "_osm"] = grid_subset[destination + "_osm"].replace(
            {0: np.nan}
        )

        grid_subset[destination + "_cvr"] = grid_subset[destination + "_cvr"].replace(
            {0: np.nan}
        )

        grid_subset.plot(
            cax=cax,
            ax=ax,
            column=destination + col,
            cmap=cmaps[j],
            norm=norms[j],
            # legend=True,
            alpha=0.5,
            # legend_kwds={
            #     "shrink": 0.9,
            #     "aspect": 30,
            # },
        )

        sm = plt.cm.ScalarMappable(
            cmap=cmaps[j],
            norm=norms[j],
        )
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.outline.set_visible(False)

        if j == 2:
            min_val = -largest_abs_value
            max_val = largest_abs_value
            cbar.set_ticks(
                [min_val, round(min_val / 2), 0, round(max_val / 2), max_val]
            )

        ax.set_axis_off()
        ax.set_title(titles[j], fontsize=font_size)

    fig.suptitle(
        suptitle,
        fontsize=font_size + 4,
        fontdict={"fontweight": "bold"},
    )

    axes[0].add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower left",
            box_alpha=0,
            font_properties={"size": font_size},
        )
    )

    cx.add_attribution(ax=axes[-1], text=attribution_text, font_size=font_size)
    txt = ax.texts[-1]
    txt.set_position([0.99, 0.01])
    txt.set_ha("right")
    txt.set_va("bottom")

    plt.tight_layout()

    plt.savefig(
        fp,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()


def count_destinations_in_municipalities(
    municipalities, muni_id_col, destinations, destination_col, csv_fp, html_fp
):

    muni_destinations = municipalities.sjoin(
        destinations, how="inner", predicate="intersects"
    )

    muni_service_counts = (
        muni_destinations.groupby([muni_id_col, destination_col])
        .size()
        .reset_index(name="count")
    )

    muni_service_pivot = muni_service_counts.pivot(
        index="navn", columns=destination_col, values="count"
    ).fillna(0)

    muni_service_pivot.loc["Total"] = muni_service_pivot.sum()

    muni_service_pivot = muni_service_pivot.astype(int)
    df_styled = (
        muni_service_pivot.style.apply(
            highlight_zero, subset=muni_service_pivot.columns, axis=1
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("font-weight", "bold")]},
            ]
        )
        .set_properties(
            **{"text-align": "right", "font-size": "12px", "width": "100px"}
        )
        .set_caption("Municipal service counts")
    )
    df_styled = df_styled.set_table_attributes(
        'style="width: 50%; border-collapse: collapse;"'
    )

    muni_service_pivot.to_csv(csv_fp, index=True)

    df_styled.to_html(
        html_fp,
    )

    return df_styled


def plot_destinations(
    data,
    study_area,
    destination_col,
    destination,
    color,
    font_size,
    fp,
    attribution_text,
    title,
    figsize=(7, 7),
    markersize=10,
):

    _, ax = plt.subplots(figsize=figsize)

    label = destination.replace("_", " ").title()

    study_area.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

    data[data[destination_col] == destination].plot(
        ax=ax, color=color, markersize=markersize, label=label, legend=True
    )

    # TODO: fix legend position so that it is aligned with scale bar and attribution text
    ax.legend(
        loc="lower left",
        fontsize=font_size,
        # title_fontsize=10,
        # title="OSM",
        markerscale=2,
        frameon=False,
        bbox_to_anchor=(-0.1, -0.01),
    )

    ax.set_title(title, fontsize=font_size + 2, fontdict={"weight": "bold"})

    ax.set_axis_off()

    ax.add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower center",
            box_alpha=0,
            font_properties={"size": font_size},
        )
    )
    cx.add_attribution(ax=ax, text=attribution_text, font_size=font_size)
    txt = ax.texts[-1]
    txt.set_position([0.99, 0.01])
    txt.set_ha("right")
    txt.set_va("bottom")

    plt.tight_layout()

    plt.savefig(
        fp,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


def plot_destinations_combined(
    data1,
    data2,
    data1_label,
    data2_label,
    study_area,
    destination_col,
    destination,
    color1,
    color2,
    font_size,
    fp,
    attribution_text,
    title,
    figsize=(7, 7),
    markersize=6,
):

    _, ax = plt.subplots(figsize=figsize)

    label1 = destination.replace("_", " ").title() + " - " + data1_label
    label2 = destination.replace("_", " ").title() + " - " + data2_label

    study_area.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

    data1[data1[destination_col] == destination].plot(
        ax=ax, color=color1, markersize=markersize, label=label1, legend=True, alpha=0.5
    )

    data2[data2[destination_col] == destination].plot(
        ax=ax, color=color2, markersize=markersize, label=label2, legend=True, alpha=0.5
    )

    # TODO: fix legend position so that it is aligned with scale bar and attribution text
    ax.legend(
        loc="lower left",
        fontsize=font_size,
        # title_fontsize=10,
        # title="OSM",
        markerscale=2,
        frameon=False,
        bbox_to_anchor=(-0.1, -0.01),
    )

    ax.set_title(title, fontsize=font_size + 2, fontdict={"weight": "bold"})

    ax.set_axis_off()

    ax.add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower center",
            box_alpha=0,
            font_properties={"size": font_size},
        )
    )
    cx.add_attribution(ax=ax, text=attribution_text, font_size=font_size)
    txt = ax.texts[-1]
    txt.set_position([0.99, 0.01])
    txt.set_ha("right")
    txt.set_va("bottom")

    plt.tight_layout()

    plt.savefig(
        fp,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()


def plot_destinations_combined_subplot(
    data1,
    data2,
    data1_label,
    data2_label,
    study_area,
    destination_col,
    color1,
    color2,
    font_size,
    fp,
    attribution_text,
    figsize=(15, 10),
    markersize=6,
):

    unique_destinations = set(data1[destination_col].unique()).union(
        data2[destination_col].unique()
    )

    unique_destinations = sorted(unique_destinations)

    _, axes = plt.subplots(
        nrows=2, ncols=math.ceil(len(unique_destinations) / 2), figsize=figsize
    )

    axes = axes.flatten()

    if len(axes) > len(unique_destinations):

        axes[-1].axis("off")

    for i, destination in enumerate(unique_destinations):

        title = f"{destination.replace('_', ' ').title()}"

        ax = axes[i]

        study_area.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

        data1[data1[destination_col] == destination].plot(
            ax=ax,
            color=color1,
            markersize=markersize,
            label=data1_label,
            legend=True,
            alpha=0.5,
        )

        data2[data2[destination_col] == destination].plot(
            ax=ax,
            color=color2,
            markersize=markersize,
            label=data2_label,
            legend=True,
            alpha=0.5,
        )

        ax.set_title(title, fontsize=font_size + 2, fontdict={"weight": "bold"})

        ax.set_axis_off()

    # TODO: fix legend position so that it is aligned with scale bar and attribution text
    middle_ax = axes[(len(axes) // 2) - 1]
    middle_ax.legend(
        loc="upper right",
        fontsize=font_size,
        # title_fontsize=10,
        # title="OSM",
        markerscale=3,
        frameon=False,
        bbox_to_anchor=(1, 1),
    )

    axes[len(axes) // 2].add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower left",
            box_alpha=0,
            font_properties={"size": font_size},
        )
    )

    cx.add_attribution(
        ax=axes[len(unique_destinations) - 1],
        text=attribution_text,
        font_size=font_size,
    )
    txt = ax.texts[-1]
    txt.set_position([0.99, 0.01])
    txt.set_ha("right")
    txt.set_va("bottom")

    plt.tight_layout()

    plt.savefig(
        fp,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

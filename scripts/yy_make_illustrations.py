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
from matplotlib.colors import rgb2hex
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from PIL import ImageColor
from generativepy.color import Color


# %%

# population density map

# Population
figsize_map = (15, 19)
dpi = 300
fontsize_legend = 12

pop_gdf = gpd.read_file("../data/processed/voting_areas.gpkg")

filepath = "../illustrations/socio_population_density"

fig, ax = plt.subplots(figsize=figsize_map)

pop_gdf.plot(
    ax=ax,
    scheme="quantiles",  # "natural_breaks",
    k=7,
    column="population_density",
    cmap="cividis",
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


def normalize_data(df, columns):
    for c in columns:
        df[c + "_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df


### function to convert hex color to rgb to Color object (generativepy package)
def hex_to_color(hexcode):
    rgb = ImageColor.getcolor(hexcode, "RGB")
    rgb = [v / 256 for v in rgb]
    rgb = Color(*rgb)
    return rgb


def create_color_grid(class_bounds, c00, c10, c01, c11):
    group_count = len(class_bounds)
    c00_to_c10 = []
    c01_to_c11 = []
    colorlist = []
    for i in range(group_count):
        c00_to_c10.append(c00.lerp(c10, 1 / (group_count - 1) * i))
        c01_to_c11.append(c01.lerp(c11, 1 / (group_count - 1) * i))
    for i in range(group_count):
        for j in range(group_count):
            colorlist.append(
                c00_to_c10[i].lerp(c01_to_c11[i], 1 / (group_count - 1) * j)
            )
    return colorlist


### function to get bivariate color given two percentiles
def get_bivariate_choropleth_color(p1, p2, class_bounds, colorlist):
    if p1 >= 0 and p2 >= 0:
        count = 0
        stop = False
        for percentile_bound_p1 in class_bounds:
            for percentile_bound_p2 in class_bounds:
                if (not stop) and (p1 <= percentile_bound_p1):
                    if (not stop) and (p2 <= percentile_bound_p2):
                        color = colorlist[count]
                        stop = True
                count += 1
    else:
        color = [0.8, 0.8, 0.8, 1]
    return color


def make_bivariate_choropleth_map(
    gdf,
    col1,
    col2,
    # attr,
    col1_label,
    col2_label,
    class_bounds,
    colorlist,
    figsize=(15, 10),
    alpha=0.8,
    fp=None,
    fs_labels=12,
    fs_tick=10,
):

    ### plot map based on bivariate choropleth
    _, ax = plt.subplots(1, 1, figsize=figsize)

    gdf["color_bivariate"] = [
        get_bivariate_choropleth_color(p1, p2, class_bounds, colorlist)
        for p1, p2 in zip(gdf[col1].values, gdf[col2].values)
    ]

    gdf.plot(
        ax=ax, color=gdf["color_bivariate"], alpha=alpha, legend=False, linewidth=0.0
    )

    ax.set_axis_off()

    ax.add_artist(
        ScaleBar(
            dx=1,
            units="m",
            dimension="si-length",
            length_fraction=0.15,
            width_fraction=0.002,
            location="lower left",
            box_alpha=0.5,
            font_properties={"size": fs_labels},
        )
    )

    ### now create inset legend
    legend_ax = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    legend_ax.set_aspect("equal", adjustable="box")
    count = 0
    xticks = [0]
    yticks = [0]
    for i, percentile_bound_p1 in enumerate(class_bounds):
        for j, percentile_bound_p2 in enumerate(class_bounds):
            percentileboxes = [Rectangle((i, j), 1, 1)]
            pc = PatchCollection(
                percentileboxes, facecolor=colorlist[count], alpha=alpha
            )
            count += 1
            legend_ax.add_collection(pc)
            if i == 0:
                yticks.append(percentile_bound_p2)
        xticks.append(percentile_bound_p1)

    _ = legend_ax.set_xlim([0, len(class_bounds)])
    _ = legend_ax.set_ylim([0, len(class_bounds)])
    _ = legend_ax.set_xticks(
        list(range(len(class_bounds) + 1)), xticks, fontsize=fs_tick
    )
    _ = legend_ax.set_yticks(
        list(range(len(class_bounds) + 1)), yticks, fontsize=fs_tick
    )
    _ = legend_ax.set_xlabel(col1_label, fontsize=fs_labels)
    _ = legend_ax.set_ylabel(col2_label, fontsize=fs_labels)

    plt.tight_layout()

    if fp:
        plt.savefig(fp, dpi=300)

    plt.show()
    plt.close()


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

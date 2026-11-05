import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from hashlib import md5
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import xarray as xr
from natsort import natsort_keygen


def get_md5_from_str(s):
    if not isinstance(s, str):
        s = str(s)
    return md5(s.encode("utf-8")).hexdigest()


def get_linecolor(s, colormap=None, hash_func=None):
    if colormap is None:
        colormap = px.colors.qualitative.D3
    if hash_func is None:

        def hash_func(s):
            return int(get_md5_from_str(s), base=16)

    return colormap[hash_func(s) % len(colormap)]


cd_plot_config = {
    "yaxes": {
        "gen_CD": {
            "title_text": "gen_CD",
            "zerolinecolor": "black",
            # "range": [-60, 60],
        },
        "g_factor": {
            "title_text": "g_factor",
            "zerolinecolor": "black",
            # "range": [-0.0008, 0.0008],
        },
        "abs_avg": {
            "title_text": "abs_avg",
            "zerolinecolor": "black",
            # "range": [-0.3, 3.3],
        },
        "ldlb_CD": {
            "title_text": "ldlb_CD",
            "zerolinecolor": "black",
            # "range": [-220, 220],
        },
    }
}


def create_cd_plot(dataset, cols=["gen_CD", "g_factor", "abs_avg", "ldlb_CD"]):
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0)

    G_FACTOR_WAVELENGTH_RANGE = (470, 510)

    # Use the new g_factor analysis module to calculate g_factor_max for all samples
    # g_factor_max_wavelength_idx = find_g_factor_max(dataset, wl_range=G_FACTOR_WAVELENGTH_RANGE, fwhm_window="right")["max_wavelength"]

    for sample_id_val in dataset.sample_id.values:
        ds_sample = dataset.sel(sample_id=sample_id_val)
        sample_name = ds_sample["sample_name"].item()
        color = get_linecolor(sample_name)
        # g_factor_max_wavelength = g_factor_max_wavelength_idx[sample_id_val]

        for row_idx, col in enumerate(cols, start=1):
            fig.add_trace(
                go.Scatter(
                    x=ds_sample["wavelength"],
                    y=ds_sample[col],
                    name=f"{sample_name}",
                    line=dict(color=color),
                    legendgroup=ds_sample["config_hash"].item(),
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )

            # fig.add_trace(
            #     go.Scatter(
            #         x=[g_factor_max_wavelength],  # Convert to list/array
            #         y=ds_sample[col]
            #         .sel(wavelength=g_factor_max_wavelength, method="nearest")
            #         .values,
            #         mode="markers",
            #         marker=dict(color=color),
            #         showlegend=False,
            #         legendgroup=ds_sample["config_hash"].item(),
            #     ),
            #     row=row_idx,
            #     col=1,
            # )

    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    fig.update_xaxes(
        title_text="wavelength",
        row=-1,
        col=1,
        gridcolor="lightgrey",
        zerolinecolor="lightgrey",
    )

    for row_idx, _cols in enumerate(cols, start=1):
        fig.update_yaxes(
            row=row_idx,
            col=1,
            **cd_plot_config["yaxes"].get(_cols, {}),
            gridcolor="lightgrey",
        )

    fig.update_layout(margin={"t": 40, "b": 60})

    return fig


def create_xdr_plot(dataset, y_mode="superimpose", y_transform=None, y_scale="linear"):
    fig = go.Figure()
    is_normalized = "normalize" in (y_transform or [])

    for i, sample_id_val in enumerate(dataset.sample_id.values):
        ds_sample = dataset.sel(sample_id=sample_id_val)
        sample_name = ds_sample["sample_name"].item()
        color = get_linecolor(sample_name)

        intensity = ds_sample["intensity"]

        if is_normalized:
            intensity = intensity / intensity.max()

        if y_mode == "stack":
            if y_scale == "log":
                intensity = intensity * 10**i
            else:
                intensity = intensity + (0.5 * i if is_normalized else 2000 * i)

        fig.add_trace(
            go.Scatter(
                x=ds_sample["2theta"],
                y=intensity,
                name=f"{sample_name}",
                line=dict(color=color),
                legendgroup=ds_sample["config_hash"].item(),
                showlegend=True,
            )
        )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        title_text="2theta (Cu)",
        gridcolor="lightgrey",
        zerolinecolor="lightgrey",
    )

    fig.update_yaxes(
        title_text="intensity",
        gridcolor="lightgrey",
    )

    fig.update_layout(margin={"t": 40, "b": 60})

    return fig


def create_param_plot(metadata):
    # Ensure g_factor_max is present for robustness, fill NaN with 0
    if "g_factor_max" not in metadata.columns:
        metadata["g_factor_max"] = 0.0
    metadata["g_factor_max"] = metadata["g_factor_max"].fillna(0.0)

    # Define parameter dimensions for parallel coordinates plot
    parameter_dimensions = [
        {
            "column": "sample_name",
            "dimension": {"label": "sample_name"},
        },
        {
            "column": "annealing_temp",
            "dimension": {"label": "annealing_temp (°C)", "range": [50, 150]},
        },
        {
            "column": "annealing_time",
            "dimension": {"label": "annealing_time (min)", "range": [0, 30]},
            "transform": lambda x: x / 60,
        },
        {
            "column": "concentration",
            "dimension": {"label": "concentration (M)", "range": [0, 2]},
        },
        {
            "column": "dispense_speed",
            "dimension": {"label": "dispense_speed (μl/s)", "range": [0, 250]},
        },
        {
            "column": "dispense_volume",
            "dimension": {"label": "dispense_volume (ul)", "range": [20, 300]},
        },
        {
            "column": "spin_speed",
            "dimension": {"label": "spin_speed (rpm)", "range": [0, 6000]},
        },
        {
            "column": "homogeneity",
            "dimension": {"label": "homogeneity"},
        },
        # {
        #     "column": "config_hash",
        #     "dimension": {"label": "config_hash"},
        #     "transform": lambda x: x[:6],
        # },
        {
            "column": "first_sample",
            "dimension": {"label": "first_sample"},
        },
        {
            "column": "g_factor_max",
            "dimension": {
                "label": "g_factor_max",
                "range": [0, 0.0015],
            },
            "transform": lambda x: np.abs(x),
        },
        {
            "column": "asymmetry",
            "dimension": {
                "label": "asymmetry",
                "range": [0, 0.05],
            },
        },
    ]

    # Apply transformations to the DataFrame directly
    df_transformed = metadata.copy()

    for param in parameter_dimensions:
        if "transform" in param:
            df_transformed[param["column"]] = df_transformed[param["column"]].apply(
                param["transform"]
            )

    config_df = df_transformed
    config_df = config_df.sort_values(by="sample_name", key=natsort_keygen())

    # Calculate the sorting metric and determine the order for config_hash
    # This does not reorder config_df, just creates a sorted list of categories
    if "homogeneity" in config_df.columns and "g_factor_max" in config_df.columns:
        config_df["sort_metric"] = config_df["g_factor_max"] * config_df["homogeneity"]
        config_df["first_sample"] = config_df.groupby("config_hash")[
            "sample_name"
        ].transform("first")

        sorted_first_samples = (
            config_df.groupby("first_sample")["sort_metric"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )

        sorted_hashes = (
            config_df.groupby("config_hash")["sort_metric"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
    else:
        sorted_hashes = config_df["config_hash"].unique().tolist()
        sorted_first_samples = config_df["first_sample"].unique().tolist()

    config_df["color"] = config_df["first_sample"].apply(
        lambda samples: get_linecolor(
            samples, hash_func=lambda x: int(x.replace("-", ""))
        )
    )

    # Build dimensions for parallel coordinates plot
    dimensions = []
    for param in parameter_dimensions:
        column = param["column"]
        dimension = param["dimension"]
        if column in ["sample_name", "config_hash", "first_sample"]:
            # Handle categorical sample_name by mapping to indices, preserving natural sort order
            if column == "sample_name":
                cat_type = pd.CategoricalDtype(
                    categories=config_df[column], ordered=True
                )
            # Handle categorical config_hash by mapping to indices, preserving order of appearance
            if column == "config_hash":
                cat_type = pd.CategoricalDtype(categories=sorted_hashes, ordered=True)
            if column == "first_sample":
                cat_type = pd.CategoricalDtype(
                    categories=sorted_first_samples, ordered=True
                )

            cat_series = config_df[column].astype(cat_type)
            codes = cat_series.cat.codes
            categories = cat_series.cat.categories

            dimension["tickvals"] = list(range(len(categories)))
            dimension["ticktext"] = list(categories)
            dimension["range"] = [0, len(categories) - 1]
            dimension["values"] = codes.tolist()
        else:
            dimension["values"] = config_df[column].tolist()
        dimensions.append(dimension)

    # Extract plot data
    plot_colors = config_df["color"].tolist()

    # Create the figure
    fig = go.Figure()

    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=list(range(len(config_df))),
                colorscale=plot_colors,
                showscale=False,
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(plot_bgcolor="white")
    fig.update_layout(margin={"t": 40, "b": 20})

    return fig


def create_pl_image_plot(dataset):
    # 1. Crop images and get homogeneity, coarsen for faster plotting
    img_height = dataset.sizes["y"]
    image_da = (
        dataset["image"]
        .sel(x=slice(600, 600 + img_height))
        .coarsen(x=3, y=3, boundary="trim")
        .mean()
        .astype(np.uint8)[..., ::-1]
    ).persist()  # Persist in memory for faster plotting
    heterogeneities = dataset.coords["homogeneity"]

    # 2. Create labels by iterating over the dimension (works for 1 or many samples)
    labels = []
    for sample_id_val in dataset.sample_id.values:
        # Select the slice from the main dataset first, then get the item.
        ds_sample = dataset.sel(sample_id=sample_id_val)
        sample_name = ds_sample["sample_name"].item()
        het_value = heterogeneities.sel(sample_id=sample_id_val).item()
        labels.append(f"{sample_name} homogeneity: {het_value:.3f}")

    dataset = dataset.assign_coords(pl_image_label=labels)

    # 3. Use the cropped image DataArray for plotting
    image_da_labeled = image_da.assign_coords(sample_id=labels)

    # 4. Call px.imshow directly with the xarray DataArray
    fig = px.imshow(
        image_da_labeled,  # Ensure it's a concrete array for plotting
        binary_string=True,
        facet_col="sample_id",
        facet_col_wrap=3,
        labels={"x": "", "y": ""},
        facet_col_spacing=0.01,
        facet_row_spacing=0.02,
    )

    for i, label in enumerate(labels):
        fig.layout.annotations[i]["text"] = label

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def create_pl_plot(dataset, wavelength_range=(360, 700)):
    fig = go.Figure()

    for i, sample_id_val in enumerate(dataset.sample_id.values):
        ds_sample = dataset.sel(
            sample_id=sample_id_val, wavelength=slice(*wavelength_range)
        )
        sample_name = ds_sample["sample_name"].item()
        color = get_linecolor(sample_name)

        mean_intensity = ds_sample["intensity"].mean(dim="position")

        fig.add_trace(
            go.Scatter(
                x=ds_sample["wavelength"],
                y=mean_intensity,
                name=f"{sample_name}",
                line=dict(color=color),
                legendgroup=ds_sample["config_hash"].item(),
                showlegend=True,
            ),
        )

    fig.update_xaxes(
        title_text="wavelength",
        gridcolor="lightgrey",
        zerolinecolor="lightgrey",
    )

    fig.update_yaxes(
        title_text="intensity",
        gridcolor="lightgrey",
    )

    fig.update_layout(
        plot_bgcolor="white",
        margin={"t": 40, "b": 60},
    )

    return fig


def create_pl_variance_plot(dataset, wavelength_range=(360, 700)):
    dataset = dataset.sel(wavelength=slice(*wavelength_range))

    mean_intensity = dataset["intensity"].mean(dim="position")

    dataset["variation"] = (
        xr.apply_ufunc(
            gaussian_filter1d,
            dataset["intensity"],
            kwargs=dict(sigma=3),
            dask="allowed",
        )
        - mean_intensity
    ) / mean_intensity

    # Zero out low intensity regions for clarity
    mask = (mean_intensity < np.percentile(mean_intensity, 10)).broadcast_like(
        dataset["variation"]
    )
    dataset["variation"] = xr.where(mask, 0, dataset["variation"])

    fig = px.imshow(
        dataset["variation"],
        color_continuous_scale="RdBu_r",
        origin="lower",
        facet_col="sample_id",
        color_continuous_midpoint=0.0,
        facet_col_wrap=3,
        labels={
            "x": "Wavelength",
            "y": "Position",
            "color": "Variation",
        },
    )
    for i, sample in enumerate(dataset["sample_name"]):
        fig.layout.annotations[i]["text"] = str(sample.values)

    return fig


def create_uv_vis_plot(dataset, wavelength_range=(360, 700)):
    fig = go.Figure()

    for i, sample_id_val in enumerate(dataset.sample_id.values):
        ds_sample = dataset.sel(
            sample_id=sample_id_val, wavelength=slice(*wavelength_range)
        )
        sample_name = ds_sample["sample_name"].item()
        color = get_linecolor(sample_name)

        mean_absorption = ds_sample["absorption"].mean(dim="position")

        fig.add_trace(
            go.Scatter(
                x=ds_sample["wavelength"],
                y=mean_absorption,
                name=f"{sample_name}",
                line=dict(color=color),
                legendgroup=ds_sample["config_hash"].item(),
                showlegend=True,
            ),
        )

    fig.update_xaxes(
        title_text="wavelength",
        gridcolor="lightgrey",
        zerolinecolor="lightgrey",
    )

    fig.update_yaxes(
        title_text="absorption",
        gridcolor="lightgrey",
    )

    fig.update_layout(
        plot_bgcolor="white",
        margin={"t": 40, "b": 60},
    )

    return fig


def create_uv_vis_variance_plot(dataset, wavelength_range=(360, 700)):
    dataset = dataset.sel(wavelength=slice(*wavelength_range))

    mean_absorption = dataset["absorption"].mean(dim="position")

    dataset["variation"] = (
        xr.apply_ufunc(
            gaussian_filter1d,
            dataset["absorption"],
            kwargs=dict(sigma=3),
            dask="allowed",
        )
        - mean_absorption
    ) / mean_absorption

    # Zero out low intensity regions for clarity
    mask = (mean_absorption < np.percentile(mean_absorption, 3)).broadcast_like(
        dataset["variation"]
    )
    dataset["variation"] = xr.where(mask, 0, dataset["variation"])

    fig = px.imshow(
        dataset["variation"],
        color_continuous_scale="RdBu_r",
        origin="lower",
        facet_col="sample_id",
        color_continuous_midpoint=0.0,
        facet_col_wrap=3,
        labels={
            "x": "Wavelength",
            "y": "Position",
            "color": "Variation",
        },
    )
    for i, sample in enumerate(dataset["sample_name"]):
        fig.layout.annotations[i]["text"] = str(sample.values)

    return fig

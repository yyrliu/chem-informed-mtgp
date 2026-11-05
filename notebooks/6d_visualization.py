# %%
import os
import sys
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gpcam import GPOptimizer
from IPython.display import display
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs.nature_mpl_rc import DOUBLE_COLUMN_WIDTH, nature_rc  # noqa: E402
from configs.paths import DATA_ROOT, GP_TRAIN_RESULT_FILE  # noqa: E402
from chem_informed_mtgp import (  # noqa: E402
    LPSProfileConfig,
    build_manual_entries_from_mappings,
    create_lps_dataset,
)
from chem_informed_mtgp.fingerprint_kernel import (  # noqa: E402
    FINGERPRINT_BITS,
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
    assemble_feature_matrix,
    build_fingerprint_metadata,
)
from chem_informed_mtgp.fingerprint_kernel_numba import (  # noqa: E402
    joint_kernel_factory_numba,
    my_mean,
    my_noise,
)
from chem_informed_mtgp.registries import FeatureBuilderSpec  # noqa: E402

########## configuration ##########
TRAIN_RESULT_PATH = GP_TRAIN_RESULT_FILE
MANUAL_ENTRIES_ROOT = DATA_ROOT
DB_PATH = "../mydb.duckdb"
FIGURE_OUTPUT_DIR = Path(SCRIPT_DIR).parent / "manuscript_assets/si/volume_plots"
FIGURE_PREFIX = ""  # prepended to every saved filename, e.g. "v2_"

# short axis labels, copied from shap_indexed_tanimoto_plot.py:49 feature_names_map "short" field
AXIS_LABELS = {
    "annealing_temp": "Temp (°C)",
    "annealing_time": "Time (s)",
    "dispense_volume": "Dvol (µL)",
    "spin_speed": "Rpm",
    "dispense_speed": "Dspd (µL/s)",
    "concentration": "Conc (M)",
}

FONT_FAMILY = "Arial"

# Resolve mpl's "small"/"medium" keyword sizes (nature_mpl_rc.py) into Plotly layout-px
# sizes, so exported PNGs look comparable to the mpl-styled manuscript figures.
mpl.rcParams.update(nature_rc)
_EXPORT_SCALE = 2.0  # must match the `scale` passed to fig.to_image() in _save_fig
_PT_TO_LAYOUT_PX = nature_rc["figure.dpi"] / 72.0 / _EXPORT_SCALE
SMALL_FONT_SIZE = FontProperties(size="small").get_size_in_points() * _PT_TO_LAYOUT_PX
MEDIUM_FONT_SIZE = FontProperties(size="medium").get_size_in_points() * _PT_TO_LAYOUT_PX
FIGURE_WIDTH = DOUBLE_COLUMN_WIDTH * nature_rc["figure.dpi"] / _EXPORT_SCALE
SCENE_CAMERA = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0.08, z=-0.15),
    eye=dict(x=-1.5, y=1.25, z=1.0),
)
# Compensates for Plotly gl3d's inconsistent rasterized tick/axis-title text sizing
# (measured up to ~1.5x the requested font size); see _apply_publication_style.
_SCENE_FONT_SCALE = 0.83
####################################


def _axis_label(param: str) -> str:
    return AXIS_LABELS.get(param, param.replace("_", " ").title())


def _apply_publication_style(fig, colorbar_title: str) -> None:
    """Style pass matching the make_2x1_volume_figure_for_a_site loop (6d_visualization.ipynb cell 25).

    Scene (3D) axes don't inherit `layout.font` the way 2D axes and annotations do —
    plotly.js defaults scene xaxis/yaxis/zaxis tickfont/title.font to a fixed
    Arial/12px/#444 regardless of the global font, so they must be set explicitly here.

    Unlike 2D annotations/colorbar text (real vector `<text>` elements sized exactly
    at the requested px value, confirmed via SVG export), scene tick/axis-title text
    is baked into a single rasterized bitmap for the whole 3D panel and Plotly's gl3d
    renderer does not size it consistently — glyph heights measured from a rendered
    PNG vary ~1.5x larger than the same nominal font size produces for 2D text, even
    with an orthographic camera. `_SCENE_FONT_SCALE` compensates for that measured
    inflation so 3D tick/axis-title text doesn't visually exceed the 2D subplot titles.
    """
    scene_font_size = SMALL_FONT_SIZE * _SCENE_FONT_SCALE
    scene_axis_style = dict(
        tickfont=dict(family=FONT_FAMILY, size=scene_font_size, color="black"),
        title=dict(font=dict(family=FONT_FAMILY, size=scene_font_size, color="black")),
    )
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=SMALL_FONT_SIZE, color="black")
    )
    fig.update_scenes(
        camera=SCENE_CAMERA,
        xaxis=scene_axis_style,
        yaxis=scene_axis_style,
        zaxis=scene_axis_style,
    )
    fig.update_traces(
        colorbar=dict(
            # thickness=20,
            len=0.85,
            # xpad=5,
            # x=1,
            tickfont=dict(size=SMALL_FONT_SIZE),
            title=dict(text=r"<br>" + colorbar_title),
        )
    )
    fig.update_annotations(font=dict(size=SMALL_FONT_SIZE))


def _save_fig(fig, name: str, scale: float = 2.0) -> None:
    """Render fig to PNG, write to FIGURE_OUTPUT_DIR, and display the static image (no interactive widget)."""
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(name).suffix:
        name += ".png"
    img_bytes = fig.to_image(format="png", scale=scale)
    (FIGURE_OUTPUT_DIR / f"{FIGURE_PREFIX}{name}").write_bytes(img_bytes)
    display(PILImage.open(BytesIO(img_bytes)))


# %%

########## parameter metadata and batch-level exclusions ##########
concentration_steps = np.array([0.25, 0.5, 0.75, 1.0, 1.25])  # num_steps = 5
annealing_time_steps = np.arange(60, 1200 + 30, 30)  # num_steps = 39
dispense_volume_steps = np.arange(25, 150 + 25, 25)  # num_steps = 6

_TRAIN_BATCHES = [
    "3",
    "4",
    "5",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
]  # noqa: E501

NOTEBOOK_CONFIG = {
    "param_config": OrderedDict(
        [
            ("annealing_temp", {"range": (40, 150), "step": 5}),  # num_steps = 24
            (
                "annealing_time",
                {"range": (60, 1800), "step": 30, "values": annealing_time_steps},
            ),
            (
                "dispense_volume",
                {"range": (25, 300), "step": 25, "values": dispense_volume_steps},
            ),
            ("spin_speed", {"range": (1000, 6000), "step": 100}),  # num_steps = 17
            ("dispense_speed", {"range": (80, 240), "step": 20}),  # num_steps = 8
            (
                "concentration",
                {"range": (0.1, 2.0), "step": 0.25, "values": concentration_steps},
            ),
        ]
    ),
    "outliers_by_sample_name": [
        "3-21",  # Wrong PbI2 annealing time
        "4-17",  # Wrong annealing time
        "4-21",  # Wrong annealing time
        "5-17",  # Dark PL image
        "5-21",  # 5-17 pair
        "7-04",  # No CD
        "11-26",
        "11-30",  # incosistent pair
        "14-17",  # CD outlier
        "14-19",  # PL outlier
        "23-01",  # Wrong annealing time
    ],
    "train_batches": _TRAIN_BATCHES,
    "default_target_column": "score",
    "a_site_to_smiles": {
        "MBAI": "C[C@@H](C1=CC=C(C=C1))N",
        "MeOMBAI": "C[C@@H](C1=CC=C(C=C1)OC)N",
        "ClMBAI": "C[C@@H](C1=CC=C(C=C1)Cl)N",
    },
    "condition_outlier_rsd_threshold": None,
}

param_config = NOTEBOOK_CONFIG["param_config"]
outliers_by_sample_name = NOTEBOOK_CONFIG["outliers_by_sample_name"]
INITIAL_TRAIN_BATCHES = NOTEBOOK_CONFIG["train_batches"]
TARGET_COLUMN = NOTEBOOK_CONFIG["default_target_column"]
A_SITE_TO_SMILES = NOTEBOOK_CONFIG["a_site_to_smiles"]

CONTINUOUS_FEATURE_COUNT = len(param_config)
CONTINUOUS_COLUMNS = list(param_config.keys())[:CONTINUOUS_FEATURE_COUNT]
#####################################################################

# %%

########## fingerprint metadata and kernel ##########
fingerprint_metadata = build_fingerprint_metadata(
    A_SITE_TO_SMILES,
    n_bits=FINGERPRINT_BITS,
    radius=2,
    use_features=False,
)

kernel_fn = joint_kernel_factory_numba(
    cont_dims=CONTINUOUS_FEATURE_COUNT,
    fingerprint_bits=1,  # indexed_tanimoto: 1 categorical index
    fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
    fingerprint_metadata=fingerprint_metadata,
    tanimoto_correlations=fingerprint_metadata.tanimoto_correlations,
    sigma_per_chemical=True,
)
#######################################################

# %%

########## manual training entries ##########
manual_entries = {
    7: [3, 4, 7, 8, 11, 12, 15, 16, 17, 19, 21, 23, 25, 29],
    11: [3, 7, 19, 23],
    12: [42, 46],
}

manual_entries_df = build_manual_entries_from_mappings(
    manual_entries=manual_entries,
    continuous_columns=CONTINUOUS_COLUMNS,
    root_dir=MANUAL_ENTRIES_ROOT,
    metric_defaults={
        "homogeneity": 0.0,
        "asymmetry": 0.0,
        "g_factor_max": 0.0,
        "asymmetry_5FWHM": 0.0,
    },
)
print(
    f"Prepared {len(manual_entries_df)} manual rows from batches {sorted(manual_entries.keys())}."
)
###############################################

# %%


def _indexed_fp_joint_builder(df: pd.DataFrame) -> np.ndarray:
    return assemble_feature_matrix(
        df=df,
        continuous_columns=CONTINUOUS_COLUMNS,
        fingerprint_column="a_site",
        fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
        fingerprint_metadata=fingerprint_metadata,
        a_site_column="a_site",
    )


# %%

########## build LPS dataset ##########
profile_config = LPSProfileConfig(
    param_config=param_config,
    a_site_to_smiles=A_SITE_TO_SMILES,
    continuous_columns=CONTINUOUS_COLUMNS,
    fingerprint_bits=FINGERPRINT_BITS,
    target_column=TARGET_COLUMN,
    db_path=DB_PATH,
)

dataset_core = create_lps_dataset(profile_config)
dataset_core.load_from_query()
if not manual_entries_df.empty:
    dataset_core.add_rows(manual_entries_df, source_tag="manual")
dataset_core.apply_inference_rules()
dataset_core.apply_nan_policies()
dataset_core.set_outliers(column={"sample_name": outliers_by_sample_name})
dataset_core.add_metric(TARGET_COLUMN, overwrite=True)

dataset_core.feature_registry.register(
    FeatureBuilderSpec(name="indexed_joint", fn=_indexed_fp_joint_builder)
)

df_all = dataset_core.get_df(normalized=False, copy=True, include_outliers=False)
print(f"Loaded rows after framework outlier filtering: {len(df_all)}")
########################################

# %%

########## normalize, build features, split train batches ##########
cols_to_scale = CONTINUOUS_COLUMNS
missing_columns = [col for col in cols_to_scale if col not in df_all.columns]
if missing_columns:
    raise ValueError(f"Missing continuous columns: {missing_columns}")

ranges = {col: tuple(param_config[col]["range"]) for col in cols_to_scale}
dataset_core.normalize(columns=cols_to_scale, fit_on="ranges", ranges=ranges)

dataset_core.build_feature("indexed_joint", from_normalized=True)
dataset_core.build_feature("target", from_normalized=False)

X_joint_all = dataset_core.get_feature("indexed_joint")
y_all = np.asarray(dataset_core.get_feature("target")).reshape(-1)

batch_series = df_all["sample_name"].astype(str).str.split("-").str[0]
train_mask = batch_series.isin(INITIAL_TRAIN_BATCHES).to_numpy()
X_joint = X_joint_all[train_mask]
y = y_all[train_mask]
df = df_all.loc[train_mask].copy()

assert len(X_joint) == len(y) == len(df), (
    "Length mismatch between X_joint, y, and training dataframe"
)
print("Training rows:", len(df))
#######################################################################

# %%

########## load trained GP ##########
train_result_npz = np.load(TRAIN_RESULT_PATH)
print("Train batches:", train_result_npz["train_batches"])
print("Log Likelihood:", train_result_npz["log_likelihood"])

fingerprint_gp = GPOptimizer(
    X_joint,
    y,
    init_hyperparameters=train_result_npz["hyperparameters"],
    kernel_function=kernel_fn,
    prior_mean_function=my_mean,
    noise_function=my_noise,
)

if not np.isclose(
    fingerprint_gp.get_hyperparameters(), train_result_npz["hyperparameters"]
).all():
    raise ValueError("Loaded hyperparameters do not match the initialized values.")

if not np.isclose(fingerprint_gp.log_likelihood(), train_result_npz["log_likelihood"]):
    raise ValueError("Loaded log likelihood does not match the initialized value.")

train_result_npz.close()
print("Loaded trained hyperparameters into `fingerprint_gp`.")
######################################

# %%

########## select volume axes (p0,p1,p2 -> x,y,z) and grid axes (p3,p4,p5 -> col,row,figure) ##########
lengthscales = fingerprint_gp.get_hyperparameters()[1 : 1 + CONTINUOUS_FEATURE_COUNT]
print("Lengthscales:", lengthscales)
print(
    "Auto shortest-lengthscale params:",
    [CONTINUOUS_COLUMNS[i] for i in np.argsort(lengthscales)[:3]],
)

# pinned to match the published figures (6d_visualization.ipynb cell 10)
shortest_lengthscale_indices = np.array([5, 0, 3])


def resolve_param_roles():
    xyz_idx = shortest_lengthscale_indices.tolist()
    grid_idx = sorted(
        (i for i in range(CONTINUOUS_FEATURE_COUNT) if i not in xyz_idx),
        key=lambda i: lengthscales[i],
    )
    xyz_params = [CONTINUOUS_COLUMNS[i] for i in xyz_idx]
    grid_params = [CONTINUOUS_COLUMNS[i] for i in grid_idx]
    return xyz_params, grid_params


XYZ_PARAMS, GRID_PARAMS = resolve_param_roles()
P3, P4, P5 = GRID_PARAMS
print(f"p0,p1,p2 (x,y,z):   {XYZ_PARAMS}")
print(f"p3,p4 (col,row):    {P3}, {P4}")
print(f"p5 (figure sweep):  {P5}")
#########################################################################################################

# %%


########## meshgrid + GP prediction helpers ##########
def get_param_range(param):
    config = param_config[param]
    if "values" in config:
        return np.min(config["values"]), np.max(config["values"])
    return config["range"]


def create_meshgrid(param_config, fixed_params, resolution=10):
    num_params = len(param_config)
    param_ranges = []
    for param_name in param_config:
        if param_name in fixed_params:
            param_ranges.append([fixed_params[param_name]])
        else:
            range_min, range_max = get_param_range(param_name)
            param_ranges.append(np.linspace(range_min, range_max, resolution))
    meshgrid = np.meshgrid(*param_ranges)
    return np.stack(meshgrid, axis=-1).reshape(-1, num_params)


def get_volume_axes(meshgrid_df, x_param, y_param, z_param):
    return (
        meshgrid_df[x_param].values,
        meshgrid_df[y_param].values,
        meshgrid_df[z_param].values,
    )


def _validate_target(target):
    if target not in {"mean", "variance"}:
        raise ValueError("target must be either 'mean' or 'variance'")


def _compute_target_values(feature_matrix, target):
    if target == "mean":
        return fingerprint_gp.posterior_mean(feature_matrix)["m(x)"]
    return fingerprint_gp.posterior_covariance(feature_matrix, variance_only=True)[
        "v(x)"
    ]


def _transform(values, target):
    vals = np.asarray(values).reshape(-1)
    if target == "mean":
        return np.log10(np.clip(vals, 2.5, None))
    return np.sqrt(vals)  # std, for a readable colorbar scale


def _colorbar_title(target):
    return "Posterior Mean" if target == "mean" else "Posterior Std. Dev."


def _exp_tick_text(ticks, target):
    if target == "mean":
        return [f"{10**t:.1f}" for t in ticks]
    return [f"{t:.1f}" for t in ticks]


def build_mesh_and_predictions(
    fixed_params, resolution=10, a_site="MBAI", target="mean"
):
    _validate_target(target)

    meshgrid = create_meshgrid(
        param_config, fixed_params=fixed_params, resolution=resolution
    )
    meshgrid_df = pd.DataFrame(meshgrid, columns=CONTINUOUS_COLUMNS)
    meshgrid_df["a_site_smiles"] = A_SITE_TO_SMILES.get(a_site)

    meshgrid_df_normalized = dataset_core.transform_with_normalizer(
        data=meshgrid_df.copy(), columns=CONTINUOUS_COLUMNS
    )
    feature_matrix = assemble_feature_matrix(
        df=meshgrid_df_normalized,
        continuous_columns=CONTINUOUS_COLUMNS,
        fingerprint_column="a_site_smiles",
        fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
        fingerprint_metadata=fingerprint_metadata,
        a_site_column="a_site",
    )
    target_values = _compute_target_values(feature_matrix, target)

    if np.isnan(target_values).any():
        raise ValueError(f"NaN values found in {target} predictions")

    return meshgrid_df, target_values


def add_volume_trace(
    fig,
    meshgrid_df,
    target_values,
    xyz_params,
    row,
    col,
    *,
    show_colorbar,
    value_min,
    value_max,
    ticks,
    tick_text,
    target,
    colorbar_title,
    surface_count=25,
):
    X, Y, Z = get_volume_axes(meshgrid_df, *xyz_params)
    plot_values = _transform(target_values, target)

    fig.add_trace(
        go.Volume(
            x=np.asarray(X).reshape(-1),
            y=np.asarray(Y).reshape(-1),
            z=np.asarray(Z).reshape(-1),
            value=plot_values,
            colorscale="rdbu_r",
            isomin=value_min,
            isomax=value_max,
            opacity=0.3,
            surface_count=surface_count,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                tickvals=ticks.tolist(),
                ticktext=tick_text,
            ),
        ),
        row=row,
        col=col,
    )

    fig.update_scenes(
        xaxis_title=_axis_label(xyz_params[0]),
        yaxis_title=_axis_label(xyz_params[1]),
        zaxis_title=_axis_label(xyz_params[2]),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=-0.25),
            eye=dict(x=-1.25, y=1.25, z=1.25),
        ),
        row=row,
        col=col,
    )


#######################################################

# %%


########## p3xp4 grid figure for one a-site, one p5 value (mirrors cell 14/15's 3x3 grid) ##########
def make_p3p4_grid_figure(
    p5_value,
    *,
    a_site="MBAI",
    resolution=20,
    target="mean",
    title_font_size=SMALL_FONT_SIZE,
    width=FIGURE_WIDTH,
    height=1000,
    global_value_min=None,
    global_value_max=None,
):
    p3_min, p3_max = get_param_range(P3)
    p4_min, p4_max = get_param_range(P4)

    base_fixed_params = {
        col: np.mean(param_config[col]["range"])
        for col in CONTINUOUS_COLUMNS
        if col not in (*XYZ_PARAMS, P3, P4, P5)
    }
    base_fixed_params[P5] = p5_value

    subplot_payloads, subplot_titles, all_plot_values = [], [], []
    for row, row_val in enumerate(np.linspace(p4_min, p4_max, 3), start=1):
        for col, col_val in enumerate(np.linspace(p3_min, p3_max, 3), start=1):
            fixed_params = {**base_fixed_params, P3: col_val, P4: row_val}
            meshgrid_df, target_values = build_mesh_and_predictions(
                fixed_params,
                resolution=resolution,
                a_site=a_site,
                target=target,
            )
            subplot_payloads.append((row, col, meshgrid_df, target_values))
            subplot_titles.append(
                f"{_axis_label(P3)}={col_val:.0f}, {_axis_label(P4)}={row_val:.0f}"
            )
            all_plot_values.append(_transform(target_values, target))

    all_plot_values = np.concatenate(all_plot_values)
    if global_value_min is not None and global_value_max is not None:
        value_min, value_max = global_value_min, global_value_max
    elif target == "mean":
        value_min = float(np.percentile(all_plot_values, 5))
        value_max = float(np.percentile(all_plot_values, 95))
    else:
        value_min = float(np.percentile(all_plot_values, 1))
        value_max = float(np.percentile(all_plot_values, 99.9))

    ticks = np.linspace(value_min, value_max, 6)
    tick_text = _exp_tick_text(ticks, target)
    colorbar_title = _colorbar_title(target)

    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[[{"type": "scene"} for _ in range(3)] for _ in range(3)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["yshift"] = -10

    fig.update_annotations(font=dict(size=title_font_size))

    for i, (row, col, meshgrid_df, target_values) in enumerate(subplot_payloads):
        add_volume_trace(
            fig,
            meshgrid_df,
            target_values,
            XYZ_PARAMS,
            row,
            col,
            show_colorbar=(i == 0),
            value_min=value_min,
            value_max=value_max,
            ticks=ticks,
            tick_text=tick_text,
            target=target,
            colorbar_title=colorbar_title,
        )

    fig.update_layout(
        title=dict(
            text=f"{_colorbar_title(target)}: {a_site} - {_axis_label(P5)}={p5_value:.0f}",
            font=dict(size=MEDIUM_FONT_SIZE),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        showlegend=False,
        width=width,
        height=height,
        margin={"t": 80, "b": 30, "l": 30, "r": 30},
    )
    return fig


def plot_p3p4_grids_for_p5_sweep(
    *,
    a_site="MBAI",
    target="mean",
    resolution=20,
    p5_values=None,
    title_font_size=SMALL_FONT_SIZE,
    width=FIGURE_WIDTH,
    height=FIGURE_WIDTH / 1.3,
):
    p5_min, p5_max = get_param_range(P5)
    if p5_values is None:
        p5_values = np.linspace(p5_min, p5_max, 5)[[1, 3]]

    print(f"Computing global color scale for a_site={a_site}, target={target}...")
    p3_min, p3_max = get_param_range(P3)
    p4_min, p4_max = get_param_range(P4)
    base_fixed_params = {
        col: np.mean(param_config[col]["range"])
        for col in CONTINUOUS_COLUMNS
        if col not in (*XYZ_PARAMS, P3, P4, P5)
    }

    all_values = []
    for p5_value in p5_values:
        fixed_params = {**base_fixed_params, P5: p5_value}
        for row_val in np.linspace(p4_min, p4_max, 3):
            for col_val in np.linspace(p3_min, p3_max, 3):
                _, target_values = build_mesh_and_predictions(
                    {**fixed_params, P3: col_val, P4: row_val},
                    resolution=resolution,
                    a_site=a_site,
                    target=target,
                )
                all_values.append(_transform(target_values, target))
    all_values = np.concatenate(all_values)

    if target == "mean":
        value_min = float(np.percentile(all_values, 5))
        value_max = float(np.percentile(all_values, 99))
    else:
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr = q3 - q1
        value_min, value_max = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        # value_min = float(np.percentile(all_values, 10))
        # value_max = float(np.percentile(all_values, 99))
        value_min = float(max(np.min(all_values), value_min))
        value_max = float(min(np.max(all_values), value_max))

    for p5_value in p5_values:
        fig = make_p3p4_grid_figure(
            p5_value,
            a_site=a_site,
            resolution=resolution,
            target=target,
            title_font_size=title_font_size,
            width=width,
            height=height,
            global_value_min=value_min,
            global_value_max=value_max,
        )
        _apply_publication_style(fig, colorbar_title=_colorbar_title(target))
        _save_fig(fig, f"volume_plot_{target}_{P5}_{p5_value:.0f}_{a_site}")


######################################################################################################

# %%

########## p3xp4 grids per a-site, sweeping p5 — mean (mirrors cell 17) ##########
for a_site in ["MBAI", "MeOMBAI", "ClMBAI"]:
    plot_p3p4_grids_for_p5_sweep(a_site=a_site, target="mean", resolution=30)
###################################################################################

# %%

########## p3xp4 grids per a-site, sweeping p5 — variance (mirrors cell 16) ##########
for a_site in ["MBAI", "MeOMBAI", "ClMBAI"]:
    plot_p3p4_grids_for_p5_sweep(a_site=a_site, target="variance", resolution=20)
#######################################################################################

# %%


########## a-site comparison at a fixed p3,p4,p5 point (mirrors cell 18) ##########
def make_a_site_comparison_figure(
    p3_value=None,
    p4_value=None,
    p5_value=None,
    *,
    a_sites=("MBAI", "MeOMBAI", "ClMBAI"),
    resolution=20,
    target="mean",
    title_font_size=SMALL_FONT_SIZE,
    width=FIGURE_WIDTH,
    height=500,
):
    p3_min, p3_max = get_param_range(P3)
    p4_min, p4_max = get_param_range(P4)
    p5_min, p5_max = get_param_range(P5)
    if p3_value is None:
        p3_value = (p3_min + p3_max) / 2
    if p4_value is None:
        p4_value = (p4_min + p4_max) / 2
    if p5_value is None:
        p5_value = (p5_min + p5_max) / 2

    fixed_params = {P3: p3_value, P4: p4_value, P5: p5_value}

    print(
        f"Computing global color scale across {len(a_sites)} a_sites (target={target})..."
    )
    subplot_payloads, all_plot_values = [], []
    for a_site in a_sites:
        meshgrid_df, target_values = build_mesh_and_predictions(
            fixed_params,
            resolution=resolution,
            a_site=a_site,
            target=target,
        )
        subplot_payloads.append((a_site, meshgrid_df, target_values))
        all_plot_values.append(_transform(target_values, target))
    all_plot_values = np.concatenate(all_plot_values)

    if target == "mean":
        value_min = float(np.percentile(all_plot_values, 5))
        value_max = float(np.percentile(all_plot_values, 95))
    else:
        value_min = float(np.percentile(all_plot_values, 10))
        value_max = float(np.percentile(all_plot_values, 99))

    ticks = np.linspace(value_min, value_max, 6)
    tick_text = _exp_tick_text(ticks, target)
    colorbar_title = _colorbar_title(target)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"} for _ in range(3)]],
        subplot_titles=list(a_sites),
        horizontal_spacing=0.08,
    )
    fig.update_annotations(font=dict(size=title_font_size))

    for idx, (a_site, meshgrid_df, target_values) in enumerate(
        subplot_payloads, start=1
    ):
        add_volume_trace(
            fig,
            meshgrid_df,
            target_values,
            XYZ_PARAMS,
            row=1,
            col=idx,
            show_colorbar=(idx == len(a_sites)),
            value_min=value_min,
            value_max=value_max,
            ticks=ticks,
            tick_text=tick_text,
            target=target,
            colorbar_title=colorbar_title,
        )

    param_str = (
        f"{_axis_label(P3)}={p3_value:.0f}, {_axis_label(P4)}={p4_value:.0f}, "
        f"{_axis_label(P5)}={p5_value:.0f}"
    )
    fig.update_layout(
        title=dict(
            text=f"A-site comparison ({param_str}) — {target}",
            font=dict(size=MEDIUM_FONT_SIZE),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        showlegend=False,
        width=width,
        height=height,
        margin={"t": 80, "b": 30, "l": 30, "r": 30},
    )
    return fig


fig = make_a_site_comparison_figure(
    p3_value=100,
    p4_value=600,
    p5_value=1.0,
    resolution=12,
    target="mean",
)
_apply_publication_style(fig, colorbar_title=_colorbar_title("mean"))
_save_fig(fig, "volume_plot_mean_a_site_comparison")
#####################################################################################

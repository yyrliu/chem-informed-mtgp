# %%
import os
import sys

# import matplotlib.transforms as mtransforms
# from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage.util import img_as_ubyte
from skimage.exposure import adjust_gamma
from chem_informed_mtgp.glcm_numba import glcm_map_numba
import cv2
from matplotlib.patches import ConnectionPatch, Rectangle

from skimage.feature import graycomatrix, graycoprops
from chem_informed_mtgp.glcm_numba import glcm_score_from_crop

import pandas as pd

from functools import partial
from matplotlib.transforms import offset_copy
from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH  # noqa: E402


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dash_lib.data_utils import query_samples, load_dataset  # noqa: E402

# %%

# %%

########## set up plotting parameters ##########
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)
################################################

db_path = Path(__file__).resolve().parent.parent / "mydb.duckdb"


# %%

samples = query_samples(db_path=db_path)
sorted_ids = samples.index.tolist()
ds = load_dataset(sorted_ids, "cbox_photo", no_load=True, db_path=db_path)
samples = ["5-07", "4-03"]

fig, ax = plt.subplots(
    1, len(samples), figsize=(4 * len(samples), 3), layout="constrained"
)

for sample_name in samples:
    sample_ds = ds.where(ds["sample_name"] == sample_name, drop=True)
    img = sample_ds["image"][0].values[..., ::-1].astype(np.uint8)
    ax[samples.index(sample_name)].imshow(img)
# %%
mask = ds["sample_name"].str.contains("9-14")
ds = ds.where(mask, drop=True)
img = ds["image"][0].values[..., ::-1].astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, channel in enumerate(["red", "green", "blue"]):
    axes[i].imshow(img[..., i], cmap="gray")
    axes[i].set_title(f"{channel} channel")
    # axes[i].axis("off")

img[..., 0] = 0
axes[3].imshow(img)

# %%


def zoom_in(
    zoom_region,
    img,
    ax,
    factor=4,
    inset_xy=(0.5, 0.5),
    origin="upper",
    connectors=(1, 2),
    box_kwargs=None,
    line_kwargs=None,
    imshow_kwargs=None,
):
    """
    Add a zoomed inset with manual connectors.

    Parameters
    ----------
    zoom_region : (x, y, w, h)
        Region in pixel coordinates.
    img : ndarray
        Image.
    ax : matplotlib.axes.Axes
        Parent axes.
    factor : float
        Scale factor for inset size.
    inset_xy : (float, float)
        Inset lower-left in axes fraction coords.
    origin : {"upper", "lower"}
        Image origin convention.
    connectors : iterable of int
        Indices of connectors to draw (0–3).
    box_kwargs : dict
        Styling for zoom rectangle.
    line_kwargs : dict
        Styling for connectors.
    """
    x, y, w, h = zoom_region
    x1, x2 = x, x + w
    y1, y2 = y, y + h

    H, W = img.shape[:2]

    # inset size (axes fraction)
    inset_w = (w / W) * factor
    inset_h = (h / H) * factor

    # y-direction handling
    ylim = (y2, y1) if origin == "upper" else (y1, y2)

    axins = ax.inset_axes(
        [*inset_xy, inset_w, inset_h],
        xlim=(x1, x2),
        ylim=ylim,
        xticks=[],
        yticks=[],
    )
    axins.imshow(
        img, origin=origin, **imshow_kwargs if imshow_kwargs is not None else {}
    )

    # --- styles ---
    box_kwargs = dict(edgecolor="k", facecolor="none", linewidth=1) | (
        dict(box_kwargs) if box_kwargs is not None else {}
    )
    line_kwargs = dict(color="k", linewidth=1) | (
        dict(line_kwargs) if line_kwargs is not None else {}
    )

    # apply box linestyle to inset ax splines for visual consistency
    if "edgecolor" in box_kwargs:
        for spine in axins.spines.values():
            spine.set_edgecolor(box_kwargs["edgecolor"])

    if "linewidth" in box_kwargs:
        for spine in axins.spines.values():
            spine.set_linewidth(box_kwargs["linewidth"])

    # --- zoom rectangle ---
    rect = Rectangle((x1, y1), w, h, **box_kwargs)
    ax.add_patch(rect)

    # --- connector definitions ---
    # (data coords on main ax) -> (axes coords on inset)
    corners = [
        ((x1, y1), (0, 1)),
        ((x2, y1), (1, 1)),
        ((x1, y2), (0, 0)),
        ((x2, y2), (1, 0)),
    ]

    for i in connectors:
        xyA, xyB = corners[i]
        con = ConnectionPatch(
            xyA=xyA,
            coordsA=ax.transData,
            xyB=xyB,
            coordsB=axins.transAxes,
            **line_kwargs,
        )
        ax.add_artist(con)

    return axins


def asym_pad(
    x0, y0, width, height, mutation_size, left=0.3, right=0.3, bottom=0.3, top=0.3
):
    x0 -= left * mutation_size
    y0 -= bottom * mutation_size
    x1 = x0 + width + (left + right) * mutation_size
    y1 = y0 + height + (bottom + top) * mutation_size
    return mpl.path.Path(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], closed=True
    )


# --- main plot ---
fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

h, w = img.shape[:2]
square_img = img[:, w // 2 - h // 2 - 80 : w // 2 + h // 2 - 80, :]

ax.imshow(square_img, origin="upper")

# example highlight (independent of zoom)
ax.add_patch(
    Rectangle(
        (1000 - (w // 2 - h // 2 - 80), 300),
        2100,
        2150,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
)

# --- insets ---
zoom_in(
    (1700, 700, 200, 200),
    square_img,
    ax,
    inset_xy=(0.7, 0.7),
    connectors=(0, 3),  # single connector
    box_kwargs=dict(edgecolor="gray", facecolor="none", linewidth=1.5),
    line_kwargs=dict(color="gray"),
)

zoom_in(
    (1200, 1500, 200, 200),
    square_img,
    ax,
    inset_xy=(0.7, 0.05),
    connectors=(1, 2),  # diagonal pair
)

plt.show()


def _compare_glcm_score(image_data, x_range, y_range):
    """Compare the Numba score against the skimage reference for one crop."""
    crop = getattr(image_data, "values", image_data).squeeze()[
        y_range[0] : y_range[1], x_range[0] : x_range[1], :
    ]
    hsv_crop = img_as_ubyte(rgb2hsv(crop))
    v_channel = hsv_crop[..., 2]

    numba_score = glcm_score_from_crop(image_data, x_range=x_range, y_range=y_range)

    glcm = graycomatrix(
        v_channel,
        distances=[1, 3, 5],
        angles=[i * np.pi / 4 for i in range(4)],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast").mean()
    energy = graycoprops(glcm, "energy").mean()
    reference_score = 1.0 / (contrast + 1e-8) * energy * 100

    return numba_score, reference_score, abs(numba_score - reference_score)


crop_specs = {
    "main crop": ((1000, 3100), (300, 2450)),
    "alternate crop": ((1200, 3000), (450, 2200)),
}

results = []
for crop_name, (x_range, y_range) in crop_specs.items():
    numba_score, reference_score, abs_diff = _compare_glcm_score(
        img, x_range=x_range, y_range=y_range
    )
    results.append(
        {
            "crop": crop_name,
            "numba_score": numba_score,
            "skimage_score": reference_score,
            "abs_diff": abs_diff,
        }
    )

score_report = pd.DataFrame(results)
print(score_report.to_string(index=False, float_format=lambda value: f"{value:.8f}"))


# plot a 8x8 grid of the GLCM matrices for demo purposes
hsv_crop = img_as_ubyte(rgb2hsv(img)) // 32
v_channel = hsv_crop[..., 2]
glcm = graycomatrix(
    v_channel,
    distances=[1, 3, 5],
    angles=[i * np.pi / 4 for i in range(4)],
    levels=8,
    # symmetric=True,
    # normed=True,
)
glcm = np.sum(glcm, axis=(2, 3))  # sum over angles

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(
    glcm,
    cmap="gray_r",
    norm=mpl.colors.LogNorm(),
    linewidth=0.5,
    linecolor="black",
    clip_on=False,
    xticklabels=False,
    yticklabels=False,
    ax=ax,
    cbar=False,
)
# add labels to the heatmap
ax.set_xlabel("Neighbor Pixel Intensity")
ax.set_ylabel("Reference Pixel Intensity")

# %%


# Define crop region for GLCM analysis
CROP_TOP = 300
CROP_LEFT = 1000
CROP_BOTTOM = 2450
CROP_RIGHT = 3100
crop_height = CROP_BOTTOM - CROP_TOP
crop_width = CROP_RIGHT - CROP_LEFT

# Extract and process image
cropped_img = img[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT, :]
image_in_hsv = img_as_ubyte(rgb2hsv(cropped_img))
glcm_maps = glcm_map_numba(image_in_hsv, (20, 20), 1, distances=[1, 3, 5])
h, w = img.shape[:2]

# Extract square region from center of image with offset
SQUARE_OFFSET = 80
square_left = w // 2 - h // 2 - SQUARE_OFFSET
square_right = w // 2 + h // 2 - SQUARE_OFFSET
square_img = img[:, square_left:square_right, :]

zoom_in_regions = [
    {
        "region": (760, 520, 150, 150),
        "connectors": (1, 2),
        "inset_xy": (0.5, 0.57),
        "title": "i) Smooth",
    },
    {
        "region": (1050, 1290, 180, 180),
        "connectors": (1, 2),
        "inset_xy": (0.53, 0.05),
        "title": "ii) Rough",
    },
]

# glcm_map is not square, so we need to adjust the x slice to match the square image width
glcm_map_shape = glcm_maps["contrast"].shape
x_fine_tune_slice = slice(
    0,
    int(
        1
        - (glcm_map_shape[0] - glcm_map_shape[1])
        / glcm_map_shape[0]
        * square_img.shape[1]
    ),
)

gray_img_float = rgb2gray(square_img)[:, x_fine_tune_slice]
gamma_corrected = img_as_ubyte(adjust_gamma(gray_img_float, gamma=1 / 2.2))
alpha = 255.0 / (np.percentile(gamma_corrected, 99) - np.percentile(gamma_corrected, 1))
beta = -np.percentile(gamma_corrected, 1) * alpha
final_img = cv2.convertScaleAbs(gamma_corrected, alpha=alpha, beta=beta)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(DOUBLE_COLUMN_WIDTH * 0.5, DOUBLE_COLUMN_WIDTH * 0.45),
    constrained_layout=True,
    dpi=300,
)
axes = axes.flatten()
im0 = axes[0].imshow(final_img, cmap="gray", origin="upper", vmin=0, vmax=255)
cbar = fig.colorbar(im0, ax=axes[0], shrink=0.8, pad=0.04)
cbar.set_ticks([])
cbar.ax.set_title("255", fontsize="small")
cbar.ax.set_xlabel("0", fontsize="small")
cbar.ax.set_ylabel(
    "Gray scale (gamma corrected)", rotation=270, labelpad=10, fontsize="small"
)
cbar.ax.yaxis.set_label_position("right")
axes[0].set_title(
    "PL image",
    y=1.0,
    pad=-4,
    x=0.015,
    ha="left",
    va="top",
    bbox=dict(facecolor=mpl.colors.to_rgba("white", 0.9), edgecolor="none"),
)

# highlight region (crop extent on the PL image)
roi_rect = plt.Rectangle(
    (CROP_LEFT - square_left, CROP_TOP),
    crop_width,
    crop_height,
    edgecolor="white",
    facecolor="none",
    linewidth=2,
)

axes[0].add_patch(roi_rect)

roi_text = axes[0].text(
    CROP_LEFT - square_left + crop_width,
    CROP_TOP,
    "GLCM ROI",
    color="black",
    fontsize="small",
    horizontalalignment="right",
    verticalalignment="bottom",
    bbox=dict(
        boxstyle=partial(asym_pad, left=0.3, right=0.3, bottom=0.01, top=0.3),
        facecolor="white",
        edgecolor="none",
    ),
)

pad_pt = 0.3 * roi_text.get_fontsize()
lw_pt = 2.0  # your rectangle's linewidth, already in points — no manual dpi math needed

roi_text.set_transform(
    offset_copy(
        axes[0].transData, fig=fig, x=-(pad_pt - lw_pt / 2), y=lw_pt / 2, units="points"
    )
)

pl_zoom_in_regions = [zoom_in_regions[0] | {"connectors": (0, 2)}, zoom_in_regions[1]]

for zoom_info in pl_zoom_in_regions:
    axins = zoom_in(
        zoom_info["region"],
        final_img,
        axes[0],
        factor=4.0 * (square_img.shape[1] / crop_width),
        inset_xy=zoom_info["inset_xy"],
        connectors=zoom_info["connectors"],
        box_kwargs=dict(edgecolor="w", linewidth=2),
        line_kwargs=dict(color="w", linewidth=2),
        imshow_kwargs=dict(cmap="gray", vmin=0, vmax=255),
    )
    axins.set_title(
        zoom_info["title"],
        y=1.0,
        pad=-2,
        x=0.015,
        fontsize="small",
        ha="left",
        va="top",
        bbox=dict(facecolor="w", edgecolor="none", pad=2),
    )

# transform zoom_in_regions from square_img coordinates to cropped_img coordinates for GLCM maps
zoom_in_regions_glcm = []
for zoom_info in zoom_in_regions:
    x, y, w_, h_ = zoom_info["region"]
    x_cropped = x - (CROP_LEFT - square_left)
    y_cropped = y - CROP_TOP
    zoom_in_regions_glcm.append(
        {**zoom_info, "region": (x_cropped, y_cropped, w_, h_), "factor": 4}
    )

for index, key in enumerate(["contrast", "energy", "score"], start=1):
    ax = axes[index]
    vmin, vmax = np.percentile(glcm_maps[key], [0.5, 99 if key == "contrast" else 99.5])
    im = ax.imshow(
        glcm_maps[key],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f"GLCM {key} map",
        y=1.0,
        pad=-4,
        x=0.015,
        ha="left",
        va="top",
        bbox=dict(facecolor=mpl.colors.to_rgba("white", 0.9), edgecolor="none"),
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    # Set colorbar ticks and labels to only show "low" and "high"
    cbar.set_ticks([])
    cbar.ax.set_title("high", fontsize="small")
    cbar.ax.set_xlabel("low", fontsize="small")
    cbar.ax.set_ylabel(
        f"{key.capitalize()} (a.u.)", rotation=270, labelpad=10, fontsize="small"
    )
    cbar.ax.yaxis.set_label_position("right")
    print(f"{key} mean: {glcm_maps[key].mean():.4f}, std: {glcm_maps[key].std():.4f}")
    for zoom_info in zoom_in_regions_glcm:
        axins = zoom_in(
            zoom_info["region"],
            glcm_maps[key],
            ax,
            factor=zoom_info["factor"],
            inset_xy=zoom_info["inset_xy"],
            connectors=zoom_info["connectors"],
            box_kwargs=dict(edgecolor="w", linewidth=2),
            line_kwargs=dict(color="w", linewidth=2),
            imshow_kwargs=dict(
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            ),
        )
        axins.set_title(
            zoom_info["title"],
            y=1.0,
            pad=-2,
            x=0.015,
            fontsize="small",
            ha="left",
            va="top",
            bbox=dict(facecolor="w", edgecolor="none", pad=2),
        )

for ax in axes:
    ax.axis("off")


fig.suptitle(
    "b)", x=-0.01, y=0.99, horizontalalignment="right", verticalalignment="top"
)
fig.get_layout_engine().set(wspace=0.02, w_pad=0.02, h_pad=0.03, hspace=0.03)

fig.savefig(
    "../manuscript_assets/2b_glcm_maps.svg",
    transparent=True,
    dpi=600,
)


############################### V2 all in one row plot ###############################

# %%
# Define crop region for GLCM analysis
CROP_TOP = 300
CROP_LEFT = 1000
CROP_BOTTOM = 2450
CROP_RIGHT = 3100
crop_height = CROP_BOTTOM - CROP_TOP
crop_width = CROP_RIGHT - CROP_LEFT

# Extract square region from center of image with offset
SQUARE_OFFSET = 80

zoom_in_regions = [
    # {
    #     "region": (760, 520, 150, 150),
    #     "connectors": (1, 2),
    #     "inset_xy": (0.5, 0.57),
    #     "title": "i) Smooth",
    # },
    # {
    #     "region": (1050, 1290, 180, 180),
    #     "connectors": (1, 2),
    #     "inset_xy": (0.53, 0.05),
    #     "title": "ii) Rough",
    # },
]


def prep_img(sample_name):
    samples = query_samples(db_path=db_path)
    sorted_ids = samples.index.tolist()
    ds = load_dataset(sorted_ids, "cbox_photo", no_load=True, db_path=db_path)
    sample_ds = ds.where(ds["sample_name"] == sample_name, drop=True)
    img = sample_ds["image"][0].values[..., ::-1].astype(np.uint8)

    # Extract and process image
    cropped_img = img[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT, :]
    image_in_hsv = img_as_ubyte(rgb2hsv(cropped_img))
    glcm_maps = glcm_map_numba(image_in_hsv, (10, 10), 1, distances=[1, 3, 5])
    h, w = img.shape[:2]
    square_left = w // 2 - h // 2 - SQUARE_OFFSET
    square_right = w // 2 + h // 2 - SQUARE_OFFSET
    square_img = img[:, square_left:square_right, :]
    # glcm_map is not square, so we need to adjust the x slice to match the square image width
    glcm_map_shape = glcm_maps["contrast"].shape
    x_fine_tune_slice = slice(
        0,
        int(
            1
            - (glcm_map_shape[0] - glcm_map_shape[1])
            / glcm_map_shape[0]
            * square_img.shape[1]
        ),
    )

    gray_img_float = rgb2gray(square_img)[:, x_fine_tune_slice]
    gamma_corrected = img_as_ubyte(adjust_gamma(gray_img_float, gamma=1 / 2.2))
    alpha = 255.0 / (
        np.percentile(gamma_corrected, 99) - np.percentile(gamma_corrected, 1)
    )
    beta = -np.percentile(gamma_corrected, 1) * alpha
    final_img = cv2.convertScaleAbs(gamma_corrected, alpha=alpha, beta=beta)

    return final_img, glcm_maps, square_img


def merge_2_imgs(img1, img2, direction="horizontal"):
    # Merge two images along the specified direction by cutting both images in half and concatenating them
    if direction == "horizontal":
        qrt1 = img1.shape[1] // 4
        qrt2 = img2.shape[1] // 4
        merged_img = np.concatenate(
            (img1[:, qrt1 : qrt1 * 3], img2[:, qrt2 : qrt2 * 3]), axis=1
        )
    elif direction == "vertical":
        qrt1 = img1.shape[0] // 4
        qrt2 = img2.shape[0] // 4
        merged_img = np.concatenate(
            (img1[qrt1 : qrt1 * 3, :], img2[qrt2 : qrt2 * 3, :]), axis=0
        )
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    merged_img = merged_img.astype(np.float32)  # Convert to float for NaN assignment

    # place a thin line of NaN values at the seam to indicate the merge
    width = 20  # width of the seam line
    if direction == "horizontal":
        merged_img[:, qrt1 * 2 - width // 2 : qrt1 * 2 + width // 2] = np.nan
    elif direction == "vertical":
        merged_img[qrt1 * 2 - width // 2 : qrt1 * 2 + width // 2, :] = np.nan

    return merged_img


prep_res = [prep_img(sample_name) for sample_name in ["5-07", "3-07"]]


# merge the final images and glcm maps from both samples
final_img = merge_2_imgs(prep_res[0][0], prep_res[1][0], direction="vertical")
glcm_maps = {
    key: merge_2_imgs(prep_res[0][1][key], prep_res[1][1][key], direction="vertical")
    for key in ["contrast", "energy", "score"]
}


def set_cbar_high_low(cbar, high_label="high", low_label="low"):
    """Set colorbar ticks and labels to only show "low" and "high"."""
    cbar.ax.set_ylabel(
        low_label, rotation=0, rotation_mode="anchor", va="center_baseline", ha="right"
    )
    twinx = cbar.ax.twinx()
    twinx.set_ylabel(
        high_label, rotation=0, rotation_mode="anchor", va="center_baseline", ha="left"
    )
    cbar.ax.set_yticks([])
    twinx.set_yticks([])
    cbar.set_ticks([])


fig, axes = plt.subplots(
    1,
    4,
    figsize=(DOUBLE_COLUMN_WIDTH * 0.75, DOUBLE_COLUMN_WIDTH * 0.235),
    constrained_layout=True,
    dpi=300,
)

fig.get_layout_engine().set(wspace=0.02, w_pad=0.02, h_pad=0.03, hspace=0.03)

axes = axes.flatten()
im0 = axes[0].imshow(final_img, cmap="gray", origin="upper", vmin=0, vmax=255)
cbar = fig.colorbar(im0, ax=axes[0], shrink=0.7, pad=0.03, orientation="horizontal")
set_cbar_high_low(cbar, high_label="255", low_label="0")

cbar.set_label("Gray scale (gamma corrected)", fontsize="small")
# cbar.ax.xaxis.set_label_position('top')

axes[0].set_title("PL image")

pad_pt = 0.3 * roi_text.get_fontsize()
lw_pt = 2.0  # your rectangle's linewidth, already in points — no manual dpi math needed

roi_text.set_transform(
    offset_copy(
        axes[0].transData, fig=fig, x=-(pad_pt - lw_pt / 2), y=lw_pt / 2, units="points"
    )
)

sample_label_smooth = axes[0].text(
    5,
    20,
    "Smooth",
    color="black",
    fontsize="small",
    verticalalignment="top",
    bbox=dict(
        boxstyle=partial(asym_pad, left=0.3, right=0.3, bottom=0.2, top=0.2),
        facecolor="white",
        edgecolor="none",
    ),
)

sample_label_smooth = axes[0].text(
    5,
    final_img.shape[0] // 2 + 35,
    "Rough",
    color="black",
    verticalalignment="top",
    fontsize="small",
    bbox=dict(
        boxstyle=partial(asym_pad, left=0.3, right=0.3, bottom=0.2, top=0.2),
        facecolor="white",
        edgecolor="none",
    ),
)

if len(zoom_in_regions) > 0:
    pl_zoom_in_regions = [
        zoom_in_regions[0] | {"connectors": (0, 2)},
        zoom_in_regions[1],
    ]
else:
    pl_zoom_in_regions = []

for zoom_info in pl_zoom_in_regions:
    axins = zoom_in(
        zoom_info["region"],
        final_img,
        axes[0],
        factor=4.0 * (square_img.shape[1] / crop_width),
        inset_xy=zoom_info["inset_xy"],
        connectors=zoom_info["connectors"],
        box_kwargs=dict(edgecolor="w", linewidth=2),
        line_kwargs=dict(color="w", linewidth=2),
        imshow_kwargs=dict(cmap="gray", vmin=0, vmax=255),
    )
    axins.set_title(
        zoom_info["title"],
        y=1.0,
        pad=-2,
        x=0.015,
        fontsize="small",
        ha="left",
        va="top",
        bbox=dict(facecolor="w", edgecolor="none", pad=2),
    )

# transform zoom_in_regions from square_img coordinates to cropped_img coordinates for GLCM maps
zoom_in_regions_glcm = []
for zoom_info in zoom_in_regions:
    x, y, w_, h_ = zoom_info["region"]
    x_cropped = x - (CROP_LEFT - square_left)
    y_cropped = y - CROP_TOP
    zoom_in_regions_glcm.append(
        {**zoom_info, "region": (x_cropped, y_cropped, w_, h_), "factor": 4}
    )

label_map = {
    "energy": r"$E_{GLCM}$",
    "contrast": r"$C_{GLCM}$",
    "score": r"$H_{PL}$",
}

for index, key in enumerate(["contrast", "energy", "score"], start=1):
    ax = axes[index]

    vmin, vmax = np.nanpercentile(glcm_maps[key], [0.5, 99.5])
    im = ax.imshow(
        glcm_maps[key],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"GLCM {key} map")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03, orientation="horizontal")

    # Set colorbar ticks and labels to only show "low" and "high"

    set_cbar_high_low(cbar)
    cbar.set_label(f"{key.capitalize()} ({label_map[key]} / a.u.)", fontsize="small")
    # cbar.ax.xaxis.set_label_position('top')
    print(f"{key} mean: {glcm_maps[key].mean():.4f}, std: {glcm_maps[key].std():.4f}")
    for zoom_info in zoom_in_regions_glcm:
        axins = zoom_in(
            zoom_info["region"],
            glcm_maps[key],
            ax,
            factor=zoom_info["factor"],
            inset_xy=zoom_info["inset_xy"],
            connectors=zoom_info["connectors"],
            box_kwargs=dict(edgecolor="w", linewidth=2),
            line_kwargs=dict(color="w", linewidth=2),
            imshow_kwargs=dict(
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            ),
        )
        axins.set_title(
            zoom_info["title"],
            y=1.0,
            pad=-2,
            x=0.015,
            fontsize="small",
            ha="left",
            va="top",
            bbox=dict(facecolor="w", edgecolor="none", pad=2),
        )

    sample_label_smooth = ax.text(
        5,
        20,
        "Smooth",
        color="black",
        fontsize="small",
        verticalalignment="top",
        bbox=dict(
            boxstyle=partial(asym_pad, left=0.3, right=0.3, bottom=0.2, top=0.2),
            facecolor="white",
            edgecolor="none",
        ),
    )

    sample_label_smooth = ax.text(
        5,
        glcm_maps[key].shape[0] // 2 + 33,
        "Rough",
        color="black",
        fontsize="small",
        verticalalignment="top",
        bbox=dict(
            boxstyle=partial(asym_pad, left=0.3, right=0.3, bottom=0.2, top=0.2),
            facecolor="white",
            edgecolor="none",
        ),
    )

for ax in axes:
    ax.axis("off")


fig.suptitle("b)", x=0.0, y=1.0, horizontalalignment="left", verticalalignment="top")

plt.draw()

for ax in fig.axes:
    ax_bbox = ax.get_position()
    print(f"ax_bbox: {ax_bbox.width}, {ax_bbox.height}")

fig.savefig(
    "../manuscript_assets/2b_glcm_maps_v2.svg",
    transparent=True,
    dpi=600,
)

# %%

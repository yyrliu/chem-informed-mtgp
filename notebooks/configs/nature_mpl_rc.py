from cycler import cycler

# 1. Multiplier for medium = 7.0 pt in print
MULTIPLIER = 5.0 / 3.0  # 1.667x

nature_rc = {
    # --- Fonts ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "sans-serif"],
    # Preset Keyword Mapping using default MPL ratios
    "axes.labelsize": "small",  # 7.00 pt
    "axes.titlesize": "medium",  # 7.00 pt
    "figure.labelsize": "small",  # 7.00 pt
    "figure.titlesize": "large",  # 7.00 pt
    "figure.titleweight": "bold",  # 7.00 pt
    "legend.title_fontsize": "small",  # 7.00 pt
    "xtick.labelsize": "small",  # 5.83 pt
    "ytick.labelsize": "small",  # 5.83 pt
    "legend.fontsize": "small",  # 5.83 pt
    # --- Vector Exports ---
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # --- Geometry & Ticks ---
    "figure.dpi": 300,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.major.size": 3,
    "xtick.minor.size": 2,
    "ytick.major.size": 3,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # --- Color Palette (Wong) ---
    "axes.prop_cycle": cycler(
        "color",
        [
            "#e69f00",
            "#56b4e9",
            "#009e73",
            "#f0e442",
            "#0072b2",
            "#d55e00",
            "#cc79a7",
            "#000000",
        ],
    ),
}

MM_TO_INCHES = 1 / 25.4  # Conversion factor

SINGLE_COLUMN_WIDTH_MM = 89.0
DOUBLE_COLUMN_WIDTH_MM = 183.0

SINGLE_COLUMN_WIDTH = SINGLE_COLUMN_WIDTH_MM * MM_TO_INCHES * MULTIPLIER
DOUBLE_COLUMN_WIDTH = DOUBLE_COLUMN_WIDTH_MM * MM_TO_INCHES * MULTIPLIER

a_site_colors = {"MBAI": "#4C72B0", "MeOMBAI": "#55A868", "ClMBAI": "#C44E52"}

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import networkx as nx

rng = np.random.default_rng(1)


def f(x):
    return np.sin(1.5 * x) + 0.3 * x


X_train = np.array([0.6, 1.8, 3.5, 4.0, 4.5, 5.8, 6.3]).reshape(-1, 1)
y_train = f(X_train).ravel() + rng.normal(0, 0.2, X_train.shape[0])
X_test = np.linspace(0, 7, 400).reshape(-1, 1)

kernel = ConstantKernel(1.0) * RBF(length_scale=0.8) + WhiteKernel(0.3)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
gp.fit(X_train, y_train)
mean, std = gp.predict(X_test, return_std=True)

# --- pick 2 points at local maxima of the variance ---
xs = X_test.ravel()
var = std**2  # plot variance as its own curve
loc_max = argrelextrema(std, np.greater, order=5)[0]
top2 = loc_max[np.argsort(std[loc_max])[-2:]]
x_next = xs[top2]


def bullseye(ax, x, y, color="#8172B3", size=100):
    ax.scatter(
        x, y, s=size, facecolors="none", edgecolors="k", linewidths=1.5, zorder=6
    )
    ax.scatter(x, y, s=size * 2, facecolors="k", linewidths=1.5, zorder=6, marker="+")
    ax.scatter(x, y, s=size * 0.65, facecolors="w", linewidths=0, zorder=6)
    ax.scatter(x, y, s=size * 0.20, c=color, zorder=7)


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3, 3), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
)

# --- top: BO view ---
ax1.fill_between(xs, mean - 2 * std, mean + 2 * std, color="#2572B0", alpha=0.22)
ax1.plot(X_test, f(X_test), color="#25B065", lw=1.8, ls="--", zorder=3)
ax1.plot(X_test, mean, color="#2572B0", lw=2.5, zorder=4)
ax1.scatter(
    X_train, y_train, c="#25B065", s=34, zorder=5, edgecolor="white", linewidth=0.7
)
# for xq in x_next:
#     bullseye(ax1, xq, (mean + 2*std)[xs == xq], size=100)

# --- bottom: variance as a curve ---
ax2.plot(xs, var, color="#8172B3", lw=2.0, zorder=4)
ax2.fill_between(xs, 0, var, color="#8172B3", alpha=0.15, zorder=1)
for xq in x_next:
    yq = var[xs == xq]
    ax2.axvline(xq, color="#8172B3", lw=1.0, ls=":", alpha=0.6, zorder=2)
    bullseye(ax2, xq, yq, size=100)

for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.01, y=0.12)

ax1.set_ylabel("objective")
ax2.set_ylabel("variance")
fig.tight_layout(pad=0.2)
fig.savefig("../manuscript_assets/gp_bo_var.svg", transparent=True, bbox_inches="tight")
plt.show()


def save_single_ax(ax, filename, padding=0.1):
    fig = ax.get_figure()
    extent = ax.get_tightbbox(fig.canvas.renderer)

    # Expand box slightly to prevent clipping
    extent = extent.padded(padding)

    fig.savefig(
        filename,
        bbox_inches=extent.transformed(fig.dpi_scale_trans.inverted()),
        transparent=True,
    )


for ax in (ax1, ax2):
    save_single_ax(
        ax, f"../manuscript_assets/gp_bo_var_{ax.get_ylabel()}.svg", padding=0.1
    )

# %%

rnd_num = np.random.default_rng().integers(1, 10000)
# override for reproducibility
rnd_num = 5
print(f"Random seed: {rnd_num}")
rng = np.random.default_rng(rnd_num)


def tanimoto(A, B):
    inter = A @ B.T
    a, b = A.sum(1), B.sum(1)
    return inter / np.maximum(a[:, None] + b[None, :] - inter, 1e-9)


bits, per_fam = 64, 6
families = ["MBAI", "MeOMBAI", "ClMBAI"]
fam_colors = {"MBAI": "#4C72B0", "MeOMBAI": "#55A868", "ClMBAI": "#C44E52"}

# shared scaffold bits so families genuinely overlap
scaffold = rng.random(bits) < 0.1
fps, labels = [], []
for fam in families:
    base = scaffold.copy()
    base[rng.random(bits) < 0.15] = True  # family-specific extras
    for _ in range(per_fam):
        flip = rng.random(bits) < 0.06
        fps.append((base ^ flip).astype(float))
        labels.append(fam)
fps = np.array(fps)

S = tanimoto(fps, fps)
np.fill_diagonal(S, 0)

# threshold on the LOWER tail so cross-family edges survive
thresh = np.quantile(S[S > 0], 0.61)
G = nx.Graph()
for i, fam in enumerate(labels):
    G.add_node(i, fam=fam)
for i in range(len(fps)):
    for j in range(i + 1, len(fps)):
        if S[i, j] > thresh:
            G.add_edge(i, j, w=S[i, j])

# report so you can see it's not empty
cross = [(u, v) for u, v in G.edges if labels[u] != labels[v]]
print(f"{G.number_of_edges()} edges, {len(cross)} cross-family")

# get equilateral triangle coordinates with (0,0) at the center, 2 ponts on the left, 1 point on the right
tri_coords = np.array(
    [[0.0, -np.sqrt(3) / 2], [0.0, np.sqrt(3) / 2], [np.sqrt(3), 0.0]]
)
# rotation matrix to rotate triangle 5 degrees clockwise
rot_deg = -5
rot = np.array(
    [
        [np.cos(np.radians(rot_deg)), -np.sin(np.radians(rot_deg))],
        [np.sin(np.radians(rot_deg)), np.cos(np.radians(rot_deg))],
    ]
)
tri_coords = tri_coords @ rot.T

# give each family a different starting anchor
anchor_multiplier = 1.0
base_anchors = {
    "ClMBAI": tri_coords[0],
    "MeOMBAI": tri_coords[1],
    "MBAI": tri_coords[2],
}
anchors = {
    k: (x * anchor_multiplier, y * anchor_multiplier)
    for k, (x, y) in base_anchors.items()
}
init = {i: np.array(anchors[labels[i]]) + 0.1 * rng.standard_normal(2) for i in G.nodes}

pos = nx.spring_layout(
    G, weight="w", pos=init, k=0.2, iterations=200, seed=int(rng.integers(10000))
)

# compute min/max of surviving edge weights for normalization
ws = [d["w"] for _, _, d in G.edges(data=True)]
w_min, w_max = min(ws), max(ws)

fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
for u, v, d in G.edges(data=True):
    is_cross = labels[u] != labels[v]
    # rescale weight to 0..1 across the actual range present
    t = (d["w"] - w_min) / max(w_max - w_min, 1e-9)
    lw = (0.4 + 4.5 * t) * (3.0 if is_cross else 1.0)
    ax.plot(
        *zip(pos[u], pos[v]),
        color="#E8A838" if is_cross else "#BBBBBB",
        lw=lw,
        alpha=0.85 if is_cross else 0.30,
        zorder=2 if is_cross else 1,
    )

for fam in families:
    idx = [i for i in G.nodes if labels[i] == fam]
    ax.scatter(
        [pos[i][0] for i in idx],
        [pos[i][1] for i in idx],
        s=180,
        c=fam_colors[fam],
        edgecolor="k",
        lw=1.5,
        label=fam,
        zorder=3,
    )

ax.legend(frameon=False, loc="best")
ax.axis("off")
fig.savefig(
    "../manuscript_assets/tanimoto_graph.svg",
    transparent=True,
    bbox_inches="tight",
    dpi=600,
)
# %%

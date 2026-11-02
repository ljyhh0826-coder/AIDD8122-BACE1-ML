"""Shared matplotlib style: a fixed categorical order, one-hue sequential maps, recessive chrome."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical hues, always assigned in this order (validated for colour-vision deficiency).
COVE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#b4b2a9"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_CMAP = "Blues"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def setup():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)

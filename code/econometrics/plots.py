"""Shared chart style: one palette and one look for the EDA and the result figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LIGHT_GREY = "#8f8d86", "#e4e3de"
TEXT, TEXT_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# two hues around a neutral midpoint: blue = first model better, orange = second better
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", [BLUE, LIGHT_GREY, ORANGE])


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": LIGHT_GREY, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": LIGHT_GREY, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "legend.frameon": False, "xtick.major.size": 0, "ytick.major.size": 0,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

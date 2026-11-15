from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

METHOD_COLORS = {"bf16": "#a3a29b", "gptq": "#2a78d6", "awq": "#1baf7a", "mxfp4": "#eb6834"}
# bf16과의 차이를 방법 색 빗금으로 표시
BF16_HATCH = "////"
ACCURACY_YMAX = 1.15
BAR_INCHES = 0.45
BAR_GAP = 1
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.titlesize": 13,
    "figure.constrained_layout.w_pad": 0.08,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.wspace": 0.08,
    "figure.constrained_layout.hspace": 0.08,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def get_panel_width(
    num_cases: int,
    num_series: int,
) -> float:
    return BAR_INCHES * num_cases * (num_series + BAR_GAP)


def get_figsize(
    panel_widths: list[float],
    num_rows: int,
) -> tuple[float, float]:
    return sum(panel_widths) + PANEL_MARGIN * len(panel_widths), PANEL_HEIGHT * num_rows + 1


def plot_bars(
    ax: Axes,
    case_labels: list[str],
    series: list[tuple[list[float], str | list[str], list[str]]],
    bf16_values: list[float] | None = None,
) -> None:
    case_width = len(series) + BAR_GAP
    for series_index, (values, color, texts) in enumerate(series):
        xs = [case_index * case_width + series_index - (len(series) - 1) / 2 for case_index in range(len(case_labels))]
        if bf16_values is not None:
            ax.bar(xs, [bf16_values[series_index]] * len(xs), width=1, facecolor="white", edgecolor=color, hatch=BF16_HATCH, linewidth=0)
        ax.bar(xs, values, width=1, color=color)
        for x, value, text in zip(xs, values, texts):
            top = max(value, 0 if bf16_values is None else bf16_values[series_index])
            ax.annotate(text, (x, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([case_index * case_width for case_index in range(len(case_labels))], case_labels)
    ax.set_xlim(-case_width / 2, (len(case_labels) - 0.5) * case_width)


def fit_bar_width(
    fig: Figure,
    axes: list[Axes],
) -> None:
    for _ in range(2):
        fig.canvas.draw()
        axes_width = sum(ax.get_window_extent().width for ax in axes) / fig.dpi
        num_bars = sum(ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes)
        fig.set_figwidth(fig.get_figwidth() + BAR_INCHES * num_bars - axes_width)


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")

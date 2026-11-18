from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter

MODEL_NAMES = {"Qwen/Qwen3-0.6B": "qwen3-0.6b", "Qwen/Qwen3-1.7B": "qwen3-1.7b"}
METHOD_COLORS = {"bf16": "#a3a29b", "gptq": "#2a78d6", "awq": "#1baf7a", "mxfp4": "#eb6834"}
BAR_INCHES = 0.45
BAR_GAP = 1
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9
# 손실은 %p, 비율은 %, 나머지는 format_value에서 kl과 배로 표시
PERCENT_UNITS = ["%p", "%"]
# 정확도는 100% 근처 막대 위 값이 잘리지 않게 여유를 두고 y축 고정
ACCURACY_YMAX = 1.1
# 잘림과 반복은 plot끼리 비교하기 쉽게 y축 고정
RATIO_YMAX = 0.3
# reasoning 길이도 plot끼리 비교하기 쉽게 y축 고정
LENGTH_YMAX = 2.0
# bf16 대비 배는 0.5배 간격 눈금
RATIO_YTICK = 0.5
# case가 적어도 제목이 잘리지 않는 너비
MIN_FIGURE_WIDTH = 9.0
# 막대 위 값 글자 크기와 막대에서 띄우는 point 수
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3
# 기준선 회색
NEUTRAL_COLOR = "#808080"
BASELINE_WIDTH = 0.8

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


def get_title(
    model_name: str,
    experiment_name: str,
) -> str:
    # 예시) "qwen3-0.6b math500 양자화 손실"
    return f"{MODEL_NAMES[model_name]} math500 {experiment_name}"


def get_figsize(
    num_cases: int,
    num_panels: int,
) -> tuple[float, float]:
    # panel을 가로로 놓고, case마다 막대 하나와 간격
    return max((BAR_INCHES * (1 + BAR_GAP) * num_cases + PANEL_MARGIN) * num_panels, MIN_FIGURE_WIDTH), PANEL_HEIGHT + 1


def format_value(
    value: float,
    unit: str,
) -> str:
    # 예시) "%p" → "5.2%p", "%" → "5.2%", "kl" → "0.123", "배" → "1.97배"
    # 반올림해 0이 된 음수가 "-0.0"으로 표시되지 않게 0.0을 더함
    if unit in PERCENT_UNITS:
        return f"{round(value * 100, 1) + 0.0:.1f}{unit}"
    if unit == "kl":
        return f"{value:.3f}"
    return f"{value:.2f}{unit}"


def plot_method_bars(
    ax: Axes,
    methods: list[str],
    values: list[float],
    unit: str,
) -> None:
    # 방법마다 방법 색 막대를 그리고 막대 위에 format_value로 값 표시
    case_width = 1 + BAR_GAP
    xs = [method_index * case_width for method_index in range(len(methods))]
    ax.bar(xs, values, width=1, color=[METHOD_COLORS[method] for method in methods])
    for x, value in zip(xs, values):
        ax.annotate(format_value(value, unit), (x, max(value, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    # 배는 1배가 기준
    ax.axhline(1 if unit == "배" else 0, color=NEUTRAL_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks(xs, methods)
    ax.set_xlim(-case_width / 2, (len(methods) - 0.5) * case_width)
    if unit in PERCENT_UNITS:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=unit))
    if unit == "배":
        ax.yaxis.set_major_locator(MultipleLocator(RATIO_YTICK))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}배"))


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")

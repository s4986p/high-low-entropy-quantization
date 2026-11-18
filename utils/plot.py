import math
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter

MODEL_NAMES = {"Qwen/Qwen3-0.6B": "qwen3-0.6b", "Qwen/Qwen3-1.7B": "qwen3-1.7b"}
METHOD_COLORS = {"bf16": "#a3a29b", "gptq": "#2a78d6", "awq": "#1baf7a", "mxfp4": "#eb6834"}
ENTROPY_NAMES = {"low": "저엔트로피 위치", "high": "고엔트로피 위치"}
ENTROPY_COLORS = {"low": "#9ecae1", "high": "#08519c"}
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
# case가 적어도 제목과 범례가 잘리지 않는 너비
MIN_FIGURE_WIDTH = 9.0
# 막대 위 값이 그림 밖으로 잘리지 않게 y축 위 여백 비율
VALUE_MARGIN = 0.15
# 막대 위 값 글자 크기와 막대에서 띄우는 point 수
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3
# 기준선 회색
NEUTRAL_COLOR = "#808080"
BASELINE_WIDTH = 0.8
# bf16과 bf16 샘플링 없는 양자화 모델은 막대 대신 점선
REFERENCE_LINESTYLE = "--"
# 단어 plot은 가로 막대, 단어마다 bf16과 양자화 모델 막대 둘
OVERTHINKING_WORDS = ["wait", "but", "however", "hmm", "wrong", "alternatively", "maybe", "perhaps"]
CHECK_WORDS = ["verify", "confirm", "remember", "sure"]
WORD_BAR_INCHES = 0.12
WORD_PANEL_WIDTH = 4.5
# 단어 plot 한 행의 제목과 x축 눈금 높이
WORD_ROW_MARGIN = 1.2
# 막대 옆 배가 잘리지 않게 x축 오른쪽 여백 비율
WORD_VALUE_MARGIN = 0.2

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


def get_panel_width(
    num_cases: int,
    num_series: int,
) -> float:
    # case마다 series 막대와 간격
    return BAR_INCHES * num_cases * (num_series + BAR_GAP)


def get_figsize(
    panel_widths: list[float],
    num_rows: int,
) -> tuple[float, float]:
    # panel을 가로로, 묶음을 세로로 놓음
    return max(sum(panel_widths) + PANEL_MARGIN * len(panel_widths), MIN_FIGURE_WIDTH), PANEL_HEIGHT * num_rows + 1


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


def plot_bars(
    ax: Axes,
    cases: list[str],
    series: list[tuple[list[float], str | list[str]]],
    unit: str,
) -> None:
    # case마다 series 막대를 나란히 그리고 막대 위에 format_value로 값 표시
    # series는 (값, 색), 색이 list면 case마다 다른 색
    # 값이 nan이면 그 막대는 비워 둠
    # 예시) cases ["숫자", "기호"], series [(저엔트로피 위치 값, 연한 파랑), (고엔트로피 위치 값, 진한 파랑)]
    case_width = len(series) + BAR_GAP
    for series_index, (values, color) in enumerate(series):
        xs = [case_index * case_width + series_index - (len(series) - 1) / 2 for case_index in range(len(cases))]
        ax.bar(xs, values, width=1, color=color)
        for x, value in zip(xs, values):
            if not math.isnan(value):
                ax.annotate(format_value(value, unit), (x, max(value, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    set_bar_axis(ax, cases, case_width, unit)


def set_bar_axis(
    ax: Axes,
    cases: list[str],
    case_width: int,
    unit: str,
) -> None:
    # 기준선, case 눈금, unit에 맞는 y축 눈금 설정
    # 배는 1배가 기준
    ax.axhline(1 if unit == "배" else 0, color=NEUTRAL_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([case_index * case_width for case_index in range(len(cases))], cases)
    ax.set_xlim(-case_width / 2, (len(cases) - 0.5) * case_width)
    if unit in PERCENT_UNITS:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=unit))
    if unit == "배":
        ax.yaxis.set_major_locator(MultipleLocator(RATIO_YTICK))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}배"))


def get_words_figsize(
    num_words: int,
    num_rows: int,
) -> tuple[float, float]:
    # 단어 묶음 3개를 가로로, 방법을 세로로 놓음
    row_height = WORD_BAR_INCHES * max(num_words, len(OVERTHINKING_WORDS + CHECK_WORDS)) * (2 + BAR_GAP) + WORD_ROW_MARGIN
    return WORD_PANEL_WIDTH * 3, row_height * num_rows + 1


def plot_words(
    axes: list[Axes],
    bf16_rates: dict[str, float],
    quantized_rates: dict[str, float],
    method: str,
    num_words: int,
) -> None:
    # 늘어난 단어, 줄어든 단어, overthinking과 확인 단어 panel에 bf16과 양자화 모델의 단어 빈도를 나란히 그림
    # 늘어난 단어와 줄어든 단어는 단어 빈도 차이 순, 막대 옆에 bf16 대비 배 표시
    words = sorted(bf16_rates.keys() | quantized_rates.keys(), key=lambda word: quantized_rates.get(word, 0) - bf16_rates.get(word, 0), reverse=True)
    word_groups = [("늘어난 단어", words[:num_words]), ("줄어든 단어", words[::-1][:num_words]), ("overthinking, 확인 단어", OVERTHINKING_WORDS + CHECK_WORDS)]
    word_width = 2 + BAR_GAP
    num_slots = max(len(group_words) for _, group_words in word_groups)
    for ax, (title, group_words) in zip(axes, word_groups):
        for series_index, (color, rates) in enumerate([(METHOD_COLORS["bf16"], bf16_rates), (METHOD_COLORS[method], quantized_rates)]):
            ys = [word_index * word_width + series_index - 0.5 for word_index in range(len(group_words))]
            ax.barh(ys, [rates.get(word, 0) for word in group_words], height=1, color=color)
        for word_index, word in enumerate(group_words):
            bf16_rate, quantized_rate = bf16_rates.get(word, 0), quantized_rates.get(word, 0)
            # bf16이 쓰지 않은 단어는 배 대신 "새 단어"
            text = format_value(quantized_rate / bf16_rate, "배") if bf16_rate else "새 단어"
            ax.annotate(text, (max(bf16_rate, quantized_rate), word_index * word_width), xytext=(VALUE_OFFSET, 0), textcoords="offset points", va="center", fontsize=VALUE_FONTSIZE)
        ax.set_yticks([word_index * word_width for word_index in range(len(group_words))], group_words)
        ax.set_ylim((num_slots - 0.5) * word_width, -word_width / 2)
        ax.margins(x=WORD_VALUE_MARGIN)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("단어 빈도")


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")

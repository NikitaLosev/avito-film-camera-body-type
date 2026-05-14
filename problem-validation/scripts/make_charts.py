import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "assets" / "charts"

CLASS_ORDER = [
    "compact_point_and_shoot",
    "SLR",
    "instant",
    "rangefinder_viewfinder",
    "TLR",
    "other_unknown",
]

CLASS_LABELS_RU = {
    "compact_point_and_shoot": "компактные\nкамеры",
    "SLR": "зеркальные\nSLR",
    "instant": "моментальные\nPolaroid/Instax",
    "rangefinder_viewfinder": "дальномерные /\nвидоискательные",
    "TLR": "двухобъективные\nTLR",
    "other_unknown": "не камера /\nнеясно",
}

WINNING_MODALITY_LABELS_RU = {
    "both": "фото и текст",
    "photo": "фото",
    "text": "текст",
    "neither": "нет ответа",
}

STRICT_REGEX_HITS = 2
CATEGORY_COUNT = 59_758
TOP30_LOOKUP_HITS = 30


def load_inputs(data_dir):
    """Загружаем ручную разметку и проверку модальностей"""
    labels = pd.read_csv(data_dir / "avito_camera_labels.csv")
    modality = pd.read_csv(data_dir / "modality_check.csv")
    return labels, modality


def get_real_cameras(labels):
    """Оставляем только объявления с реальной камерой"""
    return labels[labels["is_camera"] == "yes"].copy()


def get_class_counts(real_cameras):
    """Считаем количество карточек в каждом классе"""
    return real_cameras["label"].value_counts().reindex(CLASS_ORDER).dropna()


def render_bar_chart(
    series,
    out_path,
    *,
    title,
    ylabel,
    xlabel="",
    figsize=(9, 4.8),
    ylim=None,
    label_fmt=None,
    rotation=25,
):
    """Строит столбчатую диаграмму и сохраняет её в файл"""
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if label_fmt is None:
        ax.bar_label(ax.containers[0])
    else:
        ax.bar_label(ax.containers[0], fmt=label_fmt)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_class_distribution(class_counts, out_dir):
    """График 01: типы камер в ручной выборке"""
    render_bar_chart(
        class_counts.rename(index=CLASS_LABELS_RU),
        out_dir / "01_class_distribution.png",
        title="Типы камер в ручной выборке",
        xlabel="",
        ylabel="Количество карточек",
    )


def plot_signal_rates(labels, n_total, n_real, n_labelable, out_dir):
    """График 02: что видно в карточках"""
    metrics = pd.Series(
        {
            "реальные\nкамеры": n_real / n_total,
            "понятный\nтип корпуса": n_labelable / n_total,
            "слово типа\nв заголовке": STRICT_REGEX_HITS / n_total,
            "слово типа\nили прокси": (labels["title_keyword"] == "yes").mean(),
            "бренд или\nмодель": (labels["brand_model_in_title"] == "yes").mean(),
            "главного фото\nдостаточно": (labels["main_photo_enough"] == "yes").mean(),
            "фото даёт\nсигнал": labels["main_photo_enough"]
            .isin(["yes", "partial"])
            .mean(),
        }
    )
    render_bar_chart(
        metrics * 100,
        out_dir / "02_signal_rates.png",
        title="Что можно извлечь из карточек",
        ylabel="Доля, %",
        ylim=(0, 105),
        label_fmt="%.0f%%",
        rotation=0,
    )


def plot_baseline_ladder(labels, n_total, n_real, class_counts, out_dir):
    """График 03: покрытие простых эвристик"""
    baseline = pd.Series(
        {
            "самый частый\nкласс": class_counts.max() / n_total,
            "прямые слова\nтипа": STRICT_REGEX_HITS / n_total,
            "слова типа\n+ прокси": (labels["title_keyword"] == "yes").mean(),
            "справочник\nтоп-30": TOP30_LOOKUP_HITS / n_total,
        }
    )
    render_bar_chart(
        baseline * 100,
        out_dir / "03_baseline_ladder.png",
        title="Покрытие простых правил на ручном срезе",
        ylabel="Доля карточек, %",
        ylim=(0, 100),
        label_fmt="%.0f%%",
    )


def plot_modality_hard_cases(modality, out_dir):
    """График 04: фото и текст на спорных карточках"""
    photo_match = (modality["label_photo_only"] == modality["label_full"]).mean()
    text_match = (modality["label_text_only"] == modality["label_full"]).mean()
    mod_metrics = pd.Series(
        {
            "только фото": photo_match,
            "только текст": text_match,
        }
    )
    render_bar_chart(
        mod_metrics * 100,
        out_dir / "04_modality_hard_cases.png",
        title="Фото и текст на 8 спорных карточках",
        ylabel="Совпадение с итоговой разметкой, %",
        figsize=(7.5, 4.5),
        ylim=(0, 105),
        label_fmt="%.0f%%",
        rotation=0,
    )
    return photo_match, text_match


def plot_winning_modality(modality, out_dir):
    """График 05: источник решающего сигнала"""
    win_counts = (
        modality["winning_modality"]
        .value_counts()
        .reindex(["both", "photo", "text", "neither"])
        .fillna(0)
        .rename(index=WINNING_MODALITY_LABELS_RU)
    )
    render_bar_chart(
        win_counts,
        out_dir / "05_winning_modality.png",
        title="Источник решающего сигнала на спорных карточках",
        xlabel="",
        ylabel="Количество карточек",
        figsize=(7.5, 4.5),
        label_fmt="%.0f",
        rotation=0,
    )
    return win_counts


def plot_opportunity_proxy(n_total, n_labelable, out_dir):
    """График 06: предварительная оценка масштаба категории"""
    labelable_rate = n_labelable / n_total
    opportunity = pd.Series(
        {
            "всего объявлений\nв категории": CATEGORY_COUNT,
            "потенциально\nс понятным типом": CATEGORY_COUNT * labelable_rate,
        }
    )
    render_bar_chart(
        opportunity,
        out_dir / "06_opportunity_proxy.png",
        title="Оценка масштаба по ручному срезу",
        xlabel="",
        ylabel="Количество объявлений",
        figsize=(8, 4.6),
        label_fmt="%.0f",
        rotation=0,
    )


def build_summary(
    labels,
    n_total,
    n_real,
    n_labelable,
    class_counts,
    hard_cases_n,
    photo_match,
    text_match,
    win_counts,
):
    """Собирает итоговые числа первого этапа в словарь"""
    return {
        "n_rows": n_total,
        "real_cameras": int(n_real),
        "category_purity": round(n_real / n_total, 4),
        "labelable_real_cameras": int(n_labelable),
        "labelable_rate_among_real_cameras": round(n_labelable / n_real, 4),
        "top_class": class_counts.idxmax(),
        "top_class_share_among_real_cameras": round(class_counts.max() / n_real, 4),
        "strict_class_word_regex_rate": round(STRICT_REGEX_HITS / n_total, 4),
        "keyword_proxy_rate": round((labels["title_keyword"] == "yes").mean(), 4),
        "brand_model_in_title_rate": round(
            (labels["brand_model_in_title"] == "yes").mean(), 4
        ),
        "main_photo_enough_rate": round(
            (labels["main_photo_enough"] == "yes").mean(), 4
        ),
        "main_photo_yes_partial_rate": round(
            labels["main_photo_enough"].isin(["yes", "partial"]).mean(), 4
        ),
        "hard_cases_n": int(hard_cases_n),
        "photo_only_match_full": round(photo_match, 4),
        "text_only_match_full": round(text_match, 4),
        "winning_modality_counts": {k: int(v) for k, v in win_counts.to_dict().items()},
        "category_count_manual_snapshot": CATEGORY_COUNT,
        "estimated_labelable_ads_from_sample": int(CATEGORY_COUNT * n_labelable / n_total),
    }


def save_summary(summary, data_dir):
    """Сохраняет итоговые числа в JSON"""
    path = data_dir / "metrics_stage01.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """Считает числа, строит графики и сохраняет итоговый JSON"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels, modality = load_inputs(DATA_DIR)
    real_cameras = get_real_cameras(labels)
    labelable = real_cameras[real_cameras["label"] != "other_unknown"]
    class_counts = get_class_counts(real_cameras)

    n_total = len(labels)
    n_real = len(real_cameras)
    n_labelable = len(labelable)

    plot_class_distribution(class_counts, OUT_DIR)
    plot_signal_rates(labels, n_total, n_real, n_labelable, OUT_DIR)
    plot_baseline_ladder(labels, n_total, n_real, class_counts, OUT_DIR)
    photo_match, text_match = plot_modality_hard_cases(modality, OUT_DIR)
    win_counts = plot_winning_modality(modality, OUT_DIR)
    plot_opportunity_proxy(n_total, n_labelable, OUT_DIR)

    summary = build_summary(
        labels,
        n_total,
        n_real,
        n_labelable,
        class_counts,
        len(modality),
        photo_match,
        text_match,
        win_counts,
    )
    save_summary(summary, DATA_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

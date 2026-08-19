#!/usr/bin/env python3
"""Plot the cross-group paired performance differences from persisted summaries."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = (
    ("south_german", "German Credit"),
    ("taiwan", "Taiwan"),
    ("home_credit", "Home Credit"),
    ("lending_club", "Lending Club"),
)
INTERPRETABLE_MODELS = ("LR", "EBM")
ENSEMBLES = ("XGBoost", "CatBoost")
EXPECTED_CONTRASTS = {
    (interpretable, ensemble)
    for interpretable in INTERPRETABLE_MODELS
    for ensemble in ENSEMBLES
}
PLOTTED_METRICS = ("auc", "emp")
REQUIRED_COLUMNS = {
    "model_a",
    "model_b",
    "metric",
    "delta_point",
    "delta_ci_low",
    "delta_ci_high",
    "ci_method",
    "ci_level",
    "n_bootstrap",
}


@dataclass(frozen=True)
class Contrast:
    dataset_key: str
    dataset_label: str
    model_a: str
    model_b: str
    metric: str
    point: float
    low: float
    high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Root directory containing <dataset>/auc_delta_bootstrap.csv for "
            "south_german, taiwan, home_credit and lending_club."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the vector PDF.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        help="Optional output path for a raster PNG preview.",
    )
    return parser.parse_args()


def delta_emp_to_percentage_points(delta_emp_raw: float) -> float:
    """Convert an absolute EMP fraction difference to percentage points."""
    return 100.0 * delta_emp_raw


def _as_float(row: dict[str, str], column: str, source: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid numeric value in column '{column}'") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}: non-finite value in column '{column}'")
    return value


def _read_dataset(source: Path, dataset_key: str, dataset_label: str) -> list[Contrast]:
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing authoritative paired-results file for {dataset_label}: {source}"
        )

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{source}: missing required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    selected: list[Contrast] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        metric = row["metric"]
        pair = (row["model_a"], row["model_b"])
        if metric not in PLOTTED_METRICS or pair not in EXPECTED_CONTRASTS:
            continue

        key = (metric, *pair)
        if key in seen:
            raise ValueError(
                f"{source}: duplicate row for metric={metric}, contrast={pair[0]}-{pair[1]}"
            )
        seen.add(key)

        if row["ci_method"] != "percentile":
            raise ValueError(
                f"{source}: expected percentile intervals, found '{row['ci_method']}'"
            )
        ci_level = _as_float(row, "ci_level", source)
        if not math.isclose(ci_level, 0.95, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{source}: expected ci_level=0.95, found {ci_level}")
        try:
            n_bootstrap = int(row["n_bootstrap"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid n_bootstrap value") from exc
        if n_bootstrap <= 0:
            raise ValueError(f"{source}: n_bootstrap must be positive")

        point = _as_float(row, "delta_point", source)
        low = _as_float(row, "delta_ci_low", source)
        high = _as_float(row, "delta_ci_high", source)
        if low > high:
            raise ValueError(
                f"{source}: interval bounds are reversed for {metric}, {pair[0]}-{pair[1]}"
            )

        selected.append(
            Contrast(
                dataset_key=dataset_key,
                dataset_label=dataset_label,
                model_a=pair[0],
                model_b=pair[1],
                metric=metric,
                point=point,
                low=low,
                high=high,
            )
        )

    for metric in PLOTTED_METRICS:
        actual = {(row.model_a, row.model_b) for row in selected if row.metric == metric}
        if actual != EXPECTED_CONTRASTS:
            missing = EXPECTED_CONTRASTS - actual
            extra = actual - EXPECTED_CONTRASTS
            details = []
            if missing:
                details.append(
                    "missing " + ", ".join(f"{a}-{b}" for a, b in sorted(missing))
                )
            if extra:
                details.append(
                    "unexpected " + ", ".join(f"{a}-{b}" for a, b in sorted(extra))
                )
            raise ValueError(
                f"{source}: expected exactly four cross-group contrasts for "
                f"metric '{metric}' ({'; '.join(details)})"
            )

    return selected


def load_contrasts(input_root: Path) -> list[Contrast]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    contrasts: list[Contrast] = []
    for dataset_key, dataset_label in DATASETS:
        source = input_root / dataset_key / "auc_delta_bootstrap.csv"
        contrasts.extend(_read_dataset(source, dataset_key, dataset_label))
    return contrasts


def _lookup(
    contrasts: Iterable[Contrast], metric: str, dataset_key: str, model_a: str, model_b: str
) -> Contrast:
    matches = [
        row
        for row in contrasts
        if (row.metric, row.dataset_key, row.model_a, row.model_b)
        == (metric, dataset_key, model_a, model_b)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Internal validation error: expected one row for "
            f"{dataset_key}, {metric}, {model_a}-{model_b}; found {len(matches)}"
        )
    return matches[0]


def make_figure(contrasts: list[Contrast]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.35), sharey=True)
    row_keys = [
        (dataset_key, dataset_label, model_a)
        for dataset_key, dataset_label in DATASETS
        for model_a in INTERPRETABLE_MODELS
    ]
    y_positions = list(reversed(range(len(row_keys))))
    y_by_key = {
        (dataset_key, model_a): y
        for (dataset_key, _dataset_label, model_a), y in zip(row_keys, y_positions)
    }
    y_labels = [f"{dataset_label} – {model_a}" for _, dataset_label, model_a in row_keys]

    ensemble_styles = {
        "XGBoost": {"color": "#0072B2", "marker": "o", "offset": 0.12},
        "CatBoost": {"color": "#D55E00", "marker": "s", "offset": -0.12},
    }

    for axis, metric in zip(axes, PLOTTED_METRICS):
        for ensemble in ENSEMBLES:
            style = ensemble_styles[ensemble]
            xs: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            ys: list[float] = []
            for dataset_key, _dataset_label, model_a in row_keys:
                row = _lookup(contrasts, metric, dataset_key, model_a, ensemble)
                if metric == "emp":
                    delta_emp_raw = row.point
                    delta_emp_low_raw = row.low
                    delta_emp_high_raw = row.high
                    delta_emp_percentage_points = delta_emp_to_percentage_points(
                        delta_emp_raw
                    )
                    delta_emp_low_percentage_points = delta_emp_to_percentage_points(
                        delta_emp_low_raw
                    )
                    delta_emp_high_percentage_points = delta_emp_to_percentage_points(
                        delta_emp_high_raw
                    )
                    xs.append(delta_emp_percentage_points)
                    lows.append(
                        delta_emp_percentage_points - delta_emp_low_percentage_points
                    )
                    highs.append(
                        delta_emp_high_percentage_points - delta_emp_percentage_points
                    )
                else:
                    xs.append(row.point)
                    lows.append(row.point - row.low)
                    highs.append(row.high - row.point)
                ys.append(y_by_key[(dataset_key, model_a)] + style["offset"])
            axis.errorbar(
                xs,
                ys,
                xerr=[lows, highs],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.25,
                capsize=2.2,
                capthick=1.0,
                markersize=4.6,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linestyle="none",
                label=f"vs. {ensemble}",
                zorder=3,
            )

        axis.axvline(0.0, color="#4D4D4D", linewidth=0.9, linestyle="--", zorder=1)
        for boundary in (5.5, 3.5, 1.5):
            axis.axhline(boundary, color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
        axis.set_ylim(-0.55, 7.55)
        axis.tick_params(axis="y", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)

    axes[0].set_title("A   ROC-AUC")
    axes[0].set_xlabel(r"Difference in ROC-AUC ($\Delta$)")
    axes[1].set_title("B   EMP difference")
    axes[1].set_xlabel(r"$\Delta$EMP (percentage points)")
    axes[0].set_yticks(y_positions, y_labels)
    axes[0].tick_params(labelleft=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.005),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.255, right=0.965, top=0.92, bottom=0.145, wspace=0.22)
    return fig


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".pdf":
        raise ValueError(f"--output must name a PDF file, received: {args.output}")
    if args.png is not None and args.png.suffix.lower() != ".png":
        raise ValueError(f"--png must name a PNG file, received: {args.png}")

    contrasts = load_contrasts(args.input)
    figure = make_figure(contrasts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixed_timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)
    figure.savefig(
        args.output,
        format="pdf",
        metadata={
            "Title": "Paired cross-group differences in ROC-AUC and EMP",
            "Author": "Marius Calenborn",
            "Creator": "scripts/plot_pairwise_forest.py",
            "CreationDate": fixed_timestamp,
            "ModDate": fixed_timestamp,
        },
    )
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.png, format="png", dpi=300)
    plt.close(figure)

    print(f"Validated {len(contrasts)} plotted rows from {args.input}")
    print(f"Wrote vector PDF: {args.output}")
    if args.png is not None:
        print(f"Wrote PNG preview: {args.png}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

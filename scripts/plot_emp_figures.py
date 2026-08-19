#!/usr/bin/env python3
"""Create the EMP level and ROI-sensitivity figures from persisted results."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATASETS = (
    ("south_german", "German Credit\n(n = 1k)"),
    ("taiwan", "Taiwan\n(n = 30k)"),
    ("home_credit", "Home Credit\n(n = 307k)"),
    ("lending_club", "Lending Club\n(n = 1.35M)"),
)
MODELS = ("LR", "EBM", "XGBoost", "CatBoost")
SHORT_MODELS = ("LR", "EBM", "XGB", "CatB")
COLORS = {
    "LR": "#d62728",
    "EBM": "#2ca02c",
    "XGBoost": "#7f7f7f",
    "CatBoost": "#4d4d4d",
}
CI_REQUIRED_COLUMNS = {
    "model",
    "metric",
    "point",
    "ci_low",
    "ci_high",
    "ci_method",
    "ci_level",
    "n_bootstrap",
}
GRID_REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "roi",
    "emp",
    "eta",
    "prior_default_frozen",
    "baseline_roi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-split",
        type=Path,
        required=True,
        help=(
            "Root containing <dataset>/bootstrap_ci.csv and "
            "<dataset>/economic_config.json."
        ),
    )
    parser.add_argument(
        "--posthoc-input",
        type=Path,
        required=True,
        help="Authoritative unscaled emp_roi_grid.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for fig_emp.pdf and fig_emp_sensitivity.pdf.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write deterministic 300-dpi PNG previews.",
    )
    return parser.parse_args()


def emp_to_percent(emp_raw: float) -> float:
    """Convert an EMP fraction to percent of principal."""
    return 100.0 * emp_raw


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source}: missing required columns: {', '.join(sorted(missing))}"
        )


def _require_finite_fraction(values: pd.Series, source: Path, column: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise ValueError(f"{source}: column '{column}' contains non-finite values")
    if ((numeric < 0.0) | (numeric > 1.0)).any():
        raise ValueError(
            f"{source}: column '{column}' is not on the expected raw fraction scale [0, 1]"
        )


def load_emp_intervals(fixed_split: Path) -> dict[str, pd.DataFrame]:
    if not fixed_split.is_dir():
        raise NotADirectoryError(f"Fixed-split input is not a directory: {fixed_split}")

    tables: dict[str, pd.DataFrame] = {}
    for dataset_key, dataset_label in DATASETS:
        source = fixed_split / dataset_key / "bootstrap_ci.csv"
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing bootstrap summary for {dataset_label}: {source}"
            )
        frame = pd.read_csv(source)
        _require_columns(frame, CI_REQUIRED_COLUMNS, source)
        emp_rows = frame.loc[frame["metric"].eq("emp")].copy()
        if emp_rows["model"].duplicated().any() or set(emp_rows["model"]) != set(MODELS):
            raise ValueError(
                f"{source}: expected exactly one EMP row for each of {', '.join(MODELS)}"
            )
        if not emp_rows["ci_method"].eq("percentile").all():
            raise ValueError(f"{source}: EMP intervals must use the percentile method")
        if not emp_rows["ci_level"].map(
            lambda value: math.isclose(float(value), 0.95, rel_tol=0.0, abs_tol=1e-12)
        ).all():
            raise ValueError(f"{source}: EMP intervals must have ci_level=0.95")
        if (pd.to_numeric(emp_rows["n_bootstrap"], errors="coerce") <= 0).any():
            raise ValueError(f"{source}: n_bootstrap must be positive")
        for column in ("point", "ci_low", "ci_high"):
            _require_finite_fraction(emp_rows[column], source, column)
        if (emp_rows["ci_low"] > emp_rows["point"]).any() or (
            emp_rows["point"] > emp_rows["ci_high"]
        ).any():
            raise ValueError(f"{source}: EMP point estimate lies outside its interval")
        tables[dataset_key] = emp_rows.set_index("model").loc[list(MODELS)]
    return tables


def load_emp_roi_grid(posthoc_input: Path, fixed_split: Path) -> pd.DataFrame:
    if not posthoc_input.is_file():
        raise FileNotFoundError(f"Missing EMP ROI-grid input: {posthoc_input}")
    frame = pd.read_csv(posthoc_input)
    _require_columns(frame, GRID_REQUIRED_COLUMNS, posthoc_input)
    expected_datasets = {key for key, _label in DATASETS}
    if set(frame["dataset"]) != expected_datasets:
        raise ValueError(
            f"{posthoc_input}: expected datasets {sorted(expected_datasets)}, "
            f"found {sorted(set(frame['dataset']))}"
        )
    if frame.duplicated(["dataset", "model", "roi"]).any():
        raise ValueError(f"{posthoc_input}: duplicate dataset/model/ROI rows")
    _require_finite_fraction(frame["emp"], posthoc_input, "emp")

    for dataset_key, dataset_label in DATASETS:
        subset = frame.loc[frame["dataset"].eq(dataset_key)]
        if set(subset["model"]) != set(MODELS):
            raise ValueError(
                f"{posthoc_input}: {dataset_label} does not contain all four models"
            )
        roi_sets = {
            tuple(sorted(subset.loc[subset["model"].eq(model), "roi"].astype(float)))
            for model in MODELS
        }
        if len(roi_sets) != 1 or len(next(iter(roi_sets))) != 9:
            raise ValueError(
                f"{posthoc_input}: {dataset_label} must contain the same nine ROI values "
                "for every model"
            )

        config_path = fixed_split / dataset_key / "economic_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen economic configuration for {dataset_label}: {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        baseline_roi = float(config["roi"])
        recorded = subset["baseline_roi"].astype(float)
        if not recorded.map(
            lambda value: math.isclose(value, baseline_roi, rel_tol=0.0, abs_tol=1e-12)
        ).all():
            raise ValueError(
                f"{posthoc_input}: baseline ROI disagrees with {config_path}"
            )
    return frame


def _style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def make_interval_figure(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for axis, (dataset_key, dataset_label) in zip(axes, DATASETS):
        table = tables[dataset_key]
        for index, model in enumerate(MODELS):
            emp_raw = float(table.loc[model, "point"])
            emp_low_raw = float(table.loc[model, "ci_low"])
            emp_high_raw = float(table.loc[model, "ci_high"])
            emp_percent = emp_to_percent(emp_raw)
            emp_low_percent = emp_to_percent(emp_low_raw)
            emp_high_percent = emp_to_percent(emp_high_raw)
            axis.errorbar(
                index,
                emp_percent,
                yerr=[
                    [emp_percent - emp_low_percent],
                    [emp_high_percent - emp_percent],
                ],
                fmt="o",
                ms=10,
                capsize=5,
                lw=2,
                color=COLORS[model],
            )
        axis.set_xticks(range(4), SHORT_MODELS, fontsize=11)
        axis.set_title(dataset_label, fontsize=13)
        axis.grid(axis="y", alpha=0.3)
        axis.margins(x=0.15)
    axes[0].set_ylabel("EMP (% of principal)\n95% bootstrap interval")
    fig.tight_layout()
    return fig


def make_sensitivity_figure(grid: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for axis, (dataset_key, dataset_label) in zip(axes.ravel(), DATASETS):
        dataset_rows = grid.loc[grid["dataset"].eq(dataset_key)]
        for model in MODELS:
            model_rows = dataset_rows.loc[dataset_rows["model"].eq(model)].sort_values(
                "roi"
            )
            emp_raw = model_rows["emp"].astype(float)
            emp_percent = emp_raw.map(emp_to_percent)
            axis.plot(
                model_rows["roi"],
                emp_percent,
                "o-",
                color=COLORS[model],
                lw=2,
                ms=5,
                label=model,
            )
        baseline_roi = float(dataset_rows["baseline_roi"].iloc[0])
        axis.axvline(baseline_roi, color="grey", ls="--", lw=1.5)
        axis.annotate(
            f"baseline\nROI = {baseline_roi:.4g}",
            xy=(baseline_roi, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=10,
            color="grey",
        )
        axis.set_title(dataset_label.replace("\n", " "), fontsize=13)
        axis.set_ylabel("EMP (% of principal)")
        axis.grid(alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("ROI (return on investment)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    return fig


def save_figure(figure: plt.Figure, output_dir: Path, name: str, write_png: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{name}.pdf"
    fixed_timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)
    figure.savefig(
        pdf_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": name,
            "Author": "Marius Calenborn",
            "Creator": "scripts/plot_emp_figures.py",
            "CreationDate": fixed_timestamp,
            "ModDate": fixed_timestamp,
        },
    )
    print(f"Wrote vector PDF: {pdf_path}")
    if write_png:
        png_path = output_dir / f"{name}.png"
        figure.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Wrote PNG preview: {png_path}")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    _style()
    interval_tables = load_emp_intervals(args.fixed_split)
    roi_grid = load_emp_roi_grid(args.posthoc_input, args.fixed_split)
    save_figure(
        make_interval_figure(interval_tables), args.output_dir, "fig_emp", args.png
    )
    save_figure(
        make_sensitivity_figure(roi_grid),
        args.output_dir,
        "fig_emp_sensitivity",
        args.png,
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

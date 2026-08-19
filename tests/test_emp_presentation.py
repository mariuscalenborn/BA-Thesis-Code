import csv
from pathlib import Path

import pytest

from scripts.plot_emp_figures import emp_to_percent
from scripts.plot_pairwise_forest import delta_emp_to_percentage_points


RESULTS = Path(__file__).resolve().parents[1] / "results" / "fixed_split_v4"
DATASETS = ("south_german", "taiwan", "home_credit", "lending_club")


def test_emp_levels_are_converted_to_percent_of_principal():
    assert emp_to_percent(0.0478) == pytest.approx(4.78)
    assert emp_to_percent(0.0248) == pytest.approx(2.48)
    assert emp_to_percent(0.0016) == pytest.approx(0.16)


def test_emp_differences_are_converted_to_percentage_points():
    assert delta_emp_to_percentage_points(0.00088) == pytest.approx(0.088)
    assert delta_emp_to_percentage_points(-0.00376) == pytest.approx(-0.376)
    assert delta_emp_to_percentage_points(-0.00047) == pytest.approx(-0.047)
    assert delta_emp_to_percentage_points(0.00050) == pytest.approx(0.05)


def test_conversion_preserves_sign_and_zero_inclusion():
    old_interval_raw = (-0.00495, 0.00464)
    new_interval_pp = tuple(
        delta_emp_to_percentage_points(value) for value in old_interval_raw
    )
    assert new_interval_pp == pytest.approx((-0.495, 0.464))
    assert new_interval_pp[0] < 0 < new_interval_pp[1]

    negative_interval_raw = (-0.00054, -0.00040)
    negative_interval_pp = tuple(
        delta_emp_to_percentage_points(value) for value in negative_interval_raw
    )
    assert negative_interval_pp == pytest.approx((-0.054, -0.04))
    assert negative_interval_pp[1] < 0


def test_canonical_paired_emp_interpretation_is_preserved():
    for dataset in DATASETS:
        source = RESULTS / dataset / "auc_delta_bootstrap.csv"
        with source.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["metric"] == "emp"]
        assert len(rows) == 6
        for row in rows:
            delta_emp_raw = float(row["delta_point"])
            low_raw = float(row["delta_ci_low"])
            high_raw = float(row["delta_ci_high"])
            delta_emp_percentage_points = delta_emp_to_percentage_points(delta_emp_raw)
            low_percentage_points = delta_emp_to_percentage_points(low_raw)
            high_percentage_points = delta_emp_to_percentage_points(high_raw)

            assert (delta_emp_raw > 0) == (delta_emp_percentage_points > 0)
            assert (delta_emp_raw < 0) == (delta_emp_percentage_points < 0)
            assert (low_raw <= 0 <= high_raw) == (
                low_percentage_points <= 0 <= high_percentage_points
            )

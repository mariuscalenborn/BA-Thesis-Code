"""Audit Lending Club outcome maturity and pre-pricing feature eligibility."""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.datasets import LENDING_CLUB_ENDOGENOUS_DROPS, TARGET_COLUMN, get
from src.run_utils import write_csv_atomic, write_json_atomic


LC_CONFIG = get("lending_club")
PROCESSED = LC_CONFIG.semiraw_path
OUTDIR = Path(os.environ.get("OUTDIR", "results/data_audit_v4"))
CHUNK_SIZE = int(os.environ.get("LC_AUDIT_CHUNK_SIZE", "250000"))
RETAINED_STATUSES = {"Fully Paid", "Charged Off", "Default"}

REQUIRED_MODEL_DROPS = {
    "target",
    *LENDING_CLUB_ENDOGENOUS_DROPS,
    *LC_CONFIG.aux_cols,
}

EXCLUDED_FEATURES = {
    "target": (
        "outcome",
        "Observed repayment outcome; prediction target, never a predictor.",
    ),
    "source_row_id": (
        "identifier",
        "Stable join key from the export order; carries no information about the applicant.",
    ),
    "issue_year": (
        "split_only",
        "Origination year defines temporal partitions and is excluded from prediction.",
    ),
    "grade": (
        "decision_or_pricing",
        "Lending Club underwriting grade is produced by the credit decision process.",
    ),
    "sub_grade": (
        "decision_or_pricing",
        "Lending Club underwriting sub-grade is produced by the decision process.",
    ),
    "int_rate": (
        "post_pricing",
        "Contract interest rate is the pricing outcome; retained only for ROI context.",
    ),
    "installment": (
        "post_pricing",
        "Contract payment is determined from funded amount, rate, and term.",
    ),
    "funded_amnt": (
        "decision_or_funding",
        "Actually funded amount is an approval/funding outcome; loan_amnt remains the request.",
    ),
}

APPLICANT_CHOICE = {"loan_amnt", "term", "purpose"}
APPLICANT_REPORTED = {"emp_length", "annual_inc", "dti", "home_ownership"}
PROCESS_METADATA = {"initial_list_status", "disbursement_method", "verification_status"}


def _find_raw_path():
    configured = os.environ.get("LC_RAW_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LC_RAW_PATH does not exist: {path}")
        return path

    candidates = [
        Path("data/raw/lending_club/archive/accepted_2007_to_2018Q4.csv"),
        Path("data/raw/lending_club/archive/accepted_2007_to_2018Q4.csv.gz"),
        Path("data/raw/lending_club/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    discovered = sorted(Path("data/raw").rglob("accepted_2007_to_2018Q4.csv*"))
    if discovered:
        return discovered[0]
    raise FileNotFoundError(
        "Accepted Lending Club source missing. Set LC_RAW_PATH to the accepted-loans "
        "CSV or .csv.gz before running the audit."
    )


def _parse_month(values):
    return pd.to_datetime(values, format="%b-%Y", errors="coerce")


def _snapshot():
    value = os.environ.get("LC_OUTCOME_SNAPSHOT", "").strip()
    if not value:
        return None
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
    return pd.Timestamp(parsed)


def _verify_model_drop_lists():
    """Check the two mechanisms that keep pricing information out of the model."""
    columns = pd.read_csv(PROCESSED, nrows=0).columns.tolist()
    features = set(LC_CONFIG.feature_columns(columns))

    still_present = sorted(
        column for column in LENDING_CLUB_ENDOGENOUS_DROPS if column in columns
    )
    if still_present:
        raise RuntimeError(
            f"Endogenous pricing columns survived the export: {still_present}"
        )

    absent_aux = sorted(column for column in LC_CONFIG.aux_cols if column not in columns)
    if absent_aux:
        raise RuntimeError(f"Declared auxiliary columns are missing: {absent_aux}")

    leaked = sorted(REQUIRED_MODEL_DROPS & features)
    if leaked:
        raise RuntimeError(f"Excluded columns reached the feature matrix: {leaked}")

    return [{
        "contract": "src/datasets.py::DATASETS['lending_club']",
        "endogenous_dropped_at_export": sorted(LENDING_CLUB_ENDOGENOUS_DROPS),
        "auxiliary_present_but_excluded": sorted(LC_CONFIG.aux_cols),
        "target_column": TARGET_COLUMN,
        "n_model_features": len(features),
        "required_exclusions_present": True,
    }]


def _feature_stage(feature):
    if feature in EXCLUDED_FEATURES:
        stage, rationale = EXCLUDED_FEATURES[feature]
        return "excluded", stage, rationale, "explicit_protocol_rule"

    base = feature
    if base in APPLICANT_CHOICE:
        return (
            "included",
            "application_borrower_choice",
            "Borrower request or selected loan characteristic available before pricing.",
            "field_semantics; source-timing documentation still required",
        )
    if base in APPLICANT_REPORTED or base == "addr_state":
        return (
            "included",
            "application_reported",
            "Applicant-reported information available during the application.",
            "field_semantics; source-timing documentation still required",
        )
    if base in PROCESS_METADATA:
        return (
            "included_with_timing_caveat",
            "application_or_origination_process",
            "Process metadata retained by the frozen plan; exact pre-pricing availability must be documented.",
            "timing_not_independently_verified",
        )
    return (
        "included",
        "application_credit_file",
        "Credit-file or application characteristic treated as available at scoring time.",
        "field_semantics; source-timing documentation still required",
    )


def _feature_eligibility(processed_columns):
    rows = []
    for feature in processed_columns:
        status, stage, rationale, evidence = _feature_stage(feature)
        rows.append({
            "feature": feature,
            "model_status": status,
            "availability_stage": stage,
            "rationale": rationale,
            "timing_evidence": evidence,
        })
    return pd.DataFrame(rows)


def _scan_raw(raw_path, snapshot):
    header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    required = {"issue_d", "term", "loan_status"}
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"Raw LC source is missing audit columns: {missing}")
    usecols = ["issue_d", "term", "loan_status"]
    if "last_credit_pull_d" in header:
        usecols.append("last_credit_pull_d")

    status_parts = []
    cohort_parts = []
    rows_scanned = 0
    invalid_issue_or_term = 0
    observed_last_pull = None

    for chunk_number, chunk in enumerate(
        pd.read_csv(raw_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        rows_scanned += len(chunk)
        issue_date = _parse_month(chunk["issue_d"])
        term = pd.to_numeric(
            chunk["term"].astype("string").str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        valid = issue_date.notna() & term.notna()
        invalid_issue_or_term += int((~valid).sum())
        work = pd.DataFrame({
            "issue_year": issue_date[valid].dt.year.astype(int),
            "issue_date": issue_date[valid],
            "term_months": term[valid].astype(int),
            "loan_status": chunk.loc[valid, "loan_status"].fillna("<missing>").astype(str),
        })
        work["retained_by_preprocessing"] = work["loan_status"].isin(RETAINED_STATUSES)
        work["terminal_status"] = (
            work["loan_status"].str.contains("Fully Paid|Charged Off", regex=True)
            | work["loan_status"].eq("Default")
        )
        work["nonterminal_status"] = ~work["terminal_status"]
        work["terminal_but_excluded"] = (
            work["terminal_status"] & ~work["retained_by_preprocessing"]
        )

        if snapshot is not None:
            observation_months = (
                (snapshot.year - work["issue_date"].dt.year) * 12
                + snapshot.month
                - work["issue_date"].dt.month
            )
            work["observation_months"] = observation_months
            work["contractually_mature"] = observation_months >= work["term_months"]

        status_parts.append(
            work.groupby(["issue_year", "term_months", "loan_status"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
        aggregation = {
            "all_originated_proxy_n": ("loan_status", "size"),
            "terminal_n": ("terminal_status", "sum"),
            "nonterminal_n": ("nonterminal_status", "sum"),
            "resolved_target_retained_n": ("retained_by_preprocessing", "sum"),
            "terminal_but_excluded_n": ("terminal_but_excluded", "sum"),
        }
        if snapshot is not None:
            aggregation.update({
                "contractually_mature_n": ("contractually_mature", "sum"),
                "observation_months_min": ("observation_months", "min"),
                "observation_months_max": ("observation_months", "max"),
            })
        cohort_parts.append(
            work.groupby(["issue_year", "term_months"], as_index=False).agg(**aggregation)
        )

        if "last_credit_pull_d" in chunk:
            chunk_max = _parse_month(chunk["last_credit_pull_d"]).max()
            if pd.notna(chunk_max) and (
                observed_last_pull is None or chunk_max > observed_last_pull
            ):
                observed_last_pull = chunk_max
        print(f"  raw chunk {chunk_number}: rows_scanned={rows_scanned:,}")

    status = (
        pd.concat(status_parts, ignore_index=True)
        .groupby(["issue_year", "term_months", "loan_status"], as_index=False)["n"]
        .sum()
        .sort_values(["issue_year", "term_months", "loan_status"])
    )
    status["share_within_issue_year_term"] = (
        status["n"]
        / status.groupby(["issue_year", "term_months"])["n"].transform("sum")
    )

    sum_columns = [
        "all_originated_proxy_n",
        "terminal_n",
        "nonterminal_n",
        "resolved_target_retained_n",
        "terminal_but_excluded_n",
    ]
    if snapshot is not None:
        sum_columns.append("contractually_mature_n")
    cohort_aggregation = {column: "sum" for column in sum_columns}
    if snapshot is not None:
        cohort_aggregation.update({
            "observation_months_min": "min",
            "observation_months_max": "max",
        })
    cohort = (
        pd.concat(cohort_parts, ignore_index=True)
        .groupby(["issue_year", "term_months"], as_index=False)
        .agg(cohort_aggregation)
        .sort_values(["issue_year", "term_months"])
    )
    denominator = cohort["all_originated_proxy_n"].replace(0, np.nan)
    cohort["terminal_share"] = cohort["terminal_n"] / denominator
    cohort["nonterminal_share"] = cohort["nonterminal_n"] / denominator
    cohort["resolved_target_retained_share"] = (
        cohort["resolved_target_retained_n"] / denominator
    )
    cohort["terminal_but_excluded_share"] = (
        cohort["terminal_but_excluded_n"] / denominator
    )
    if snapshot is not None:
        cohort["contractually_mature_share"] = cohort["contractually_mature_n"] / denominator
        cohort["contractual_maturity_status"] = "computed_from_user_supplied_snapshot"
    else:
        cohort["contractually_mature_n"] = np.nan
        cohort["contractually_mature_share"] = np.nan
        cohort["observation_months_min"] = np.nan
        cohort["observation_months_max"] = np.nan
        cohort["contractual_maturity_status"] = "not_computable_without_documented_snapshot"

    return status, cohort, {
        "rows_scanned": int(rows_scanned),
        "invalid_issue_or_term_rows": int(invalid_issue_or_term),
        "observed_max_last_credit_pull_d": (
            observed_last_pull.strftime("%Y-%m-%d") if observed_last_pull is not None else None
        ),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not PROCESSED.exists():
        raise FileNotFoundError(f"Processed Lending Club data missing: {PROCESSED}")
    raw_path = _find_raw_path()
    snapshot = _snapshot()
    print(f"LC maturity audit: raw={raw_path} snapshot={snapshot or 'NOT DOCUMENTED'}")

    drop_checks = _verify_model_drop_lists()
    processed_columns = pd.read_csv(PROCESSED, nrows=0).columns.tolist()
    feature_table = _feature_eligibility(processed_columns)
    included = set(feature_table.loc[
        feature_table["model_status"].str.startswith("included"), "feature"
    ])
    prohibited_included = sorted(REQUIRED_MODEL_DROPS & included)
    if prohibited_included:
        raise RuntimeError(f"Prohibited LC fields marked as model features: {prohibited_included}")

    status, cohort, scan = _scan_raw(raw_path, snapshot)
    processed_rows = 0
    for chunk in pd.read_csv(PROCESSED, usecols=["target"], chunksize=CHUNK_SIZE):
        processed_rows += len(chunk)
    retained_rows = int(cohort["resolved_target_retained_n"].sum())

    write_csv_atomic(OUTDIR / "loan_status_by_issue_year_term.csv", status)
    write_csv_atomic(OUTDIR / "cohort_maturity_retention.csv", cohort)
    write_csv_atomic(OUTDIR / "feature_timing_eligibility.csv", feature_table)
    summary = {
        "raw_source": str(raw_path.resolve()),
        "raw_source_size_bytes": raw_path.stat().st_size,
        "processed_source": str(PROCESSED.resolve()),
        "processed_rows": int(processed_rows),
        "raw_rows_scanned": scan["rows_scanned"],
        "raw_rows_with_invalid_issue_or_term": scan["invalid_issue_or_term_rows"],
        "all_originated_proxy_n": int(cohort["all_originated_proxy_n"].sum()),
        "resolved_target_retained_n": retained_rows,
        "resolved_target_retained_share": float(
            retained_rows / cohort["all_originated_proxy_n"].sum()
        ),
        "processed_matches_raw_retained_count": processed_rows == retained_rows,
        "retained_status_definition": sorted(RETAINED_STATUSES),
        "terminal_status_definition": (
            "loan_status contains Fully Paid or Charged Off, or equals Default"
        ),
        "denominator_definition": (
            "accepted-loans records with parseable issue_d and term; used as the "
            "originated-loan proxy before the resolved-only target filter"
        ),
        "outcome_snapshot": snapshot.strftime("%Y-%m-%d") if snapshot is not None else None,
        "outcome_snapshot_source": (
            "LC_OUTCOME_SNAPSHOT supplied from external documentation"
            if snapshot is not None
            else "unavailable; contractual maturity intentionally not computed"
        ),
        "observed_max_last_credit_pull_d": scan["observed_max_last_credit_pull_d"],
        "last_credit_pull_used_as_snapshot": False,
        "model_drop_checks": drop_checks,
        "feature_timing_caveat": (
            "Rows marked included_with_timing_caveat require independent source "
            "documentation before claiming strict pre-pricing availability."
        ),
    }
    write_json_atomic(OUTDIR / "audit_summary.json", summary)
    print(f"LC audit complete: {OUTDIR}")
    if processed_rows != retained_rows:
        print(
            "WARNING: processed row count differs from retained raw status count; "
            "verify that raw and processed files are the same dataset release."
        )


if __name__ == "__main__":
    main()

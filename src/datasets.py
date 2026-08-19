"""Declarative dataset contracts: join key, auxiliary columns and predictor roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ID_COLUMN = "source_row_id"
TARGET_COLUMN = "target"
MISSING_THRESHOLD = 0.40
WINSOR_QUANTILE = 0.99


@dataclass(frozen=True)
class DatasetConfig:
    """Everything the pipeline needs to know about one dataset."""

    name: str
    display_name: str
    semiraw_path: Path
    roi: float | str
    aux_cols: tuple[str, ...] = ()
    categorical_cols: tuple[str, ...] = ()
    binary_numeric_cols: tuple[str, ...] = ()
    winsor_cols: tuple[str, ...] = ()
    protected_cols: tuple[str, ...] = ()
    fairness_col: str | None = None
    explain_full_test: bool = False
    sigmoid_sensitivity: bool = False
    missing_threshold: float = MISSING_THRESHOLD
    notes: str = ""
    _expected_string_cols: tuple[str, ...] = field(default=(), repr=False)

    def feature_columns(self, columns) -> list[str]:
        """Predictor columns in file order: everything except target, ID and aux."""
        excluded = {TARGET_COLUMN, ID_COLUMN, *self.aux_cols}
        return [column for column in columns if column not in excluded]

    def numeric_columns(self, columns) -> list[str]:
        """Continuous/ordinal predictors: features minus categorical and binary."""
        special = {*self.categorical_cols, *self.binary_numeric_cols}
        return [column for column in self.feature_columns(columns) if column not in special]


DATASETS: dict[str, DatasetConfig] = {
    "south_german": DatasetConfig(
        name="south_german",
        display_name="German Credit (Statlog)",
        semiraw_path=Path("data/processed/v4/south_german_credit.csv"),
        roi=0.2644,
        categorical_cols=(
            "purpose",
            "personal_status_sex",
            "other_debtors",
            "other_installment",
            "housing",
        ),
        explain_full_test=True,
        sigmoid_sensitivity=True,
        notes=(
            "Statlog german.data. Ordinal maps (checking_account, credit_history, "
            "savings_account, employment_since, property, job) and the two binary "
            "recodes are documented literal maps applied at export. No missing values."
        ),
        _expected_string_cols=(
            "purpose",
            "personal_status_sex",
            "other_debtors",
            "other_installment",
            "housing",
        ),
    ),
    "taiwan": DatasetConfig(
        name="taiwan",
        display_name="Taiwan Credit Card Default",
        semiraw_path=Path("data/processed/v4/taiwan_credit.csv"),
        roi=0.2644,
        categorical_cols=("EDUCATION", "MARRIAGE"),
        protected_cols=("SEX", "AGE", "MARRIAGE"),
        fairness_col="SEX",
        notes=(
            "EDUCATION and MARRIAGE are nominal and one-hot encoded at runtime; "
            "the export keeps their documented integer levels (undefined codes "
            "repaired in notebook 02). PAY_* stay ordinal. No missing values."
        ),
    ),
    "home_credit": DatasetConfig(
        name="home_credit",
        display_name="Home Credit Default Risk",
        semiraw_path=Path("data/processed/v4/home_credit.csv"),
        roi=0.2644,
        categorical_cols=(
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START",
            "ORGANIZATION_TYPE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "WALLSMATERIAL_MODE",
            "EMERGENCYSTATE_MODE",
        ),
        winsor_cols=("AMT_INCOME_TOTAL",),
        fairness_col="CODE_GENDER",
        notes=(
            "All 16 string columns keep their raw levels; XNA is repaired to missing "
            "at export and becomes the explicit Unknown level inside the pipeline. "
            "DAYS_EMPLOYED == 365243 is repaired to missing with a companion flag."
        ),
        _expected_string_cols=(
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
        ),
    ),
    "lending_club": DatasetConfig(
        name="lending_club",
        display_name="Lending Club",
        semiraw_path=Path("data/processed/v4/lending_club_full.csv"),
        roi="int_rate_train_mean",
        aux_cols=("int_rate", "issue_year"),
        categorical_cols=(
            "home_ownership",
            "verification_status",
            "purpose",
            "addr_state",
            "initial_list_status",
            "application_type",
            "disbursement_method",
        ),
        winsor_cols=("annual_inc", "dti", "revol_bal"),
        notes=(
            "grade, sub_grade, installment and funded_amnt are endogenous to Lending "
            "Club's own pricing decision and are dropped at export. int_rate supplies "
            "the ROI parameter and issue_year the out-of-time cutoff; both are "
            "auxiliary, never predictors."
        ),
        _expected_string_cols=(
            "home_ownership",
            "verification_status",
            "purpose",
            "addr_state",
        ),
    ),
}


LENDING_CLUB_ENDOGENOUS_DROPS = (
    "grade",
    "sub_grade",
    "installment",
    "funded_amnt",
)


def get(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


def validate_frame(config: DatasetConfig, frame) -> None:
    """Fail loudly if a semi-raw CSV does not match its declared contract."""
    columns = list(frame.columns)
    missing = [
        column
        for column in (ID_COLUMN, TARGET_COLUMN, *config.aux_cols, *config.categorical_cols)
        if column not in columns
    ]
    if missing:
        raise ValueError(f"{config.name}: semi-raw CSV lacks columns {missing}")

    ids = frame[ID_COLUMN]
    if ids.duplicated().any():
        raise ValueError(f"{config.name}: {ID_COLUMN} is not unique")
    if ids.isnull().any():
        raise ValueError(f"{config.name}: {ID_COLUMN} contains missing values")

    if frame[TARGET_COLUMN].isnull().any():
        raise ValueError(f"{config.name}: target contains missing values")
    labels = set(frame[TARGET_COLUMN].dropna().unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{config.name}: target is not binary, found {sorted(labels)}")

    import pandas as pd

    for column in config._expected_string_cols:
        if pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(
                f"{config.name}: {column} is numeric in the semi-raw CSV; the export "
                "must not encode categories (that is the pipeline's job)"
            )
    for column in config.winsor_cols:
        if column not in columns:
            raise ValueError(f"{config.name}: winsorization column {column} is absent")
    for column in config.protected_cols:
        if column not in columns:
            raise ValueError(f"{config.name}: protected column {column} is absent")
    if config.fairness_col is not None and config.fairness_col not in columns:
        raise ValueError(f"{config.name}: fairness column {config.fairness_col} is absent")

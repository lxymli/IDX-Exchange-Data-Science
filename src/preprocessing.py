"""Cleaning, encoding, and time-based splitting for the CRMLS single-family dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, TargetEncoder

TARGET = "ClosePrice"

RAW_COLS = [
    "ListingKey", "PropertyType", "PropertySubType", "StateOrProvince", "CloseDate", "ClosePrice",
    "ListPrice", "OriginalListPrice", "LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeSquareFeet", "YearBuilt", "Latitude", "Longitude", "GarageSpaces", "ParkingTotal",
    "Stories", "MainLevelBedrooms", "AssociationFee", "AssociationFeeFrequency", "Levels", "Flooring",
    "ViewYN", "PoolPrivateYN", "FireplaceYN", "NewConstructionYN", "AttachedGarageYN", "BasementYN",
    "WaterfrontYN", "CountyOrParish", "City", "PostalCode",
]

# Plausibility bounds; values outside become NaN (features) or drop the row (target).
PRICE_BOUNDS = (50_000, 30_000_000)
FEATURE_BOUNDS = {
    "LivingArea": (300, 20_000),
    "BedroomsTotal": (1, 15),
    "BathroomsTotalInteger": (1, 15),
    "LotSizeSquareFeet": (500, 100 * 43_560),
    "YearBuilt": (1850, 2027),
    "GarageSpaces": (0, 20),
    "ParkingTotal": (0, 30),
    "MainLevelBedrooms": (0, 15),
    "AssociationFee": (0, 20_000),
}
CA_LAT, CA_LON = (32.0, 42.5), (-124.6, -114.0)

BINARY_COLS = ["ViewYN", "PoolPrivateYN", "FireplaceYN", "NewConstructionYN", "AttachedGarageYN"]
# Only "True" or missing in the source, so missing means "No".
TRUE_OR_MISSING_COLS = ["BasementYN", "WaterfrontYN"]
LEVELS = ["One", "Two", "ThreeOrMore", "MultiSplit"]
FLOORING = ["Carpet", "Tile", "Wood", "Laminate", "Vinyl", "Stone"]
TOP_COUNTIES = 12
FEE_PER_MONTH = {"Monthly": 1, "Quarterly": 1 / 3, "SemiAnnually": 1 / 6, "Annually": 1 / 12}

# Imputed (median, fit on the training window) inside the model pipeline.
NUMERIC_FEATURES = [
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet", "YearBuilt",
    "Latitude", "Longitude", "GarageSpaces", "ParkingTotal", "Stories", "MainLevelBedrooms",
    "AssociationFeeMonthly", "close_month_index", "close_month",
]
# Long right tails: log-transformed before scaling in the linear model.
SKEWED_FEATURES = ["LivingArea", "LotSizeSquareFeet", "AssociationFeeMonthly", "ParkingTotal"]
# High-cardinality location codes: target-encoded, fit on the training window only.
TARGET_ENCODED = ["PostalCode", "City"]
# Columns not used as features (identifiers, list prices, which would leak the answer).
NON_FEATURES = ["ListingKey", "CloseDate", "close_period", TARGET, "ListPrice", "OriginalListPrice"]


def load_raw(data_dir="Data"):
    files = sorted(Path(data_dir).glob("CRMLSSold*.csv"))
    frames = [pd.read_csv(f, usecols=lambda c: c in RAW_COLS, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True)


def _to_bool(s):
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})


def clean(raw):
    """Filter to valid single-family sales and encode fields. Returns (cleaned_df, cleaning_log)."""
    log = {"raw rows": len(raw)}
    df = raw[(raw["PropertyType"] == "Residential") & (raw["PropertySubType"] == "SingleFamilyResidence")].copy()
    log["single-family residential"] = len(df)

    df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df = df.sort_values("CloseDate").drop_duplicates("ListingKey", keep="last")
    log["after dropping duplicate listings"] = len(df)

    df = df[df["StateOrProvince"].eq("CA") & df["CloseDate"].notna()]
    log["after keeping CA with a close date"] = len(df)

    df = df[df[TARGET].between(*PRICE_BOUNDS)]
    log[f"after ClosePrice in ${PRICE_BOUNDS[0]:,}-${PRICE_BOUNDS[1]:,}"] = len(df)

    # Implausible feature values -> NaN, so they get imputed rather than dropping the row.
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        df[col] = df[col].where(df[col].between(lo, hi))
    df["Stories"] = df["Stories"].where(df["Stories"].isin([1, 2, 3]))
    in_ca = df["Latitude"].between(*CA_LAT) & df["Longitude"].between(*CA_LON)
    df.loc[~in_ca, ["Latitude", "Longitude"]] = np.nan

    # Living area is the strongest predictor and is almost never missing, so require it.
    df = df[df["LivingArea"].notna()]
    log["after requiring valid LivingArea"] = len(df)

    out = df[["ListingKey", "CloseDate", TARGET, "ListPrice", "OriginalListPrice"]].copy()

    # Time
    out["close_period"] = df["CloseDate"].dt.to_period("M").astype(str)
    out["close_month"] = df["CloseDate"].dt.month
    first = pd.Period("2022-01", "M")
    out["close_month_index"] = df["CloseDate"].dt.to_period("M").map(lambda p: (p - first).n)

    # Numeric (NaNs kept; imputed in the model pipeline). HOA fee normalised to a monthly amount.
    for col in ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet", "YearBuilt",
                "Latitude", "Longitude", "GarageSpaces", "ParkingTotal", "Stories", "MainLevelBedrooms"]:
        out[col] = df[col]
    out["AssociationFeeMonthly"] = df["AssociationFee"] * df["AssociationFeeFrequency"].map(FEE_PER_MONTH).fillna(1)
    out["has_hoa"] = (out["AssociationFeeMonthly"] > 0).astype(int)

    # Missing-value flags for features with meaningful missingness.
    for col in ["LotSizeSquareFeet", "YearBuilt", "GarageSpaces", "ParkingTotal", "Stories",
                "MainLevelBedrooms", "AssociationFeeMonthly", "Latitude"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)

    # Binary Y/N -> 0/1; missing -> 0 plus a flag.
    for col in BINARY_COLS:
        b = _to_bool(df[col])
        out[f"{col}_missing"] = b.isna().astype(int)
        out[col] = b.fillna(0).astype(int)
    for col in TRUE_OR_MISSING_COLS:
        out[col] = _to_bool(df[col]).fillna(0).astype(int)

    # Low-cardinality categoricals -> one-hot.
    levels = df["Levels"].where(df["Levels"].isin(LEVELS), np.where(df["Levels"].isna(), "missing", "Other"))
    out = out.join(pd.get_dummies(levels, prefix="Levels", dtype=int))
    counties = df["CountyOrParish"].value_counts().head(TOP_COUNTIES).index
    county = df["CountyOrParish"].where(df["CountyOrParish"].isin(counties), "Other")
    out = out.join(pd.get_dummies(county.str.replace(" ", "_"), prefix="County", dtype=int))

    # Flooring is a comma-separated multi-value field -> one indicator per material.
    flooring = df["Flooring"].fillna("")
    for m in FLOORING:
        out[f"Flooring_{m}"] = flooring.str.contains(m, case=False).astype(int)
    out["Flooring_missing"] = df["Flooring"].isna().astype(int)

    # High-cardinality location codes kept as strings; target-encoded inside the pipeline.
    out["PostalCode"] = df["PostalCode"].astype(str).str.extract(r"(\d{5})")[0].fillna("missing")
    out["City"] = df["City"].fillna("missing").str.strip().str.title()

    log["final cleaned rows"] = len(out)
    return out.reset_index(drop=True), pd.Series(log, name="rows")


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def time_split(df, test_period, train_months):
    """Test = rows closing in `test_period`; train = the `train_months` months immediately before it."""
    test_idx = (pd.Period(test_period, "M") - pd.Period("2022-01", "M")).n
    idx = df["close_month_index"]
    return df[idx.between(test_idx - train_months, test_idx - 1)], df[idx == test_idx]


def make_model(kind="hgb", columns=None):
    """Pipeline: impute -> (log + scale for linear models) -> target-encode locations -> regressor on log price."""
    columns = list(columns)
    numeric = [c for c in NUMERIC_FEATURES if c in columns]
    passthrough = [c for c in columns if c not in numeric + TARGET_ENCODED]

    if kind == "ridge":
        skewed = [c for c in numeric if c in SKEWED_FEATURES]
        other = [c for c in numeric if c not in skewed]
        numeric_steps = [
            ("skewed", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p),
                                     StandardScaler()), skewed),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other),
        ]
        regressor = Ridge(alpha=1.0)
    else:
        # Tree models are insensitive to monotonic scaling, so no normalisation is applied.
        numeric_steps = [("numeric", SimpleImputer(strategy="median"), numeric)]
        regressor = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=0)

    encoder = make_pipeline(TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)), StandardScaler())
    pre = ColumnTransformer(
        numeric_steps + [("location", encoder, TARGET_ENCODED), ("flags", "passthrough", passthrough)],
        verbose_feature_names_out=False,
    )
    return TransformedTargetRegressor(regressor=make_pipeline(pre, regressor), func=np.log, inverse_func=np.exp)


def evaluate(y_true, y_pred):
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "MAPE": ape.mean() * 100,
        "MdAPE": np.median(ape) * 100,
        "within_10pct": (ape <= 0.10).mean() * 100,
        "R2": 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum(),
    }

"""
predict_pycharm_ready.py

PyCharm-ready prediction script for your t+24h Bicing model.

What this version adds
----------------------
1) It can still be used from the command line with arguments
2) It can also be run directly inside PyCharm with default test values
3) It prints a clean result block
4) It uses your current total/share model logic:
   - predict total bikes
   - predict ebike share
   - reconstruct ebikes and mechanical

Important
---------
This script assumes:
- your ETL already created bicing_selected_features.parquet
- your training script already created:
    bike_predictor_bundle_t_plus_24h.pkl

Because your model is trained for t+24h, to predict a target datetime,
the script looks for the feature row around target_datetime - 24h.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd


# =========================================================
# PATHS / CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_FEATURES_PATH = PROJECT_ROOT / "bicing_selected_features.parquet"
DEFAULT_BUNDLE_PATH = (
    PROJECT_ROOT
    / "model"
    / "models_bicing_planning_xgb_total_share_t24h_from_v10"
    / "t_plus_24h"
    / "bike_predictor_bundle_t_plus_24h.pkl"
)

LOCAL_TZ = ZoneInfo("Europe/Madrid")
UTC_TZ = ZoneInfo("UTC")

SNAPSHOT_FREQUENCY_MIN = 5
HORIZON_STEPS = 288
HORIZON_MINUTES = SNAPSHOT_FREQUENCY_MIN * HORIZON_STEPS
DEFAULT_MAX_LOOKUP_MINUTES = 30

# PyCharm fallback values when no CLI args are passed
PYCHARM_DEFAULTS = {
    "station_id": 369,
    "target_date": "2026-04-10",
    "target_time": "08:00",
    "bundle_path": str(DEFAULT_BUNDLE_PATH),
    "features_path": str(DEFAULT_FEATURES_PATH),
    "max_lookup_minutes": DEFAULT_MAX_LOOKUP_MINUTES,
}


# =========================================================
# RESULT OBJECT
# =========================================================
@dataclass
class PredictionResult:
    station_id: int
    target_local: str
    target_utc: str
    anchor_utc: str
    matched_snapshot_utc: str
    matched_delay_minutes: float
    capacity: int
    predicted_total_bikes: int
    predicted_mechanical: int
    predicted_ebike: int
    predicted_ebike_share: float


# =========================================================
# LOADERS
# =========================================================
def load_bundle(bundle_path: str | Path) -> dict:
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle file not found:\n{bundle_path}\n\n"
            "Run your training script first, or update DEFAULT_BUNDLE_PATH."
        )

    bundle = joblib.load(bundle_path)

    required_keys = {
        "model_total_bikes",
        "model_ebike_share",
        "features_total_bikes",
        "features_ebike_share",
    }
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(f"Bundle is missing required keys: {sorted(missing)}")

    return bundle


def load_features_dataset(features_path: str | Path) -> pd.DataFrame:
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found:\n{features_path}\n\n"
            "Run your ETL first, or update DEFAULT_FEATURES_PATH."
        )

    df = pd.read_parquet(features_path)

    required_cols = {"station_id", "snapshot_ts", "capacity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Features dataset is missing required columns: {sorted(missing)}")

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df = df.sort_values(["station_id", "snapshot_ts"]).reset_index(drop=True)
    return df


# =========================================================
# TIME HELPERS
# =========================================================
def floor_to_snapshot(ts: pd.Timestamp, freq_min: int = SNAPSHOT_FREQUENCY_MIN) -> pd.Timestamp:
    return ts.floor(f"{freq_min}min")


def parse_target_datetime(date_str: str, time_str: str) -> pd.Timestamp:
    naive = pd.Timestamp(f"{date_str} {time_str}")
    local_dt = naive.tz_localize(LOCAL_TZ)
    return local_dt.tz_convert(UTC_TZ)


def compute_anchor_timestamp(target_utc: pd.Timestamp) -> pd.Timestamp:
    anchor_utc = target_utc - pd.Timedelta(minutes=HORIZON_MINUTES)
    return floor_to_snapshot(anchor_utc, SNAPSHOT_FREQUENCY_MIN)


# =========================================================
# FEATURE SELECTION
# =========================================================
def get_nearest_station_row(
    df: pd.DataFrame,
    station_id: int,
    anchor_utc: pd.Timestamp,
    max_lookup_minutes: int = DEFAULT_MAX_LOOKUP_MINUTES,
) -> pd.Series:
    station_df = df[df["station_id"] == station_id].copy()

    if station_df.empty:
        raise ValueError(f"station_id={station_id} was not found in the features dataset.")

    station_df["abs_diff_sec"] = (station_df["snapshot_ts"] - anchor_utc).abs().dt.total_seconds()
    best_idx = station_df["abs_diff_sec"].idxmin()
    best_row = station_df.loc[best_idx].copy()

    diff_minutes = float(best_row["abs_diff_sec"]) / 60.0
    if diff_minutes > max_lookup_minutes:
        raise ValueError(
            f"No row found close enough to anchor timestamp for station_id={station_id}.\n"
            f"Anchor UTC: {anchor_utc}\n"
            f"Closest row UTC: {best_row['snapshot_ts']}\n"
            f"Gap: {diff_minutes:.1f} minutes\n\n"
            "Increase max_lookup_minutes or generate denser feature data."
        )

    return best_row


def ensure_feature_columns(row: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in row.index]
    if missing:
        raise ValueError(f"Missing feature columns required by the model: {missing}")

    X = pd.DataFrame([{col: row[col] for col in feature_cols}])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "The selected feature row contains NaN values in required columns: "
            f"{nan_cols}"
        )

    return X


# =========================================================
# RECONSTRUCTION
# =========================================================
def reconstruct_predictions(
    pred_total: float,
    pred_share: float,
    capacity: float,
) -> tuple[float, float, float]:
    pred_total = float(np.clip(pred_total, 0, capacity))
    pred_share = float(np.clip(pred_share, 0, 1))

    pred_ebike = pred_total * pred_share
    pred_mechanical = pred_total - pred_ebike

    return pred_total, pred_mechanical, pred_ebike


def round_and_enforce(
    pred_total: float,
    pred_mechanical: float,
    pred_ebike: float,
    capacity: float,
) -> tuple[int, int, int]:
    capacity_int = int(round(capacity))

    total_int = int(np.clip(np.round(pred_total), 0, capacity_int))
    ebike_int = int(np.clip(np.round(pred_ebike), 0, total_int))
    mechanical_int = int(np.clip(total_int - ebike_int, 0, total_int))

    return total_int, mechanical_int, ebike_int


# =========================================================
# PREDICTION
# =========================================================
def predict_station_bikes(
    station_id: int,
    target_date: str,
    target_time: str,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    max_lookup_minutes: int = DEFAULT_MAX_LOOKUP_MINUTES,
) -> PredictionResult:
    bundle = load_bundle(bundle_path)
    df = load_features_dataset(features_path)

    target_utc = parse_target_datetime(target_date, target_time)
    anchor_utc = compute_anchor_timestamp(target_utc)

    row = get_nearest_station_row(
        df=df,
        station_id=station_id,
        anchor_utc=anchor_utc,
        max_lookup_minutes=max_lookup_minutes,
    )

    total_features = bundle["features_total_bikes"]
    share_features = bundle["features_ebike_share"]
    model_total = bundle["model_total_bikes"]
    model_share = bundle["model_ebike_share"]

    X_total = ensure_feature_columns(row, total_features)
    X_share = ensure_feature_columns(row, share_features)

    pred_total_raw = float(model_total.predict(X_total)[0])
    pred_share_raw = float(model_share.predict(X_share)[0])

    capacity = float(row["capacity"])

    pred_total, pred_mechanical, pred_ebike = reconstruct_predictions(
        pred_total=pred_total_raw,
        pred_share=pred_share_raw,
        capacity=capacity,
    )

    pred_total_int, pred_mechanical_int, pred_ebike_int = round_and_enforce(
        pred_total=pred_total,
        pred_mechanical=pred_mechanical,
        pred_ebike=pred_ebike,
        capacity=capacity,
    )

    matched_delay_minutes = abs((row["snapshot_ts"] - anchor_utc).total_seconds()) / 60.0

    return PredictionResult(
        station_id=int(station_id),
        target_local=str(target_utc.tz_convert(LOCAL_TZ)),
        target_utc=str(target_utc),
        anchor_utc=str(anchor_utc),
        matched_snapshot_utc=str(row["snapshot_ts"]),
        matched_delay_minutes=round(float(matched_delay_minutes), 2),
        capacity=int(round(capacity)),
        predicted_total_bikes=int(pred_total_int),
        predicted_mechanical=int(pred_mechanical_int),
        predicted_ebike=int(pred_ebike_int),
        predicted_ebike_share=round(float(np.clip(pred_share_raw, 0, 1)), 4),
    )


# =========================================================
# PRINTING
# =========================================================
def print_result(result: PredictionResult) -> None:
    print("\n" + "=" * 84)
    print("BICING BIKE AVAILABILITY PREDICTION")
    print("=" * 84)
    print(f"Station ID                : {result.station_id}")
    print(f"Target datetime (local)   : {result.target_local}")
    print(f"Target datetime (UTC)     : {result.target_utc}")
    print(f"Anchor datetime (UTC)     : {result.anchor_utc}")
    print(f"Matched snapshot (UTC)    : {result.matched_snapshot_utc}")
    print(f"Match gap (minutes)       : {result.matched_delay_minutes}")
    print(f"Capacity                  : {result.capacity}")
    print("-" * 84)
    print(f"Predicted total bikes     : {result.predicted_total_bikes}")
    print(f"Predicted mechanical      : {result.predicted_mechanical}")
    print(f"Predicted ebikes          : {result.predicted_ebike}")
    print(f"Predicted ebike share     : {result.predicted_ebike_share}")
    print("=" * 84)


# =========================================================
# ARGUMENTS
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict mechanical bikes and ebikes for a station and target datetime."
    )
    parser.add_argument("--station_id", type=int, help="Station ID")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD (Barcelona local date)")
    parser.add_argument("--time", type=str, help="Target time in HH:MM (Barcelona local time)")
    parser.add_argument("--bundle_path", type=str, default=str(DEFAULT_BUNDLE_PATH), help="Path to bike_predictor_bundle_t_plus_24h.pkl")
    parser.add_argument("--features_path", type=str, default=str(DEFAULT_FEATURES_PATH), help="Path to bicing_selected_features.parquet")
    parser.add_argument("--max_lookup_minutes", type=int, default=DEFAULT_MAX_LOOKUP_MINUTES, help="Maximum allowed gap between anchor timestamp and matched feature row")
    return parser


def run_with_cli_args(args: argparse.Namespace) -> None:
    missing_required = [
        name for name, value in {
            "station_id": args.station_id,
            "date": args.date,
            "time": args.time,
        }.items()
        if value is None
    ]
    if missing_required:
        raise ValueError(
            "Missing required CLI arguments: "
            + ", ".join(missing_required)
            + "\nUse either full CLI args or run without args in PyCharm fallback mode."
        )

    result = predict_station_bikes(
        station_id=args.station_id,
        target_date=args.date,
        target_time=args.time,
        bundle_path=args.bundle_path,
        features_path=args.features_path,
        max_lookup_minutes=args.max_lookup_minutes,
    )
    print_result(result)


def run_with_pycharm_defaults() -> None:
    print("No CLI arguments detected. Running with PyCharm default values:")
    for key, value in PYCHARM_DEFAULTS.items():
        print(f"  - {key}: {value}")

    result = predict_station_bikes(
        station_id=int(PYCHARM_DEFAULTS["station_id"]),
        target_date=str(PYCHARM_DEFAULTS["target_date"]),
        target_time=str(PYCHARM_DEFAULTS["target_time"]),
        bundle_path=str(PYCHARM_DEFAULTS["bundle_path"]),
        features_path=str(PYCHARM_DEFAULTS["features_path"]),
        max_lookup_minutes=int(PYCHARM_DEFAULTS["max_lookup_minutes"]),
    )
    print_result(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    has_any_cli_args = len(sys.argv) > 1

    if has_any_cli_args:
        run_with_cli_args(args)
    else:
        run_with_pycharm_defaults()


if __name__ == "__main__":
    main()

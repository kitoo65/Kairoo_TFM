"""
Modified version of train_test_v10_split_features.py

Main change:
- No longer trains 2 direct models for mechanical / ebike
- Now trains:
    1) total_bikes
    2) ebike_share
- Then reconstructs:
    ebike = total_bikes * ebike_share
    mechanical = total_bikes - ebike

This mirrors the logic of train_test_total_share_t24h.py while keeping the
overall structure and style of your split-features script.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# PATHS / CONFIG
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "bicing_selected_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "model" / "models_bicing_planning_xgb_total_share_t24h_from_v10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
SNAPSHOT_FREQUENCY_MIN = 5
TRAIN_SPLIT_QUANTILE = 0.80
HORIZON_CONFIGS = {"t_plus_24h": 288}

# Columns that must never be used as predictors
DROP_COLUMNS_BASE = [
    "y_total_bikes",
    "y_ebike_share",
    "y_mechanical_future",
    "y_ebike_future",
    "y_total_future",
    "snapshot_ts",
    "scrapeid",
    "last_reported",
    "fetched_at_utc",
    "weather_ts",
]

# Features explicitly forbidden in planning mode because they would not be available at prediction time
FORBIDDEN_FEATURES = ["mechanical", "ebike"]


# FEATURE CONFIGURATION
# 1) FEATURES USED BY BOTH MODELS
COMMON_FEATURES = [
    "capacity",
    "lat",
    "lon",
    "altitude",
    "temperature_2m",
    "is_cold",
    "bad_weather_flag",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "week_sin",
    "hist_pct_bikes_available_station_hour",
    "pct_bikes_available",
    "dist_center_proxy",
]

# 2) FEATURES ONLY FOR THE TOTAL BIKES MODEL
TOTAL_ONLY_FEATURES = [
    "hist_mech_station_weekend_hour",
    "hist_mech_station_hour_dow",
    "hist_std_mech_station_hour",
    "hist_mech_peak_ratio_station",
    "hist_mech_station_hour",
    "hist_mech_station",
    "hist_ebike_station_hour",
    "hist_ebike_station_hour_dow",
    "hist_ebike_peak_ratio_station",
    "hist_ebike_station_weekend_hour",
]

# 3) FEATURES ONLY FOR THE EBIKE SHARE MODEL
SHARE_ONLY_FEATURES = [
    "hist_ebike_station_hour",
    "hist_ebike_station_hour_dow",
    "hist_ebike_peak_ratio_station",
    "hist_ebike_station_weekend_hour",
    "hist_mech_station_weekend_hour",
    "hist_mech_station_hour_dow",
    "hist_mech_peak_ratio_station",
    "is_morning_peak",
    "is_business_hour",
]

# 4) OPTIONAL FINAL MANUAL OVERRIDE PER TARGET
TOTAL_MANUAL_ADD = [
    # "station_id",
]

TOTAL_MANUAL_REMOVE = [
    # "lat",
]

SHARE_MANUAL_ADD = [
    # "station_id",
]

SHARE_MANUAL_REMOVE = [
    # "week_sin",
]


# LOAD DATASET
def load_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")
    df = pd.read_parquet(path)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df = df.sort_values(["station_id", "snapshot_ts"]).reset_index(drop=True)
    print(f"Loaded dataset shape: {df.shape}")
    return df


# =========================================================
# CREATE TARGETS
# =========================================================
def create_targets(df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    df = df.copy()

    df["y_mechanical_future"] = df.groupby("station_id")["mechanical"].shift(-forecast_horizon)
    df["y_ebike_future"] = df.groupby("station_id")["ebike"].shift(-forecast_horizon)
    df["y_total_future"] = df["y_mechanical_future"] + df["y_ebike_future"]

    df["y_total_bikes"] = df["y_total_future"]
    df["y_ebike_share"] = np.where(
        df["y_total_future"] > 0,
        df["y_ebike_future"] / df["y_total_future"],
        0.0,
    )

    df = df.dropna(
        subset=[
            "y_mechanical_future",
            "y_ebike_future",
            "y_total_future",
            "y_total_bikes",
            "y_ebike_share",
        ]
    ).reset_index(drop=True)

    horizon_hours = forecast_horizon * SNAPSHOT_FREQUENCY_MIN / 60
    print(f"Forecast horizon (steps): {forecast_horizon}")
    print(f"Forecast horizon (hours): {horizon_hours:.2f}")
    print(f"Dataset shape after target creation: {df.shape}")
    return df


# =========================================================
# ENCODE CATEGORICALS
# =========================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    remaining_non_numeric = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("Remaining object/category columns:", remaining_non_numeric)
    return df


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
def split_train_test(df: pd.DataFrame, split_quantile: float = 0.80):
    split_date = df["snapshot_ts"].quantile(split_quantile)
    train = df[df["snapshot_ts"] < split_date].copy()
    test = df[df["snapshot_ts"] >= split_date].copy()
    print(f"Split date: {split_date}")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test, split_date


# =========================================================
# FEATURE HELPERS
# =========================================================
def deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_requested_features_for_target(target_name: str) -> list[str]:
    if target_name == "total_bikes":
        requested_features = COMMON_FEATURES + TOTAL_ONLY_FEATURES + TOTAL_MANUAL_ADD
        remove_features = set(TOTAL_MANUAL_REMOVE)
    elif target_name == "ebike_share":
        requested_features = COMMON_FEATURES + SHARE_ONLY_FEATURES + SHARE_MANUAL_ADD
        remove_features = set(SHARE_MANUAL_REMOVE)
    else:
        raise ValueError(f"Unknown target_name: {target_name}")

    requested_features = deduplicate_preserve_order(requested_features)
    requested_features = [col for col in requested_features if col not in remove_features]
    return requested_features


# BUILD FEATURE MATRIX
def build_feature_matrix_for_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_name: str,
    drop_cols: list[str],
    forbidden_features: list[str] | None = None,
):
    forbidden_features = forbidden_features or []

    requested_features = get_requested_features_for_target(target_name)
    requested_features = [col for col in requested_features if col not in forbidden_features]
    requested_features = [col for col in requested_features if col not in drop_cols]

    feature_cols = [col for col in requested_features if col in train.columns and col in test.columns]

    missing_requested = [col for col in requested_features if col not in train.columns]
    if missing_requested:
        print(f"\nRequested features not found in dataset for target={target_name}:")
        print(missing_requested)

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    print("\n" + "-" * 100)
    print(f"TARGET: {target_name}")
    print(f"Number of features used: {len(feature_cols)}")
    print("Final feature list:")
    print(feature_cols)

    return X_train, X_test, feature_cols


# =========================================================
# MODEL
# =========================================================
def build_xgb_model(target_name: str) -> XGBRegressor:
    if target_name == "total_bikes":
        return XGBRegressor(
            n_estimators=1100, #1100
            learning_rate=0.02,
            max_depth=10,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif target_name == "ebike_share":
        return XGBRegressor(
            n_estimators=900, #900
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown target_name: {target_name}")


# =========================================================
# METRICS / IMPORTANCE
# =========================================================
def evaluate_predictions(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def get_feature_importance_df(model, feature_cols):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if not importance_df.empty and importance_df["importance"].sum() > 0:
        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum()
        )
    else:
        importance_df["importance_pct"] = 0.0

    return importance_df


def top_importance_summary(importance_df: pd.DataFrame, top_n: int = 5) -> dict:
    if importance_df.empty:
        return {
            "top_1_feature": None,
            "top_1_importance_pct": np.nan,
            "top_5_importance_pct": np.nan,
        }

    return {
        "top_1_feature": importance_df.iloc[0]["feature"],
        "top_1_importance_pct": float(importance_df.iloc[0]["importance_pct"]),
        "top_5_importance_pct": float(importance_df.head(top_n)["importance_pct"].sum()),
    }


# =========================================================
# SAVE HELPERS
# =========================================================
def save_model(model, path: Path):
    joblib.dump(model, path)
    print(f"Saved model to: {path}")


def save_feature_list(features: list[str], path: Path):
    joblib.dump(features, path)
    print(f"Saved feature list to: {path}")


def save_feature_importance(importance_df: pd.DataFrame, path: Path):
    importance_df.to_csv(path, index=False)
    print(f"Saved feature importance to: {path}")


def plot_top_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
):
    if importance_df.empty:
        return

    plot_df = importance_df.copy()

    fig_height = max(8, len(plot_df) * 0.38)
    plt.figure(figsize=(12, fig_height))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved feature importance plot to: {output_path}")


def plot_dual_model_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    horizon_name: str,
):
    if importance_df.empty:
        return

    targets = ["total_bikes", "ebike_share"]
    max_features = importance_df.groupby("target")["feature"].count().max()
    fig_height = max(8, max_features * 0.38)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, fig_height),
        sharex=False
    )

    for ax, target in zip(axes, targets):
        plot_df = (
            importance_df[importance_df["target"] == target]
            .sort_values("importance", ascending=False)
            .copy()
        )

        if plot_df.empty:
            ax.text(0.5, 0.5, f"No importance data for {target}", ha="center", va="center")
            ax.set_title(target.capitalize())
            ax.axis("off")
            continue

        plot_df = plot_df.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["importance"])
        ax.set_title(target.replace("_", " ").title())
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

    fig.suptitle(f"All Feature Importances by Model - {horizon_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined feature importance plot to: {output_path}")


def save_predictions_dataframe(pred_df: pd.DataFrame, output_path: Path):
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


def save_unified_bundle(model_total, model_share, total_features: list[str], share_features: list[str], path: Path):
    bundle = {
        "model_total_bikes": model_total,
        "model_ebike_share": model_share,
        "features_total_bikes": total_features,
        "features_ebike_share": share_features,
        "reconstruction_logic": "ebikes = total_bikes * ebike_share ; mechanical = total_bikes - ebikes",
    }
    joblib.dump(bundle, path)
    print(f"Saved unified bundle to: {path}")


def reconstruct_predictions(pred_total, pred_share, capacity):
    pred_total = np.asarray(pred_total, dtype=float)
    pred_share = np.asarray(pred_share, dtype=float)
    capacity = np.asarray(capacity, dtype=float)

    pred_total = np.clip(pred_total, 0, capacity)
    pred_share = np.clip(pred_share, 0, 1)

    pred_ebikes = pred_total * pred_share
    pred_mechanical = pred_total - pred_ebikes

    return pred_total, pred_mechanical, pred_ebikes


def round_and_enforce(pred_total, pred_mechanical, pred_ebikes, capacity):
    capacity = np.asarray(capacity, dtype=int)

    total_int = np.round(pred_total).astype(int)
    total_int = np.clip(total_int, 0, capacity)

    ebike_int = np.round(pred_ebikes).astype(int)
    ebike_int = np.clip(ebike_int, 0, total_int)

    mechanical_int = total_int - ebike_int
    mechanical_int = np.clip(mechanical_int, 0, total_int)

    return total_int, mechanical_int, ebike_int


def build_reconstruction_preview(test_df, pred_total, pred_mech, pred_ebike):
    preview_df = pd.DataFrame({
        "station_id": test_df["station_id"].values,
        "snapshot_ts": test_df["snapshot_ts"].values,
        "capacity": test_df["capacity"].values,
        "y_true_total_bikes": test_df["y_total_future"].values,
        "y_pred_total_bikes": pred_total,
        "y_true_mechanical": test_df["y_mechanical_future"].values,
        "y_pred_mechanical": pred_mech,
        "y_true_ebike": test_df["y_ebike_future"].values,
        "y_pred_ebike": pred_ebike,
    })
    return preview_df


# =========================================================
# TRAIN ONE MODEL
# =========================================================
def train_one_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_cols,
    model_name: str,
    output_dir: Path,
    target_name: str,
):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)

    importance_df = get_feature_importance_df(model, feature_cols)
    imp_summary = top_importance_summary(importance_df)

    safe_model_name = model_name.lower().replace(" ", "_")

    model_path = output_dir / f"{safe_model_name}.pkl"
    features_path = output_dir / f"{safe_model_name}_features.pkl"
    importance_csv_path = output_dir / f"{safe_model_name}_feature_importance.csv"
    importance_png_path = output_dir / f"{safe_model_name}_feature_importance.png"

    save_model(model, model_path)
    save_feature_list(feature_cols, features_path)
    save_feature_importance(importance_df, importance_csv_path)
    plot_top_feature_importance(
        importance_df=importance_df,
        model_name=model_name,
        output_path=importance_png_path,
    )

    print(
        f"{model_name} | "
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"Top1={imp_summary['top_1_feature']} "
        f"({imp_summary['top_1_importance_pct']:.2%})"
    )

    print("\n" + "-" * 100)
    print(f"FULL FEATURE IMPORTANCE - {model_name}")
    print("-" * 100)
    print(importance_df.to_string(index=False))

    return {
        "model_name": model_name,
        "target_name": target_name,
        "model": model,
        "preds_raw": preds,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "top_1_feature": imp_summary["top_1_feature"],
        "top_1_importance_pct": imp_summary["top_1_importance_pct"],
        "top_5_importance_pct": imp_summary["top_5_importance_pct"],
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "importance_df": importance_df,
    }


# =========================================================
# RUN ONE HORIZON
# =========================================================
def run_horizon(df_base: pd.DataFrame, horizon_name: str, horizon_steps: int, output_root: Path):
    print("\n" + "=" * 100)
    print(f"RUNNING HORIZON: {horizon_name}")
    print("=" * 100)

    df = create_targets(df_base, horizon_steps)
    df = encode_categoricals(df)
    train, test, split_date = split_train_test(df, TRAIN_SPLIT_QUANTILE)

    horizon_output_dir = output_root / horizon_name
    horizon_output_dir.mkdir(parents=True, exist_ok=True)

    # TOTAL MODEL
    X_train_total, X_test_total, total_feature_cols = build_feature_matrix_for_target(
        train=train,
        test=test,
        target_name="total_bikes",
        drop_cols=DROP_COLUMNS_BASE,
        forbidden_features=FORBIDDEN_FEATURES,
    )

    total_result = train_one_model(
        model=build_xgb_model("total_bikes"),
        X_train=X_train_total,
        y_train=train["y_total_bikes"],
        X_test=X_test_total,
        y_test=test["y_total_bikes"],
        feature_cols=total_feature_cols,
        model_name=f"xgboost_planning_total_bikes_{horizon_name}",
        output_dir=horizon_output_dir,
        target_name="total_bikes",
    )

    # SHARE MODEL
    train_share = train[train["y_total_bikes"] > 0].copy()
    test_share = test[test["y_total_bikes"] > 0].copy()

    share_test_ref = test_share if not test_share.empty else train_share.iloc[:1].copy()

    X_train_share, _, share_feature_cols = build_feature_matrix_for_target(
        train=train_share,
        test=share_test_ref,
        target_name="ebike_share",
        drop_cols=DROP_COLUMNS_BASE,
        forbidden_features=FORBIDDEN_FEATURES,
    )

    X_test_share_all = test[share_feature_cols].copy()

    share_result = train_one_model(
        model=build_xgb_model("ebike_share"),
        X_train=X_train_share,
        y_train=train_share["y_ebike_share"],
        X_test=X_test_share_all,
        y_test=test["y_ebike_share"],
        feature_cols=share_feature_cols,
        model_name=f"xgboost_planning_ebike_share_{horizon_name}",
        output_dir=horizon_output_dir,
        target_name="ebike_share",
    )

    # RECONSTRUCTION
    capacity_test = test["capacity"].values.astype(float)

    pred_total_raw = total_result["preds_raw"]
    pred_share_raw = share_result["preds_raw"]

    pred_total, pred_mechanical, pred_ebike = reconstruct_predictions(
        pred_total=pred_total_raw,
        pred_share=pred_share_raw,
        capacity=capacity_test,
    )

    pred_total_int, pred_mechanical_int, pred_ebike_int = round_and_enforce(
        pred_total=pred_total,
        pred_mechanical=pred_mechanical,
        pred_ebikes=pred_ebike,
        capacity=capacity_test,
    )

    # METRICS
    real_total = test["y_total_future"].values
    real_mechanical = test["y_mechanical_future"].values
    real_ebike = test["y_ebike_future"].values

    total_metrics = evaluate_predictions(real_total, pred_total_int)
    share_metrics = evaluate_predictions(test["y_ebike_share"].values, np.clip(pred_share_raw, 0, 1))
    mech_metrics = evaluate_predictions(real_mechanical, pred_mechanical_int)
    ebike_metrics = evaluate_predictions(real_ebike, pred_ebike_int)

    results = [
        {
            "model_name": f"xgboost_planning_total_bikes_{horizon_name}",
            "target": "total_bikes",
            "mae": total_metrics["mae"],
            "rmse": total_metrics["rmse"],
            "r2": total_metrics["r2"],
            "top_1_feature": total_result["top_1_feature"],
            "top_1_importance_pct": total_result["top_1_importance_pct"],
            "top_5_importance_pct": total_result["top_5_importance_pct"],
            "n_features": len(total_feature_cols),
            "horizon": horizon_name,
            "horizon_steps": horizon_steps,
            "horizon_hours": horizon_steps * SNAPSHOT_FREQUENCY_MIN / 60,
            "split_date": split_date,
        },
        {
            "model_name": f"xgboost_planning_ebike_share_{horizon_name}",
            "target": "ebike_share",
            "mae": share_metrics["mae"],
            "rmse": share_metrics["rmse"],
            "r2": share_metrics["r2"],
            "top_1_feature": share_result["top_1_feature"],
            "top_1_importance_pct": share_result["top_1_importance_pct"],
            "top_5_importance_pct": share_result["top_5_importance_pct"],
            "n_features": len(share_feature_cols),
            "horizon": horizon_name,
            "horizon_steps": horizon_steps,
            "horizon_hours": horizon_steps * SNAPSHOT_FREQUENCY_MIN / 60,
            "split_date": split_date,
        },
        {
            "model_name": f"reconstructed_mechanical_{horizon_name}",
            "target": "mechanical",
            "mae": mech_metrics["mae"],
            "rmse": mech_metrics["rmse"],
            "r2": mech_metrics["r2"],
            "top_1_feature": "derived_from_total_and_share",
            "top_1_importance_pct": np.nan,
            "top_5_importance_pct": np.nan,
            "n_features": len(total_feature_cols),
            "horizon": horizon_name,
            "horizon_steps": horizon_steps,
            "horizon_hours": horizon_steps * SNAPSHOT_FREQUENCY_MIN / 60,
            "split_date": split_date,
        },
        {
            "model_name": f"reconstructed_ebike_{horizon_name}",
            "target": "ebike",
            "mae": ebike_metrics["mae"],
            "rmse": ebike_metrics["rmse"],
            "r2": ebike_metrics["r2"],
            "top_1_feature": "derived_from_total_and_share",
            "top_1_importance_pct": np.nan,
            "top_5_importance_pct": np.nan,
            "n_features": len(share_feature_cols),
            "horizon": horizon_name,
            "horizon_steps": horizon_steps,
            "horizon_hours": horizon_steps * SNAPSHOT_FREQUENCY_MIN / 60,
            "split_date": split_date,
        },
    ]

    predictions_preview = build_reconstruction_preview(
        test_df=test,
        pred_total=pred_total_int,
        pred_mech=pred_mechanical_int,
        pred_ebike=pred_ebike_int,
    )
    predictions_preview_path = horizon_output_dir / f"reconstructed_predictions_{horizon_name}.csv"
    save_predictions_dataframe(predictions_preview, predictions_preview_path)

    total_importance_df = total_result["importance_df"].copy()
    total_importance_df["model_name"] = f"xgboost_planning_total_bikes_{horizon_name}"
    total_importance_df["horizon"] = horizon_name
    total_importance_df["horizon_steps"] = horizon_steps
    total_importance_df["horizon_hours"] = horizon_steps * SNAPSHOT_FREQUENCY_MIN / 60
    total_importance_df["target"] = "total_bikes"

    share_importance_df = share_result["importance_df"].copy()
    share_importance_df["model_name"] = f"xgboost_planning_ebike_share_{horizon_name}"
    share_importance_df["horizon"] = horizon_name
    share_importance_df["horizon_steps"] = horizon_steps
    share_importance_df["horizon_hours"] = horizon_steps * SNAPSHOT_FREQUENCY_MIN / 60
    share_importance_df["target"] = "ebike_share"

    horizon_importance_df = pd.concat([total_importance_df, share_importance_df], ignore_index=True)
    horizon_importance_path = horizon_output_dir / f"all_feature_importance_{horizon_name}.csv"
    horizon_importance_df.to_csv(horizon_importance_path, index=False)
    print(f"Saved all feature importance for {horizon_name} to: {horizon_importance_path}")

    combined_importance_png_path = horizon_output_dir / f"combined_feature_importance_{horizon_name}.png"
    plot_dual_model_feature_importance(
        importance_df=horizon_importance_df,
        output_path=combined_importance_png_path,
        horizon_name=horizon_name,
    )

    horizon_results_df = pd.DataFrame(results)
    horizon_results_path = horizon_output_dir / f"results_{horizon_name}.csv"
    horizon_results_df.to_csv(horizon_results_path, index=False)
    print(f"Saved horizon results to: {horizon_results_path}")

    bundle_path = horizon_output_dir / f"bike_predictor_bundle_{horizon_name}.pkl"
    save_unified_bundle(
        model_total=total_result["model"],
        model_share=share_result["model"],
        total_features=total_feature_cols,
        share_features=share_feature_cols,
        path=bundle_path,
    )

    return horizon_results_df, horizon_importance_df


# =========================================================
# SUMMARY TABLES
# =========================================================
def build_clean_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.copy()

    summary["mae"] = summary["mae"].round(4)
    summary["rmse"] = summary["rmse"].round(4)
    summary["r2"] = summary["r2"].round(4)
    summary["top_1_importance_pct"] = (summary["top_1_importance_pct"] * 100).round(2)
    summary["top_5_importance_pct"] = (summary["top_5_importance_pct"] * 100).round(2)
    summary["horizon_hours"] = summary["horizon_hours"].round(2)

    summary = summary.rename(columns={
        "horizon": "Horizon",
        "horizon_steps": "Horizon Steps",
        "horizon_hours": "Horizon Hours",
        "target": "Target",
        "n_features": "N Features",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R2",
        "top_1_feature": "Top 1 Feature",
        "top_1_importance_pct": "Top 1 Importance %",
        "top_5_importance_pct": "Top 5 Importance %",
    })

    summary = summary[
        [
            "Horizon",
            "Horizon Steps",
            "Horizon Hours",
            "Target",
            "N Features",
            "MAE",
            "RMSE",
            "R2",
            "Top 1 Feature",
            "Top 1 Importance %",
            "Top 5 Importance %",
        ]
    ].sort_values(["Target", "Horizon Hours"]).reset_index(drop=True)

    return summary


def build_horizon_pivot(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot_table(
        index="target",
        columns="horizon",
        values="mae",
        aggfunc="first"
    ).round(4)

    return pivot.reset_index()


# =========================================================
# MAIN
# =========================================================
def main():
    df_base = load_dataset(DATA_PATH)

    all_results = []
    all_importances = []

    for horizon_name, horizon_steps in HORIZON_CONFIGS.items():
        horizon_results, horizon_importance = run_horizon(
            df_base=df_base,
            horizon_name=horizon_name,
            horizon_steps=horizon_steps,
            output_root=OUTPUT_DIR,
        )
        all_results.append(horizon_results)
        all_importances.append(horizon_importance)

    results = pd.concat(all_results, ignore_index=True)
    full_importance = pd.concat(all_importances, ignore_index=True)

    raw_results_path = OUTPUT_DIR / "all_planning_results_raw.csv"
    results.to_csv(raw_results_path, index=False)
    print(f"Saved raw results to: {raw_results_path}")

    clean_summary = build_clean_summary(results)
    clean_summary_path = OUTPUT_DIR / "tfm_planning_summary_table.csv"
    clean_summary.to_csv(clean_summary_path, index=False)
    print(f"Saved TFM summary table to: {clean_summary_path}")

    mae_pivot = build_horizon_pivot(results)
    mae_pivot_path = OUTPUT_DIR / "tfm_planning_mae_pivot.csv"
    mae_pivot.to_csv(mae_pivot_path, index=False)
    print(f"Saved MAE pivot table to: {mae_pivot_path}")

    full_importance_path = OUTPUT_DIR / "all_feature_importance_full.csv"
    full_importance.to_csv(full_importance_path, index=False)
    print(f"Saved full feature importance to: {full_importance_path}")

    print("\n" + "=" * 100)
    print("TFM PLANNING SUMMARY TABLE")
    print("=" * 100)
    print(clean_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("FULL FEATURE IMPORTANCE")
    print("=" * 100)
    print(full_importance.to_string(index=False))

    print("\n" + "=" * 100)
    print("MAE PIVOT TABLE")
    print("=" * 100)
    print(mae_pivot.to_string(index=False))


if __name__ == "__main__":
    main()

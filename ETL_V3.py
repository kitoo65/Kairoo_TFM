import pandas as pd
import numpy as np
import logging
import traceback
import time
from pathlib import Path
import json
import ast
import requests
import os
import certifi
import urllib3


# =========================
# Logger
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

INFO_PATH = PROJECT_ROOT / "info_appended.parquet"
STATUS_PATH = PROJECT_ROOT / "status_appended.parquet"
HOLIDAYS_PATH = PROJECT_ROOT / "barcelona_holidays.csv"
WEATHER_PATH = PROJECT_ROOT / "barcelona_historical_weather.csv"

OUTPUT_BASE_PARQUET = PROJECT_ROOT / "bicing_model_base.parquet"
OUTPUT_FEATURES_PARQUET = PROJECT_ROOT / "bicing_selected_features.parquet"
OUTPUT_MECHANICAL_FEATURES_PARQUET = PROJECT_ROOT / "bicing_mechanical_features.parquet"
OUTPUT_EBIKE_FEATURES_PARQUET = PROJECT_ROOT / "bicing_ebike_features.parquet"

CORPORATE_CA_PATH = os.getenv("REQUESTS_CA_BUNDLE", "")
# =========================
# Feature configuration
# =========================
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

MECHANICAL_ONLY_FEATURES = [
    "hist_mech_station_weekend_hour",
    "hist_mech_station_hour_dow",
    "hist_std_mech_station_hour",
    "hist_mech_peak_ratio_station",
    "hist_mech_station_hour",
    "hist_mech_station",
]

EBIKE_ONLY_FEATURES = [
    "hist_ebike_station_hour",
    "hist_ebike_station_hour_dow",
    # "hist_std_ebike_station_hour",
    "hist_ebike_peak_ratio_station",
    "hist_ebike_station_weekend_hour",
    "is_morning_peak",
    "is_business_hour",
]

MECHANICAL_TARGET = "mechanical"
EBIKE_TARGET = "ebike"


# =========================
# Constants
# =========================
BARCELONA_CENTER_LAT = 41.3874
BARCELONA_CENTER_LON = 2.1686

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_TIMEOUT_SEC = 60
ALLOW_INSECURE_SSL = True
WEATHER_CHUNK_DAYS = 60


# =========================
# Generic helpers
# =========================
def load_from_parquet(file_path: str | Path) -> pd.DataFrame | None:
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Loading parquet file from '{file_path}'...")
        start_time = time.time()

        df = pd.read_parquet(file_path)

        elapsed_time = time.time() - start_time
        logger.info(f"Loaded '{file_path}' in {elapsed_time:.2f} seconds. Shape={df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")
        traceback.print_exc()
        return None

def get_requests_verify_value():
    """
    Priority:
    1) REQUESTS_CA_BUNDLE env var if provided
    2) certifi default bundle
    3) False only if ALLOW_INSECURE_SSL=True
    """
    if CORPORATE_CA_PATH:
        logger.info(f"Using custom CA bundle: {CORPORATE_CA_PATH}")
        return CORPORATE_CA_PATH

    if ALLOW_INSECURE_SSL:
        logger.warning("SSL verification is DISABLED. Use only for temporary local testing.")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return False

    ca_path = certifi.where()
    logger.info(f"Using certifi CA bundle: {ca_path}")
    return ca_path


def save_to_parquet(df: pd.DataFrame, file_path: str | Path) -> bool:
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving parquet file to '{file_path}'...")
        start_time = time.time()

        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)

        elapsed_time = time.time() - start_time
        logger.info(f"Saved '{file_path}' in {elapsed_time:.2f} seconds. Shape={df.shape}")
        return True

    except Exception as e:
        logger.error(f"Error saving parquet file: {e}")
        traceback.print_exc()
        return False


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan)


def load_holidays(path: Path) -> set:
    if not path.exists():
        logger.warning(f"Holiday file not found: {path}. Holiday features will default to 0.")
        return set()

    df = pd.read_csv(path)
    if "date" not in df.columns:
        logger.warning("Holiday file does not contain a 'date' column. Holiday features will default to 0.")
        return set()

    holiday_dates = set(pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().tolist())
    logger.info(f"Loaded {len(holiday_dates)} holidays")
    return holiday_dates


def to_nullable_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
        "on": True,
        "off": False,
    }

    s = series.astype("string").str.strip().str.lower()
    s = s.map(mapping)

    return s.astype("boolean")


def parse_bike_types(x):
    if pd.isna(x):
        return {}

    if isinstance(x, dict):
        return x

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return {}

        try:
            return json.loads(x)
        except Exception:
            pass

        try:
            return ast.literal_eval(x)
        except Exception:
            return {}

    return {}


# =========================
# Cleaning: info
# =========================
def clean_info(df_info: pd.DataFrame) -> pd.DataFrame:
    df = df_info.copy()
    logger.info("Cleaning df_info...")

    df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
    df["scrapeid"] = pd.to_numeric(df["scrapeid"], errors="coerce")
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], errors="coerce", utc=True)

    df = df.dropna(subset=["station_id", "scrapeid", "fetched_at_utc"])
    df["station_id"] = df["station_id"].astype("int64")
    df["scrapeid"] = df["scrapeid"].astype("int64")

    if "post_code" in df.columns:
        df["post_code"] = pd.to_numeric(df["post_code"], errors="coerce").astype("Int64")

    for col in ["lat", "lon", "altitude", "capacity", "nearby_distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["is_charging_station", "_ride_code_support"]:
        if col in df.columns:
            df[col] = to_nullable_boolean(df[col])

    keep_info = [
        "station_id",
        "scrapeid",
        "fetched_at_utc",
        "lat",
        "lon",
        "altitude",
        "post_code",
        "capacity",
        "is_charging_station",
    ]
    keep_info = [c for c in keep_info if c in df.columns]
    df = df[keep_info]

    df = (
        df.sort_values(["station_id", "scrapeid", "fetched_at_utc"])
          .drop_duplicates(subset=["station_id", "scrapeid"], keep="last")
          .reset_index(drop=True)
    )

    logger.info(f"df_info cleaned. Shape={df.shape}")
    return df


# =========================
# Cleaning: status
# =========================
def clean_status(df_status: pd.DataFrame) -> pd.DataFrame:
    df = df_status.copy()
    logger.info("Cleaning df_status...")

    if "num_bikes_available_types" in df.columns:
        parsed = df["num_bikes_available_types"].apply(parse_bike_types)
        df["mechanical"] = parsed.apply(
            lambda d: d.get("mechanical") if isinstance(d, dict) else None
        )
        df["ebike"] = parsed.apply(
            lambda d: d.get("ebike") if isinstance(d, dict) else None
        )

    df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
    df["scrapeid"] = pd.to_numeric(df["scrapeid"], errors="coerce")
    df["last_reported"] = pd.to_numeric(df["last_reported"], errors="coerce")

    df = df.dropna(subset=["station_id", "scrapeid", "last_reported"])
    df["station_id"] = df["station_id"].astype("int64")
    df["scrapeid"] = df["scrapeid"].astype("int64")

    df["last_reported"] = pd.to_datetime(df["last_reported"], unit="s", errors="coerce", utc=True)
    df = df.dropna(subset=["last_reported"])

    numeric_cols = [
        "num_bikes_available",
        "num_bikes_disabled",
        "num_docks_available",
        "num_docks_disabled",
        "mechanical",
        "ebike",
        "is_installed",
        "is_renting",
        "is_returning",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "status" in df.columns:
        df["status"] = df["status"].astype("category")

    if "is_charging_station" in df.columns:
        df["is_charging_station"] = to_nullable_boolean(df["is_charging_station"])

    keep_status = [
        "station_id",
        "scrapeid",
        "last_reported",
        "status",
        "num_bikes_available",
        "num_bikes_disabled",
        "num_docks_available",
        "num_docks_disabled",
        "mechanical",
        "ebike",
        "is_installed",
        "is_renting",
        "is_returning",
        "is_charging_station",
    ]
    keep_status = [c for c in keep_status if c in df.columns]
    df = df[keep_status]

    required_for_targets = ["mechanical", "ebike"]
    missing = [c for c in required_for_targets if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns in status data: {missing}")

    df = (
        df.sort_values(["station_id", "scrapeid", "last_reported"])
          .drop_duplicates(subset=["station_id", "scrapeid"], keep="last")
          .reset_index(drop=True)
    )

    logger.info(f"df_status cleaned. Shape={df.shape}")
    logger.info(f"df_status cleaned columns: {df.columns.tolist()}")
    return df


# =========================
# Join info + status
# =========================
def join_info_status(df_info_clean: pd.DataFrame, df_status_clean: pd.DataFrame) -> pd.DataFrame:
    logger.info("Joining df_info and df_status...")

    df = df_status_clean.merge(
        df_info_clean,
        on=["station_id", "scrapeid"],
        how="left",
        suffixes=("_status", "_info")
    )

    df["snapshot_ts"] = df["last_reported"].combine_first(df["fetched_at_utc"])

    if "is_charging_station_status" in df.columns and "is_charging_station_info" in df.columns:
        df["is_charging_station"] = df["is_charging_station_status"].combine_first(
            df["is_charging_station_info"]
        )
        df = df.drop(columns=["is_charging_station_status", "is_charging_station_info"])

    # Fill slow-changing / static station info by station_id
    static_info_cols = [
        "lat",
        "lon",
        "altitude",
        "post_code",
        "capacity",
        "is_charging_station",
    ]
    static_info_cols = [c for c in static_info_cols if c in df.columns]

    if static_info_cols:
        df = df.sort_values(["station_id", "snapshot_ts", "scrapeid"]).reset_index(drop=True)
        df[static_info_cols] = (
            df.groupby("station_id")[static_info_cols]
              .transform(lambda g: g.ffill().bfill())
        )

    df = (
        df.sort_values(["station_id", "snapshot_ts", "scrapeid"])
          .drop_duplicates(subset=["station_id", "snapshot_ts"], keep="last")
          .reset_index(drop=True)
    )

    logger.info(f"Joined dataset created. Shape={df.shape}")
    return df

# =========================
# Weather: automatic fetch from Open-Meteo
# =========================
def _build_weather_hourly_df(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response does not contain expected hourly data.")

    df = pd.DataFrame({
        "weather_ts": pd.to_datetime(hourly["time"], utc=True, errors="coerce"),
        "temperature_2m": hourly.get("temperature_2m"),
        "rain_mm": hourly.get("precipitation"),
        "wind_speed_10m": hourly.get("wind_speed_10m"),
        "cloud_cover": hourly.get("cloud_cover"),
        "sunshine_duration_sec": hourly.get("sunshine_duration"),
    })

    if "sunshine_duration_sec" in df.columns:
        df["sunshine_hours"] = pd.to_numeric(df["sunshine_duration_sec"], errors="coerce") / 3600.0
    else:
        df["sunshine_hours"] = np.nan

    df["is_raining"] = (pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0) > 0).astype(int)

    df = df.dropna(subset=["weather_ts"]).reset_index(drop=True)
    return df


def fetch_open_meteo_weather_chunk(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "sunshine_duration",
        ]),
    }

    logger.info(f"Fetching weather from Open-Meteo: {start_date} -> {end_date}")

    verify_value = get_requests_verify_value()

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=OPEN_METEO_TIMEOUT_SEC,
        verify=verify_value,
    )
    response.raise_for_status()

    payload = response.json()

    df = _build_weather_hourly_df(payload)
    logger.info(f"Fetched weather chunk. Shape={df.shape}")
    return df


def fetch_weather_dataset_auto(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    latitude: float = BARCELONA_CENTER_LAT,
    longitude: float = BARCELONA_CENTER_LON,
    output_path: str | Path | None = WEATHER_PATH,
    chunk_days: int = WEATHER_CHUNK_DAYS,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_ts, utc=True)
    end_ts = pd.to_datetime(end_ts, utc=True)

    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Weather fetch range contains NaT values.")

    start_date = start_ts.floor("D")
    end_date = end_ts.floor("D")

    if end_date < start_date:
        raise ValueError("Weather fetch end date is before start date.")

    logger.info(
        f"Auto-fetching weather for Barcelona from {start_date.date()} to {end_date.date()}"
    )

    parts = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_date)

        chunk_df = fetch_open_meteo_weather_chunk(
            latitude=latitude,
            longitude=longitude,
            start_date=current_start.strftime("%Y-%m-%d"),
            end_date=current_end.strftime("%Y-%m-%d"),
        )
        parts.append(chunk_df)

        current_start = current_end + pd.Timedelta(days=1)

    if not parts:
        raise ValueError("No weather data could be fetched.")

    df_weather = pd.concat(parts, ignore_index=True)

    for col in ["temperature_2m", "rain_mm", "wind_speed_10m", "cloud_cover", "sunshine_hours"]:
        if col in df_weather.columns:
            df_weather[col] = pd.to_numeric(df_weather[col], errors="coerce")

    df_weather = (
        df_weather.sort_values("weather_ts")
        .drop_duplicates(subset=["weather_ts"], keep="last")
        .reset_index(drop=True)
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_weather.to_csv(output_path, index=False)
        logger.info(f"Weather cache saved to: {output_path}")

    logger.info(
        f"Weather fetch complete. Shape={df_weather.shape} | "
        f"MIN={df_weather['weather_ts'].min()} | MAX={df_weather['weather_ts'].max()}"
    )
    return df_weather


# =========================
# Weather merge
# =========================
def merge_weather(df_base: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging weather dataset with bike dataset...")

    df = df_base.copy()
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["weather_ts"] = df["snapshot_ts"].dt.floor("h")

    weather_cols = [
        "weather_ts",
        "temperature_2m",
        "rain_mm",
        "wind_speed_10m",
        "cloud_cover",
        "sunshine_hours",
        "is_raining",
    ]
    weather_cols = [c for c in weather_cols if c in df_weather.columns]

    df = df.merge(df_weather[weather_cols], on="weather_ts", how="left")

    logger.info(f"Dataset shape after weather merge: {df.shape}")
    logger.info(f"snapshot_ts MIN={df['snapshot_ts'].min()} | MAX={df['snapshot_ts'].max()}")
    if "weather_ts" in df_weather.columns:
        logger.info(f"weather_ts  MIN={df_weather['weather_ts'].min()} | MAX={df_weather['weather_ts'].max()}")

    return df


# =========================
# Base planning features
# =========================
def add_planning_base_features(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    logger.info("Creating planning-safe base features...")

    df = df.copy()
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df = df.sort_values(["station_id", "snapshot_ts"]).reset_index(drop=True)

    df["hour"] = df["snapshot_ts"].dt.hour
    df["minute"] = df["snapshot_ts"].dt.minute
    df["day_of_week"] = df["snapshot_ts"].dt.dayofweek
    df["month"] = df["snapshot_ts"].dt.month
    df["week_of_year"] = df["snapshot_ts"].dt.isocalendar().week.astype(int)
    df["date_only"] = df["snapshot_ts"].dt.date

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hour"] = df["hour"].between(8, 18).astype(int)
    df["is_morning_peak"] = df["hour"].between(7, 9).astype(int)
    df["is_holiday"] = df["date_only"].isin(holiday_dates).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["week_sin"] = np.sin(2 * np.pi * (df["week_of_year"] - 1) / 52)

    if "temperature_2m" in df.columns:
        df["is_cold"] = (df["temperature_2m"] <= 8).astype(int)
    else:
        df["is_cold"] = 0

    if "rain_mm" in df.columns:
        df["heavy_rain"] = (df["rain_mm"] >= 2).astype(int)
    else:
        df["heavy_rain"] = 0

    if "wind_speed_10m" in df.columns:
        df["strong_wind"] = (df["wind_speed_10m"] >= 20).astype(int)
    else:
        df["strong_wind"] = 0

    df["bad_weather_flag"] = (
        df["heavy_rain"].fillna(0)
        | df["strong_wind"].fillna(0)
        | df["is_cold"].fillna(0)
    ).astype(int)

    if "lat" in df.columns and "lon" in df.columns:
        df["dist_center_proxy"] = np.sqrt(
            (df["lat"] - BARCELONA_CENTER_LAT) ** 2 +
            (df["lon"] - BARCELONA_CENTER_LON) ** 2
        )
    else:
        df["dist_center_proxy"] = np.nan

    if "capacity" in df.columns and "mechanical" in df.columns and "ebike" in df.columns:
        bikes_total = df["mechanical"].fillna(0) + df["ebike"].fillna(0)
        df["pct_bikes_available"] = safe_ratio(bikes_total, df["capacity"]).fillna(0)
    else:
        df["pct_bikes_available"] = np.nan

    logger.info(f"Planning-safe base feature engineering complete. Shape={df.shape}")
    return df


# =========================
# Historical features
# =========================
def add_planning_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating planning historical features...")

    df = df.copy()
    df = df.sort_values(["station_id", "snapshot_ts"]).reset_index(drop=True)

    mech_station_sum = df.groupby("station_id")["mechanical"].cumsum() - df["mechanical"]
    ebike_station_sum = df.groupby("station_id")["ebike"].cumsum() - df["ebike"]
    station_count = df.groupby("station_id").cumcount()

    df["hist_mech_station"] = mech_station_sum / station_count.replace(0, np.nan)
    df["hist_ebike_station"] = ebike_station_sum / station_count.replace(0, np.nan)

    mech_station_hour_sum = df.groupby(["station_id", "hour"])["mechanical"].cumsum() - df["mechanical"]
    ebike_station_hour_sum = df.groupby(["station_id", "hour"])["ebike"].cumsum() - df["ebike"]
    station_hour_count = df.groupby(["station_id", "hour"]).cumcount()

    df["hist_mech_station_hour"] = mech_station_hour_sum / station_hour_count.replace(0, np.nan)
    df["hist_ebike_station_hour"] = ebike_station_hour_sum / station_hour_count.replace(0, np.nan)

    mech_station_hour_dow_sum = (
        df.groupby(["station_id", "hour", "day_of_week"])["mechanical"].cumsum() - df["mechanical"]
    )
    ebike_station_hour_dow_sum = (
        df.groupby(["station_id", "hour", "day_of_week"])["ebike"].cumsum() - df["ebike"]
    )
    station_hour_dow_count = df.groupby(["station_id", "hour", "day_of_week"]).cumcount()

    df["hist_mech_station_hour_dow"] = mech_station_hour_dow_sum / station_hour_dow_count.replace(0, np.nan)
    df["hist_ebike_station_hour_dow"] = ebike_station_hour_dow_sum / station_hour_dow_count.replace(0, np.nan)

    mech_station_weekend_hour_sum = (
        df.groupby(["station_id", "is_weekend", "hour"])["mechanical"].cumsum() - df["mechanical"]
    )
    ebike_station_weekend_hour_sum = (
        df.groupby(["station_id", "is_weekend", "hour"])["ebike"].cumsum() - df["ebike"]
    )
    station_weekend_hour_count = df.groupby(["station_id", "is_weekend", "hour"]).cumcount()

    df["hist_mech_station_weekend_hour"] = (
        mech_station_weekend_hour_sum / station_weekend_hour_count.replace(0, np.nan)
    )
    df["hist_ebike_station_weekend_hour"] = (
        ebike_station_weekend_hour_sum / station_weekend_hour_count.replace(0, np.nan)
    )

    if "pct_bikes_available" in df.columns:
        pct_bikes_station_hour_sum = (
            df.groupby(["station_id", "hour"])["pct_bikes_available"].cumsum() - df["pct_bikes_available"]
        )
        pct_bikes_station_hour_count = df.groupby(["station_id", "hour"]).cumcount()
        df["hist_pct_bikes_available_station_hour"] = (
            pct_bikes_station_hour_sum / pct_bikes_station_hour_count.replace(0, np.nan)
        )
    else:
        df["hist_pct_bikes_available_station_hour"] = np.nan

    df["hist_std_mech_station_hour"] = (
        df.groupby(["station_id", "hour"])["mechanical"]
          .transform(lambda s: s.expanding().std(ddof=0).shift(1))
    )

    df["_is_commute_peak"] = (df["is_morning_peak"] == 1).astype(int)
    df["_mech_peak_value"] = df["mechanical"] * df["_is_commute_peak"]
    df["_ebike_peak_value"] = df["ebike"] * df["_is_commute_peak"]

    mech_peak_sum = df.groupby("station_id")["_mech_peak_value"].cumsum() - df["_mech_peak_value"]
    ebike_peak_sum = df.groupby("station_id")["_ebike_peak_value"].cumsum() - df["_ebike_peak_value"]
    peak_count = df.groupby("station_id")["_is_commute_peak"].cumsum() - df["_is_commute_peak"]

    hist_mech_peak_mean = mech_peak_sum / peak_count.replace(0, np.nan)
    hist_ebike_peak_mean = ebike_peak_sum / peak_count.replace(0, np.nan)

    df["hist_mech_peak_ratio_station"] = hist_mech_peak_mean / df["hist_mech_station"].replace(0, np.nan)
    df["hist_ebike_peak_ratio_station"] = hist_ebike_peak_mean / df["hist_ebike_station"].replace(0, np.nan)

    df = df.drop(columns=["_is_commute_peak", "_mech_peak_value", "_ebike_peak_value"])

    logger.info(f"Planning historical features complete. Shape={df.shape}")
    return df


# =========================
# Final selection
# =========================
def build_feature_sets(df: pd.DataFrame):
    base_cols = ["station_id", "scrapeid", "snapshot_ts", MECHANICAL_TARGET, EBIKE_TARGET]

    all_needed = list(dict.fromkeys(
        base_cols + COMMON_FEATURES + MECHANICAL_ONLY_FEATURES + EBIKE_ONLY_FEATURES
    ))
    all_needed = [c for c in all_needed if c in df.columns]

    mechanical_cols = list(dict.fromkeys(
        base_cols + COMMON_FEATURES + MECHANICAL_ONLY_FEATURES
    ))
    mechanical_cols = [c for c in mechanical_cols if c in df.columns]

    ebike_cols = list(dict.fromkeys(
        base_cols + COMMON_FEATURES + EBIKE_ONLY_FEATURES
    ))
    ebike_cols = [c for c in ebike_cols if c in df.columns]

    df_all = df[all_needed].copy()
    df_mechanical = df[mechanical_cols].copy()
    df_ebike = df[ebike_cols].copy()

    required_all = [c for c in COMMON_FEATURES if c in df_all.columns]
    required_mech = [c for c in COMMON_FEATURES + MECHANICAL_ONLY_FEATURES if c in df_mechanical.columns]
    required_ebike = [c for c in COMMON_FEATURES + EBIKE_ONLY_FEATURES if c in df_ebike.columns]

    df_all = df_all.dropna(subset=required_all + [MECHANICAL_TARGET, EBIKE_TARGET]).reset_index(drop=True)
    df_mechanical = df_mechanical.dropna(subset=required_mech + [MECHANICAL_TARGET]).reset_index(drop=True)
    df_ebike = df_ebike.dropna(subset=required_ebike + [EBIKE_TARGET]).reset_index(drop=True)

    return df_all, df_mechanical, df_ebike


# =========================
# Main
# =========================
def main():
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    holiday_dates = load_holidays(HOLIDAYS_PATH)

    df_info = load_from_parquet(INFO_PATH)
    df_status = load_from_parquet(STATUS_PATH)

    if df_info is None or df_status is None:
        logger.error("Failed to load one or both bike datasets.")
        return

    df_info_clean = clean_info(df_info)
    df_status_clean = clean_status(df_status)

    df_base = join_info_status(df_info_clean, df_status_clean)
    save_to_parquet(df_base, OUTPUT_BASE_PARQUET)
    print("\nMissing info fields in df_base:")
    for col in ["capacity", "lat", "lon", "altitude"]:
        if col in df_base.columns:
            print(col, "missing =", df_base[col].isna().sum())

    # ============================================
    # AUTO WEATHER FETCH
    # ============================================
    bike_min_ts = pd.to_datetime(df_base["snapshot_ts"], utc=True).min()
    bike_max_ts = pd.to_datetime(df_base["snapshot_ts"], utc=True).max()

    logger.info(f"Bike coverage -> MIN={bike_min_ts} | MAX={bike_max_ts}")

    df_weather = fetch_weather_dataset_auto(
        start_ts=bike_min_ts,
        end_ts=bike_max_ts,
        latitude=BARCELONA_CENTER_LAT,
        longitude=BARCELONA_CENTER_LON,
        output_path=WEATHER_PATH,
    )

    df_final = merge_weather(df_base, df_weather)

    # Defensive fill in case the weather API misses a few boundary hours
    weather_fill_cols = [
        "temperature_2m",
        "rain_mm",
        "wind_speed_10m",
        "cloud_cover",
        "sunshine_hours",
        "is_raining",
    ]
    for col in weather_fill_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].ffill().bfill()

    logger.info("Missing weather counts after fill:")
    for col in weather_fill_cols:
        if col in df_final.columns:
            logger.info(f"{col}: {df_final[col].isna().sum()}")

    df_final = add_planning_base_features(df_final, holiday_dates)
    df_final = add_planning_historical_features(df_final)

    df_all, df_mechanical, df_ebike = build_feature_sets(df_final)

    save_to_parquet(df_all, OUTPUT_FEATURES_PARQUET)
    save_to_parquet(df_mechanical, OUTPUT_MECHANICAL_FEATURES_PARQUET)
    save_to_parquet(df_ebike, OUTPUT_EBIKE_FEATURES_PARQUET)

    logger.info(f"All selected features saved to: {OUTPUT_FEATURES_PARQUET}")
    logger.info(f"Mechanical feature dataset saved to: {OUTPUT_MECHANICAL_FEATURES_PARQUET}")
    logger.info(f"Ebike feature dataset saved to: {OUTPUT_EBIKE_FEATURES_PARQUET}")

    print("\nDate coverage diagnostics")
    print("df_status_clean max:", df_status_clean["last_reported"].max())
    print("df_base max:", df_base["snapshot_ts"].max())
    print("df_weather max:", df_weather["weather_ts"].max())
    print("df_all max:", df_all["snapshot_ts"].max())

    print("\nAll selected features shape:")
    print(df_all.shape)

    print("\nMechanical feature dataset shape:")
    print(df_mechanical.shape)

    print("\nEbike feature dataset shape:")
    print(df_ebike.shape)

    print("\nAll selected feature columns:")
    print(df_all.columns.tolist())


if __name__ == "__main__":
    main()

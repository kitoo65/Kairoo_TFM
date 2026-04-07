r"""
streamlit_app_with_nearest_station_map_fixed.py

Enhanced Streamlit app for your Bicing model.

What this version adds
----------------------
1) Keeps the original station-id based prediction workflow
2) Adds an address search box for users who do not know the station ID
3) Geocodes the address with Nominatim via geopy
4) Suggests the closest stations on an interactive map
5) Lets the user choose one of the suggested station IDs, then run prediction
6) Fixes Streamlit session-state handling when applying a suggested station

Required extra packages
-----------------------
pip install geopy folium streamlit-folium
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import folium
import pandas as pd
import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

from predict_pycharm_ready import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_FEATURES_PATH,
    HORIZON_MINUTES,
    compute_anchor_timestamp,
    predict_station_bikes,
)


MADRID_TZ = ZoneInfo("Europe/Madrid")
FORECAST_HORIZON_HOURS = HORIZON_MINUTES // 60


# =========================================================
# HELPERS
# =========================================================
def clamp_naive_madrid_wall_dt(value: datetime, lo: datetime, hi: datetime) -> datetime:
    vn = value.replace(tzinfo=None) if value.tzinfo else value
    if vn < lo:
        return lo
    if vn > hi:
        return hi
    return vn


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


@st.cache_data
def load_station_reference(features_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(features_path, columns=["station_id", "lat", "lon"])
    df = df.dropna(subset=["station_id", "lat", "lon"]).copy()
    df["station_id"] = df["station_id"].astype(int)

    station_ref = (
        df.groupby("station_id", as_index=False)[["lat", "lon"]]
        .first()
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    return station_ref


@st.cache_data
def load_station_ids(features_path: str | Path) -> list[int]:
    station_ref = load_station_reference(features_path)
    return station_ref["station_id"].astype(int).tolist()


@st.cache_data
def load_features_snapshot_bounds(features_path: str | Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_parquet(features_path, columns=["snapshot_ts"])
    ts = pd.to_datetime(df["snapshot_ts"], utc=True)
    return ts.min(), ts.max()


def format_ts_madrid(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MADRID_TZ).strftime("%Y-%m-%d %H:%M %Z (Europe/Madrid)")


def format_ts_madrid_compact(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MADRID_TZ).strftime("%Y-%m-%d %H:%M %Z")


def friendly_window_message(max_target_display: str) -> str:
    return (
        "This request is outside the prediction window supported by the current trained data. "
        f"Please choose a datetime on or before {max_target_display}."
    )


@st.cache_data(show_spinner=False)
def geocode_address(address: str) -> tuple[float, float, str, str] | None:
    if not address or not address.strip():
        return None

    geolocator = Nominatim(user_agent="bicing_station_locator_app")
    raw = address.strip()

    candidate_queries = [
        raw,
        f"{raw}, Barcelona",
        f"{raw}, Barcelona, Spain",
    ]

    for query in candidate_queries:
        try:
            location = geolocator.geocode(
                query,
                timeout=10,
                country_codes="es",
                exactly_one=True,
            )
            if location is not None:
                return float(location.latitude), float(location.longitude), str(location.address), query
        except (GeocoderServiceError, GeocoderTimedOut):
            continue

    return None


def find_nearest_stations(
    station_ref: pd.DataFrame,
    address_lat: float,
    address_lon: float,
    top_n: int = 5,
) -> pd.DataFrame:
    nearest = station_ref.copy()
    nearest["distance_km"] = nearest.apply(
        lambda row: haversine_distance_km(address_lat, address_lon, row["lat"], row["lon"]),
        axis=1,
    )
    nearest = nearest.sort_values(["distance_km", "station_id"]).head(top_n).reset_index(drop=True)
    nearest["distance_m"] = (nearest["distance_km"] * 1000).round(0).astype(int)
    return nearest


def build_station_map(
    address_lat: float,
    address_lon: float,
    address_label: str,
    nearest_df: pd.DataFrame,
) -> folium.Map:
    fmap = folium.Map(location=[address_lat, address_lon], zoom_start=15)

    folium.Marker(
        location=[address_lat, address_lon],
        popup=folium.Popup(f"Address: {address_label}", max_width=300),
        tooltip="Provided address",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(fmap)

    for idx, row in nearest_df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=(
                f"Station ID: {int(row['station_id'])}<br>"
                f"Distance: {int(row['distance_m'])} m"
            ),
            tooltip=f"#{idx + 1} - Station {int(row['station_id'])}",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(fmap)

    return fmap


# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(
    page_title="Bicing Bike Predictor",
    page_icon="🚲",
    layout="wide",
)

st.title("Bicing Bike Predictor")
st.caption("Prediction app with station ID selection or nearest-station search from an address.")

with st.expander("How this app works", expanded=False):
    st.markdown(
        """
        This app predicts bike availability for a chosen target datetime.

        The model predicts at **t+24h**, so the app:
        1. converts the selected Barcelona datetime to UTC
        2. looks for the engineered feature row around **target time - 24h**
        3. predicts:
           - **total bikes**
           - **ebike share**
        4. reconstructs:
           - **ebikes = total × share**
           - **mechanical = total - ebikes**

        If the user does not know the station ID, the app can geocode an address,
        show the nearest stations on a map, and let the user choose one.
        """
    )

with st.sidebar:
    st.header("Paths")
    bundle_path = st.text_input("Bundle path", value=str(DEFAULT_BUNDLE_PATH))
    features_path = st.text_input("Features parquet path", value=str(DEFAULT_FEATURES_PATH))
    max_lookup_minutes = st.number_input(
        "Max lookup gap (minutes)",
        min_value=1,
        max_value=180,
        value=30,
        step=1,
    )
    nearest_station_count = st.number_input(
        "Number of nearest stations to suggest",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    st.header("Help")
    st.caption("If a file is not found, update the path here.")
    st.caption("If no close anchor row is found, try increasing the lookup gap.")
    st.caption("Address search uses Nominatim geocoding and needs internet access at runtime.")


try:
    station_ref = load_station_reference(features_path)
    station_ids = station_ref["station_id"].astype(int).tolist()
    min_snap_utc, max_snap_utc = load_features_snapshot_bounds(features_path)
except Exception as e:
    st.error(f"Unable to load station reference from the features parquet:\n{e}")
    st.stop()

if not station_ids:
    st.error("No station IDs were found in the features parquet.")
    st.stop()

if pd.isna(min_snap_utc) or pd.isna(max_snap_utc):
    st.error("Could not determine min/max snapshot_ts from the features parquet.")
    st.stop()

min_target_utc = min_snap_utc + pd.Timedelta(minutes=HORIZON_MINUTES)
max_target_utc = max_snap_utc + pd.Timedelta(minutes=HORIZON_MINUTES)
max_target_display = format_ts_madrid(max_target_utc)
min_target_display = format_ts_madrid(min_target_utc)
min_snap_display = format_ts_madrid(min_snap_utc)
max_snap_display = format_ts_madrid(max_snap_utc)


default_station = 369 if 369 in station_ids else station_ids[0]

if "station_id_input" not in st.session_state:
    st.session_state.station_id_input = 369

if "pending_station_id" not in st.session_state:
    st.session_state.pending_station_id = None

if st.session_state.pending_station_id is not None:
    st.session_state.station_id_input = int(st.session_state.pending_station_id)
    st.session_state.pending_station_id = None

if st.session_state.station_id_input not in station_ids:
    st.session_state.station_id_input = int(default_station)

if "selected_station_id" not in st.session_state:
    st.session_state.selected_station_id = int(st.session_state.station_id_input)
elif st.session_state.selected_station_id not in station_ids:
    st.session_state.selected_station_id = int(st.session_state.station_id_input)


# =========================================================
# TOP INPUTS: STATION ID + ADDRESS
# =========================================================
input_col1, input_col2 = st.columns(2)

_sid_min = int(min(station_ids))
_sid_max = int(max(station_ids))

with input_col1:
    st.subheader("Select a station ID")
    st.number_input(
        "Station ID",
        min_value=_sid_min,
        max_value=_sid_max,
        step=1,
        key="station_id_input",
    )
    st.session_state.selected_station_id = int(st.session_state.station_id_input)

with input_col2:
    st.subheader("Or search from an address")
    address_input = st.text_input(
        "Address in Barcelona",
        placeholder="Example: Carrer de Mallorca 401, Barcelona",
    )
    search_clicked = st.button("Find closest stations", width="stretch")


# =========================================================
# ADDRESS SEARCH + MAP
# =========================================================
if search_clicked:
    if not address_input.strip():
        st.warning("Please enter an address before searching.")
    else:
        try:
            geocoded = geocode_address(address_input.strip())
            if geocoded is None:
                st.error("Address not found. Please try a more precise address in Barcelona.")
            else:
                address_lat, address_lon, resolved_address, used_query = geocoded
                nearest_df = find_nearest_stations(
                    station_ref=station_ref,
                    address_lat=address_lat,
                    address_lon=address_lon,
                    top_n=int(nearest_station_count),
                )

                st.session_state["nearest_df"] = nearest_df
                st.session_state["resolved_address"] = resolved_address
                st.session_state["used_query"] = used_query
                st.session_state["address_lat"] = address_lat
                st.session_state["address_lon"] = address_lon

        except (GeocoderServiceError, GeocoderTimedOut) as e:
            st.error(f"Geocoding service error: {e}")
        except Exception as e:
            st.error(f"Address search failed:\n{e}")


if "nearest_df" in st.session_state:
    st.subheader("Closest stations from the provided address")
    st.write(f"Resolved address: {st.session_state['resolved_address']}")
    st.caption(f"Query used: {st.session_state['used_query']}")

    nearest_df = st.session_state["nearest_df"].copy()

    c_map, c_pick = st.columns([2.2, 1.0])

    with c_map:
        station_map = build_station_map(
            address_lat=float(st.session_state["address_lat"]),
            address_lon=float(st.session_state["address_lon"]),
            address_label=str(st.session_state["resolved_address"]),
            nearest_df=nearest_df,
        )
        st_folium(station_map, width=None, height=500)

    with c_pick:
        st.markdown("**Nearest station suggestions**")
        display_df = nearest_df[["station_id", "distance_m"]].rename(
            columns={"station_id": "Station ID", "distance_m": "Distance (m)"}
        )
        st.dataframe(display_df, width="stretch", hide_index=True)

        suggested_station = st.selectbox(
            "Use one of these stations",
            options=nearest_df["station_id"].astype(int).tolist(),
            format_func=lambda x: f"Station {x}",
            key="nearest_station_selectbox",
        )

        if st.button("Use this suggested station", width="stretch"):
            st.session_state.pending_station_id = int(suggested_station)
            st.rerun()


# =========================================================
# PREDICTION INPUTS
# =========================================================
st.divider()

st.info(
    f"**Available prediction window ends at:** {max_target_display}\n\n"
    f"Model horizon: **{FORECAST_HORIZON_HOURS} h** (t+24h bundle). "
    f"Feature coverage: **{min_snap_display}** → **{max_snap_display}** (snapshot timestamps, Europe/Madrid)."
)

b1, b2, b3 = st.columns(3)
b1.metric("Min snapshot (Europe/Madrid)", format_ts_madrid_compact(min_snap_utc))
b2.metric("Max snapshot (Europe/Madrid)", format_ts_madrid_compact(max_snap_utc))
b3.metric("Latest allowed target", format_ts_madrid_compact(max_target_utc))

# Pick limits (Europe/Madrid wall clock, naive): min = earliest snapshot in data; max = latest allowed target.
min_dt = min_snap_utc.tz_convert(MADRID_TZ).to_pydatetime().replace(tzinfo=None)
max_dt = max_target_utc.tz_convert(MADRID_TZ).to_pydatetime().replace(tzinfo=None)
min_date = min_dt.date()
max_date = max_dt.date()

now_madrid_wall = datetime.now(MADRID_TZ).replace(tzinfo=None)
if now_madrid_wall > max_dt:
    clamped_default = max_dt
elif now_madrid_wall < min_dt:
    clamped_default = min_dt
else:
    floored_now = (
        pd.Timestamp(datetime.now(MADRID_TZ)).floor("5min").to_pydatetime().replace(tzinfo=None)
    )
    clamped_default = clamp_naive_madrid_wall_dt(floored_now, min_dt, max_dt)

_date_key = "target_pred_date_madrid"
_time_key = "target_pred_time_madrid"
_bounds_id = (str(features_path), str(min_snap_utc), str(max_snap_utc))
if st.session_state.get("prediction_bounds_id") != _bounds_id:
    st.session_state.prediction_bounds_id = _bounds_id
    st.session_state[_date_key] = clamped_default.date()
    st.session_state[_time_key] = clamped_default.time().replace(microsecond=0)
if _date_key not in st.session_state:
    st.session_state[_date_key] = clamped_default.date()
if _time_key not in st.session_state:
    st.session_state[_time_key] = clamped_default.time().replace(microsecond=0)

_sd = st.session_state[_date_key]
if not isinstance(_sd, date):
    st.session_state[_date_key] = clamped_default.date()
else:
    if _sd < min_date:
        st.session_state[_date_key] = min_date
    elif _sd > max_date:
        st.session_state[_date_key] = max_date

_st = st.session_state[_time_key]
if not isinstance(_st, time):
    st.session_state[_time_key] = clamped_default.time().replace(microsecond=0)

pred_col_date, pred_col_time = st.columns(2)
with pred_col_date:
    selected_date = st.date_input(
        "Target date (Europe/Madrid)",
        min_value=min_date,
        max_value=max_date,
        key=_date_key,
    )
with pred_col_time:
    selected_time = st.time_input(
        "Target time (Europe/Madrid)",
        step=timedelta(minutes=5),
        key=_time_key,
    )

st.markdown(f"**Selected station for prediction:** {int(st.session_state.station_id_input)}")

combined_local = datetime.combine(selected_date, selected_time)
selected_naive = clamp_naive_madrid_wall_dt(combined_local, min_dt, max_dt)
if selected_naive != combined_local:
    st.caption(
        f"Adjusted to the valid range [{min_dt.isoformat(sep=' ', timespec='minutes')}, "
        f"{max_dt.isoformat(sep=' ', timespec='minutes')}] (Europe/Madrid wall time)."
    )

sel_ts = pd.Timestamp(selected_naive).tz_localize(MADRID_TZ)

target_utc_pre = sel_ts.tz_convert("UTC")
anchor_utc_pre = compute_anchor_timestamp(target_utc_pre)
anchor_madrid_str = format_ts_madrid(anchor_utc_pre)

st.markdown(f"**Anchor timestamp used for prediction:** {anchor_madrid_str}")

_window_exceeded = bool(target_utc_pre > max_target_utc or target_utc_pre < min_target_utc)

if _window_exceeded:
    if target_utc_pre > max_target_utc:
        st.error(friendly_window_message(max_target_display))
    else:
        st.error(
            "This target is earlier than the first time the model can predict with the current data. "
            f"Please choose a datetime on or after {min_target_display}."
        )

predict_clicked = st.button(
    "Predict",
    type="primary",
    width="stretch",
    disabled=_window_exceeded,
)

if predict_clicked and not _window_exceeded:
    try:
        result = predict_station_bikes(
            station_id=int(st.session_state.station_id_input),
            target_date=sel_ts.strftime("%Y-%m-%d"),
            target_time=sel_ts.strftime("%H:%M"),
            bundle_path=bundle_path,
            features_path=features_path,
            max_lookup_minutes=int(max_lookup_minutes),
        )

        st.success("Prediction completed successfully.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mechanical", result.predicted_mechanical)
        c2.metric("Ebikes", result.predicted_ebike)
        c3.metric("Total bikes", result.predicted_total_bikes)

        st.subheader("Prediction details")
        anchor_pd = pd.Timestamp(result.anchor_utc)
        matched_pd = pd.Timestamp(result.matched_snapshot_utc)
        details_df = pd.DataFrame(
            [
                ["Station ID", str(result.station_id)],
                ["Target datetime (Europe/Madrid)", str(result.target_local)],
                ["Anchor timestamp (Europe/Madrid)", str(format_ts_madrid(anchor_pd))],
                ["Matched snapshot (Europe/Madrid)", str(format_ts_madrid(matched_pd))],
                ["Match gap (minutes)", str(result.matched_delay_minutes)],
                ["Capacity", str(result.capacity)],
                ["Predicted ebike share", f"{result.predicted_ebike_share:.4f}"],
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(details_df, width="stretch", hide_index=True)

        st.info("Physical constraints enforced: 0 ≤ ebikes ≤ total ≤ capacity")

    except FileNotFoundError as e:
        st.error(
            "A required file was not found. Check the bundle and features paths in the sidebar.\n\n"
            f"Detail: {e}"
        )
    except Exception:
        st.error(friendly_window_message(max_target_display))


st.divider()
st.caption(
    "Tip: if the user does not know the station ID, they can search an address and pick one of the closest stations from the map."
)

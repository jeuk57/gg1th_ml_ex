from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from src.charging import add_soc_with_charging, build_greedy_charging_plan
from src.data_loader import load_route_rest_areas
from src.model_utils import load_model_assets, predict_segments
from src.route_utils import create_or_load_segments, downsample_route, load_fixed_route, route_summary

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models/truck_energy_model.pkl"
METADATA_PATH = BASE_DIR / "models/truck_energy_metadata.pkl"
ROUTE_PATH = BASE_DIR / "data/processed/fixed_tmap_route.json"
SEGMENT_PATH = BASE_DIR / "data/processed/fixed_segments.csv"
LOCATION_PATH = BASE_DIR / "data/processed/fixed_locations.json"
REST_PATH = BASE_DIR / "project_data/전국휴게소정보표준데이터.csv"


@st.cache_resource
def load_local_assets():
    """로컬 모델과 고정 경로만 읽으며 외부 API를 호출하지 않는다."""
    model, metadata = load_model_assets(MODEL_PATH, METADATA_PATH)
    route = load_fixed_route(ROUTE_PATH)
    info = route_summary(route)
    segments, _ = create_or_load_segments(route, SEGMENT_PATH)
    return model, metadata, info, segments


@st.cache_data(ttl=3600, show_spinner=False)
def get_weather_at_point(latitude, longitude, timestamp_text):
    """선택한 위치와 시간에서 가장 가까운 예보 기온을 가져온다."""
    timestamp = pd.Timestamp(timestamp_text)
    response = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": latitude, "longitude": longitude, "hourly": "temperature_2m",
        "timezone": "Asia/Seoul", "forecast_days": 16,
    }, timeout=20)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    times = pd.to_datetime(hourly["time"])
    nearest = int(np.argmin(np.abs(times - timestamp)))
    return float(hourly["temperature_2m"][nearest])


def get_segment_temperatures(segments, departure_time):
    """경로의 대표 지점 6곳에서 받은 기온을 전체 구간에 나눠 넣는다."""
    anchor_indices = np.unique(np.linspace(0, len(segments) - 1, 6, dtype=int))
    elapsed = segments["estimated_duration_min"].cumsum()
    distances, temperatures = [], []
    for index in anchor_indices:
        row = segments.iloc[index]
        pass_time = departure_time + timedelta(minutes=float(elapsed.iloc[index]))
        temperatures.append(get_weather_at_point(
            round(float(row["mid_lat"]), 4), round(float(row["mid_lon"]), 4),
            pass_time.isoformat(timespec="minutes")))
        distances.append(float(row["cumulative_distance_km"]))
    return np.interp(segments["cumulative_distance_km"], distances, temperatures)


def prepare_rest_candidates(route_info, segments):
    """경로와 가까운 휴게소를 찾아 가장 가까운 구간 끝에 연결한다."""
    rests = load_route_rest_areas(REST_PATH, route_info["coordinates"])
    if rests.empty:
        return rests
    scale = route_info["total_distance_km"] / route_info["geometry_distance_km"]
    route_positions = rests["route_position_raw_km"] * scale
    ends = segments["end_km"].to_numpy()
    rests["route_position_km"] = route_positions.apply(
        lambda value: float(ends[np.argmin(np.abs(ends - value))]))
    # 같은 Segment 위치의 반대 방향 후보는 경로선에 더 가까운 한 곳만 남긴다.
    return (rests.sort_values("distance_to_route_km").drop_duplicates("route_position_km")
            .sort_values("route_position_km").reset_index(drop=True))


def make_map(route_info, locations, plan):
    """고정 경로와 출발지·도착지·추천 휴게소를 지도에 표시한다."""
    points = [
        {"name": locations["start"]["name"], "lat": locations["start"]["lat"],
         "lon": locations["start"]["lon"], "color": [30, 120, 255]},
        {"name": locations["end"]["name"], "lat": locations["end"]["lat"],
         "lon": locations["end"]["lon"], "color": [220, 60, 60]},
    ]
    if plan is not None and not plan["stops"].empty:
        points += [{"name": row["휴게소"], "lat": row["위도"], "lon": row["경도"],
                    "color": [20, 170, 80]} for _, row in plan["stops"].iterrows()]
    return pdk.Deck(map_style=None,
        initial_view_state=pdk.ViewState(latitude=36.3, longitude=127.6, zoom=6.3),
        layers=[
            pdk.Layer("PathLayer", [{"path": downsample_route(route_info["coordinates"])}],
                      get_path="path", get_color=[28, 105, 212], width_min_pixels=3),
            pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]",
                      get_fill_color="color", get_radius=4500, pickable=True),
        ], tooltip={"text": "{name}"})


st.set_page_config(page_title="대형 전기 화물차 충전 계획", page_icon="🚛", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    .block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5eaf1; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e9f2; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 3px 12px rgba(34, 60, 90, 0.05);
    }
    [data-testid="stMetricLabel"] { color: #64748b; }
    [data-testid="stMetricValue"] { color: #12355b; font-weight: 700; }
    .hero {
        background: linear-gradient(120deg, #0b3b70, #1261a0); color: white;
        padding: 1.7rem 2rem; border-radius: 18px; margin-bottom: 1.3rem;
        box-shadow: 0 8px 24px rgba(11, 59, 112, 0.16);
    }
    .hero h1 { margin: 0 0 .45rem 0; font-size: 2rem; }
    .hero p { margin: .15rem 0; color: #e6f1fb; }
    .route-badge {
        display: inline-block; margin-top: .75rem; padding: .4rem .8rem;
        background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px; font-weight: 600;
    }
    .summary-card {
        background: white; border: 1px solid #e3e9f2; border-left: 5px solid #19a66f;
        border-radius: 14px; padding: 1.25rem 1.35rem; min-height: 220px;
        box-shadow: 0 3px 12px rgba(34,60,90,.05);
    }
    .summary-card h3 { margin-top: 0; color: #12355b; }
    .summary-card .highlight { color: #087f5b; font-weight: 700; }
    .section-label { color: #12355b; font-size: 1.35rem; font-weight: 700; margin: 1.7rem 0 .7rem; }
    div[data-testid="stFormSubmitButton"] button { min-height: 3rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>🚛 대형 전기 화물차 충전 계획 추천 시스템</h1>
  <p>주행환경 기반 에너지 소비 추정과 최소 충전 정차 시뮬레이션</p>
  <div class="route-badge">부산신항 &nbsp;→&nbsp; 인천국제공항 화물터미널</div>
</div>
""", unsafe_allow_html=True)

try:
    model, metadata, route_info, base_segments = load_local_assets()
    import json
    locations = json.loads(LOCATION_PATH.read_text(encoding="utf-8"))
except Exception as error:
    st.error(f"필수 프로젝트 파일을 불러오지 못했습니다: {error}")
    st.stop()

with st.sidebar:
    st.header("운행 조건 설정")
    st.caption("기본값으로도 바로 시뮬레이션할 수 있습니다.")
    with st.form("simulation_form"):
        st.markdown("#### 기본 입력")
        start_soc = st.slider("출발 SOC (%)", 20, 100, 80)
        load_percent = st.slider("적재 수준 (%)", 0, 100, 50)
        hvac_power_kw = st.slider("HVAC 출력 (kW)", 0.0, 5.0, 2.0, 0.1)
        departure_date = st.date_input("출발 날짜")
        departure_clock = st.time_input("출발 시간")
        with st.expander("고급 입력", expanded=False):
            battery_temp = st.slider("배터리 온도 (°C)",
                float(metadata["feature_min"]["battery_temp_C"]),
                float(metadata["feature_max"]["battery_temp_C"]),
                float(metadata["feature_medians"]["battery_temp_C"]), 0.1)
            style_name = st.selectbox("운전 스타일", ["경제적", "보통", "공격적"], index=1)
            tire_pressure = st.slider("타이어 공기압 (bar)",
                float(metadata["feature_min"]["tire_pressure_bar"]),
                float(metadata["feature_max"]["tire_pressure_bar"]),
                float(metadata["feature_medians"]["tire_pressure_bar"]), 0.1)
            safety_soc = st.slider("안전 SOC (%)", 5, 30, 15)
        with st.expander("날씨·충전 시뮬레이션", expanded=False):
            weather_mode = st.radio("날씨 모드", ["수동 기온 입력", "Open-Meteo 예보 사용"])
            manual_temperature = st.slider("수동/실패 시 기온 (°C)", -10.0, 40.0, 15.0, 0.5)
            simulated_charger_kw = st.slider("MVP 시뮬레이션 충전 출력 (kW)", 50, 350, 200, 10,
                help="휴게소 CSV에 실제 출력이 없어 사용자가 정하는 시뮬레이션 값입니다.")
        submitted = st.form_submit_button("충전 계획 계산", type="primary")

if not submitted:
    intro_left, intro_right = st.columns([1.7, 1], gap="large")
    with intro_left:
        st.markdown('<div class="section-label">고정 운송 경로</div>', unsafe_allow_html=True)
        st.pydeck_chart(make_map(route_info, locations, None), width="stretch", height=480)
    with intro_right:
        st.markdown('<div class="section-label">서비스 안내</div>', unsafe_allow_html=True)
        st.metric("고정 경로 거리", f"{route_info['total_distance_km']:.2f} km")
        st.metric("기본 예상 주행시간", str(timedelta(seconds=int(route_info["total_time_s"]))))
        st.info("왼쪽에서 운행 조건을 설정한 뒤 **충전 계획 계산**을 눌러주세요.")
        st.caption("저장된 TMAP 경로를 사용하므로 경로 API를 다시 호출하지 않습니다.")
    st.stop()

departure_time = datetime.combine(departure_date, departure_clock)
style_value = {"경제적": 0.2, "보통": 0.5, "공격적": 0.8}[style_name]
segments = base_segments.copy()
weather_warning = None
if weather_mode == "Open-Meteo 예보 사용":
    try:
        temperatures = get_segment_temperatures(segments, departure_time)
    except Exception as error:
        temperatures = np.full(len(segments), manual_temperature)
        weather_warning = f"Open-Meteo 실패로 수동 기온 {manual_temperature}°C를 사용했습니다: {error}"
else:
    temperatures = np.full(len(segments), manual_temperature)

try:
    segments = predict_segments(segments, model, metadata, {
        "load_percent": load_percent, "ambient_temp_C": temperatures,
        "hvac_power_kw": hvac_power_kw, "battery_temp_C": battery_temp,
        "driving_style_index": style_value, "tire_pressure_bar": tire_pressure})
    rest_candidates = prepare_rest_candidates(route_info, segments)
    plan, plan_error = build_greedy_charging_plan(
        segments, rest_candidates, start_soc, safety_soc, simulated_charger_kw, departure_time)
    detailed = add_soc_with_charging(segments, start_soc, plan)
except Exception as error:
    st.error(f"시뮬레이션을 계산하지 못했습니다: {error}")
    st.stop()

if weather_warning:
    st.warning(weather_warning)
if plan_error:
    st.warning(plan_error)
total_energy = float(segments["segment_energy_kwh"].sum())
drive_hours = route_info["total_time_hours"]
destination_soc = plan["destination_soc"] if plan else float(detailed["end_soc"].iloc[-1])
charge_count = plan["charge_count"] if plan else 0
charge_hours = plan["charge_hours"] if plan else 0.0
arrival_time = departure_time + timedelta(hours=drive_hours + charge_hours)

recommended_names = (
    " · ".join(plan["stops"]["휴게소"].tolist())
    if plan is not None and not plan["stops"].empty
    else ("직행 가능" if plan is not None else "계획 없음")
)
soc_status = "안전 기준 충족" if destination_soc >= safety_soc else "안전 기준 미달"

st.markdown('<div class="section-label">핵심 추천 결과</div>', unsafe_allow_html=True)
cards = st.columns(5, gap="medium")
cards[0].metric("추천 충전소", recommended_names)
cards[1].metric("충전 횟수", f"{charge_count}회")
cards[2].metric("총 충전시간", f"{charge_hours * 60:.0f}분")
cards[3].metric("도착 예상 SOC", f"{destination_soc:.1f}%", soc_status)
cards[4].metric("예상 에너지 소비", f"{total_energy:.1f} kWh", arrival_time.strftime("도착 %m-%d %H:%M"))

st.markdown('<div class="section-label">고정 경로 기준 충전 계획</div>', unsafe_allow_html=True)
map_column, summary_column = st.columns([1.8, 1], gap="large")
with map_column:
    st.pydeck_chart(make_map(route_info, locations, plan), width="stretch", height=455)
with summary_column:
    if plan is not None:
        summary_text = (
            f"출발 SOC <b>{start_soc}%</b>, 적재수준 <b>{load_percent}%</b> 조건에서 "
            f"<span class='highlight'>{recommended_names}</span> "
            f"{charge_count}회 충전 계획을 추천합니다.<br><br>"
            f"이 계획은 안전 SOC <b>{safety_soc}%</b>를 유지하면서 최소 정차로 "
            f"인천공항 화물터미널에 도착하는 시나리오입니다."
        )
    else:
        summary_text = (
            f"현재 출발 SOC <b>{start_soc}%</b> 조건에서는 안전 SOC <b>{safety_soc}%</b>를 "
            "유지하는 충전 계획을 찾지 못했습니다. 출발 SOC 또는 충전 조건을 확인해 주세요."
        )
    st.markdown(f"""
    <div class="summary-card">
      <h3>추천 결과 요약</h3>
      <p>{summary_text}</p>
      <hr style="border:none;border-top:1px solid #e8edf3;margin:1rem 0;">
      <p><b>경로</b> {route_info['total_distance_km']:.2f} km<br>
      <b>예상 주행</b> {str(timedelta(seconds=int(route_info['total_time_s'])))}<br>
      <b>최종 도착</b> {arrival_time.strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)
    if plan is not None and not plan["stops"].empty:
        first_stop = plan["stops"].iloc[0]
        st.caption(f"📍 {first_stop['휴게소']} · 출발지에서 약 {first_stop['경로 누적거리(km)']:.0f} km")

st.markdown('<div class="section-label">주행 거리별 배터리 잔량(SOC) 변화</div>', unsafe_allow_html=True)
soc_rows = [{"누적거리(km)": 0.0, "SOC(%)": float(start_soc), "상태": "주행"}]
for _, row in detailed.iterrows():
    soc_rows.append({"누적거리(km)": float(row["end_km"]),
                     "SOC(%)": float(row["end_soc_before_charge"]), "상태": "주행"})
    if row["end_soc"] > row["end_soc_before_charge"] + 1e-6:
        soc_rows.append({"누적거리(km)": float(row["end_km"]),
                         "SOC(%)": float(row["end_soc"]), "상태": "충전"})
soc_chart = pd.DataFrame(soc_rows)
base = alt.Chart(soc_chart).encode(
    x=alt.X("누적거리(km):Q", title="누적 주행거리 (km)"),
    y=alt.Y("SOC(%):Q", title="배터리 SOC (%)", scale=alt.Scale(domain=[0, 100])),
)
soc_line = base.mark_line(color="#1769aa", strokeWidth=3).encode(
    tooltip=[alt.Tooltip("누적거리(km):Q", format=".1f"), alt.Tooltip("SOC(%):Q", format=".1f")]
)
safety_line = alt.Chart(pd.DataFrame({"SOC(%)": [safety_soc]})).mark_rule(
    color="#dc3545", strokeDash=[7, 5], strokeWidth=2
).encode(y="SOC(%):Q")
charge_points = base.transform_filter(alt.datum["상태"] == "충전").mark_point(
    color="#13a36d", filled=True, size=140, stroke="white", strokeWidth=2
)
st.altair_chart((soc_line + safety_line + charge_points).properties(height=350), width="stretch")
st.caption(f"빨간 점선은 안전 SOC {safety_soc}% 기준이며, 초록색 지점은 충전 후 SOC입니다.")

tab_plan, tab_segments, tab_model = st.tabs(
    ["충전 계획 상세", "구간별 상세", "모델 정보"]
)
with tab_plan:
    st.markdown("#### 추천 충전 계획")
    if plan is not None and not plan["stops"].empty:
        plan_columns = ["순서", "휴게소", "경로 누적거리(km)", "도착 SOC(%)", "충전 후 SOC(%)",
                        "충전 에너지(kWh)", "충전기 출력(kW)", "충전시간(분)", "도착 시각", "예상 출발시각"]
        st.dataframe(plan["stops"][plan_columns], width="stretch", hide_index=True)
    elif plan is not None:
        st.success("추가 충전 없이 목적지까지 직행할 수 있습니다.")
    else:
        st.warning("현재 입력 조건으로 안전한 충전 계획을 만들지 못했습니다.")
    st.caption("충전시간은 평균 유효출력을 이용한 시뮬레이션 추정값입니다.")

with tab_segments:
    st.markdown("#### 세부 구간 분석")
    columns = ["segment_id", "segment_distance_km", "speed_kmh", "ambient_temp_C",
        "segment_raw_prediction", "truck_kwh_per_100km",
        "segment_energy_kwh", "start_soc", "end_soc"]
    st.dataframe(detailed[columns], width="stretch", hide_index=True)
    with st.expander("Segment 속도 산정 방식"):
        st.write("TMAP 원본 LineString별 거리와 시간을 Segment와 겹치는 비율로 배분해 계산했습니다.")

with tab_model:
    test_metrics = metadata["test_metrics"]
    model_cards = st.columns(4)
    model_cards[0].metric("Model", "XGBoost")
    model_cards[1].metric("Features", metadata["feature_count"])
    model_cards[2].metric("Test R²", f"{test_metrics['R2']:.6f}")
    model_cards[3].metric("RMSE", f"{test_metrics['RMSE']:.6f}")
    st.write(f"**MAE:** {test_metrics['MAE']:.6f}")

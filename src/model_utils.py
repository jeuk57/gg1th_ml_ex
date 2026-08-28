from pathlib import Path

import joblib
import pandas as pd


EXPECTED_FEATURES = [
    "speed_kmh", "load_index", "ambient_temp_C", "hvac_power_kw",
    "road_grade_pct", "battery_temp_C", "driving_style_index",
    "tire_pressure_bar", "trip_distance_km",
]


def load_model_assets(model_path, metadata_path):
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("서비스 모델 또는 metadata 파일이 없습니다.")
    model, metadata = joblib.load(model_path), joblib.load(metadata_path)
    features = metadata.get("feature_names", metadata.get("feature_columns"))
    if features != EXPECTED_FEATURES or metadata.get("feature_count") != 9:
        raise ValueError("서비스 모델 metadata의 9개 Feature 이름·순서가 확정 규격과 다릅니다.")
    if float(metadata.get("simulated_road_grade_pct", -999)) != 1.5:
        raise ValueError("서비스 모델 metadata 설정이 올바르지 않습니다.")
    return model, metadata


def predict_segments(segments, model, metadata, inputs):
    """9개 Feature를 정확한 순서로 만들어 Volvo 상대 보정 소비량을 계산한다."""
    result = segments.copy()
    result["load_index"] = inputs["load_percent"] / 100
    result["ambient_temp_C"] = inputs["ambient_temp_C"]
    result["hvac_power_kw"] = inputs["hvac_power_kw"]
    result["road_grade_pct"] = metadata["simulated_road_grade_pct"]
    result["battery_temp_C"] = inputs["battery_temp_C"]
    result["driving_style_index"] = inputs["driving_style_index"]
    result["tire_pressure_bar"] = inputs["tire_pressure_bar"]
    result["trip_distance_km"] = result["segment_distance_km"]
    features = metadata["feature_names"]
    missing = [name for name in features if name not in result]
    if missing or len(features) != metadata["feature_count"]:
        raise ValueError(f"예측 Feature 불일치: 누락={missing}, 개수={len(features)}")
    prediction_input = result.loc[:, features]
    if list(prediction_input.columns) != features:
        raise ValueError("예측 Feature 순서가 학습 순서와 다릅니다.")
    result["segment_raw_prediction"] = model.predict(prediction_input)
    result["correction_factor"] = result["segment_raw_prediction"] / metadata["reference_raw_prediction"]
    result["truck_kwh_per_100km"] = 110.0 * result["correction_factor"]
    result["segment_energy_kwh"] = result["truck_kwh_per_100km"] * result["segment_distance_km"] / 100
    return result

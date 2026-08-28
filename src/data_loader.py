import numpy as np
import pandas as pd

from .route_utils import haversine_km


def read_csv_with_encoding(path):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSV 인코딩을 읽을 수 없습니다: {path}")


def load_route_rest_areas(rest_path, route_coordinates, max_distance_km=3.0):
    """실제 컬럼을 검증하고 경로 좌표에서 가까운 EV 휴게소를 찾는다."""
    rest = read_csv_with_encoding(rest_path)
    required = {"휴게소명", "도로노선명", "도로노선방향", "위도", "경도", "전기차충전소유무"}
    missing = required - set(rest.columns)
    if missing:
        raise ValueError(f"휴게소 CSV 필수 컬럼이 없습니다: {sorted(missing)}")
    coords = np.asarray(route_coordinates, dtype=float)
    route_leg = haversine_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    route_pos = np.insert(np.cumsum(route_leg), 0, 0.0)
    rows = []
    for _, row in rest.dropna(subset=["위도", "경도"]).iterrows():
        distances = haversine_km(coords[:, 0], coords[:, 1], float(row["경도"]), float(row["위도"]))
        nearest = int(np.argmin(distances))
        if distances[nearest] <= max_distance_km and str(row["전기차충전소유무"]).upper() == "Y":
            item = row.to_dict()
            item.update({
                "distance_to_route_km": float(distances[nearest]),
                "route_position_raw_km": float(route_pos[nearest]),
            })
            rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("route_position_raw_km").reset_index(drop=True)

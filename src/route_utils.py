import json
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lon1, lat1, lon2, lat2):
    """두 WGS84 좌표 사이의 대권거리를 km로 계산한다."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def load_fixed_route(route_path):
    """저장된 TMAP JSON만 읽는다. 이 함수에는 API 호출 코드가 없다."""
    path = Path(route_path)
    if not path.exists():
        raise FileNotFoundError(f"고정 TMAP 경로 파일이 없습니다: {path}")
    route = json.loads(path.read_text(encoding="utf-8"))
    if not route.get("features"):
        raise ValueError("고정 TMAP 경로에 features가 없습니다.")
    return route


def route_summary(route):
    """TMAP 응답의 총 거리·시간과 LineString 좌표를 정리한다."""
    total_distance_m = total_time_s = None
    line_features = []
    coordinates = []
    for feature in route["features"]:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        if total_distance_m is None and props.get("totalDistance") is not None:
            total_distance_m = float(props["totalDistance"])
            total_time_s = float(props["totalTime"])
        if geom.get("type") == "LineString":
            points = geom.get("coordinates", [])
            if not points:
                continue
            line_features.append({
                "coordinates": points,
                "distance_m": float(props.get("distance", 0)),
                "time_s": float(props.get("time", 0)),
            })
            if coordinates and coordinates[-1] == points[0]:
                coordinates.extend(points[1:])
            else:
                coordinates.extend(points)
    if total_distance_m is None or total_time_s is None or len(coordinates) < 2:
        raise ValueError("TMAP JSON에서 거리·시간·경로 좌표를 찾지 못했습니다.")
    return {
        "total_distance_km": total_distance_m / 1000,
        "total_time_s": total_time_s,
        "total_time_hours": total_time_s / 3600,
        "line_count": len(line_features),
        "coordinate_count": sum(len(x["coordinates"]) for x in line_features),
        "coordinates": coordinates,
        "geometry_distance_km": float(np.sum(haversine_km(
            np.asarray(coordinates[:-1])[:, 0], np.asarray(coordinates[:-1])[:, 1],
            np.asarray(coordinates[1:])[:, 0], np.asarray(coordinates[1:])[:, 1]
        ))),
        "line_features": line_features,
    }


def _point_at_distance(coords, cumulative_km, distance_km):
    """누적거리상의 위치를 경로 좌표 사이에서 선형 보간한다."""
    index = int(np.searchsorted(cumulative_km, distance_km, side="right") - 1)
    index = min(max(index, 0), len(coords) - 2)
    start, end = cumulative_km[index], cumulative_km[index + 1]
    ratio = 0 if end == start else (distance_km - start) / (end - start)
    lon = coords[index, 0] + ratio * (coords[index + 1, 0] - coords[index, 0])
    lat = coords[index, 1] + ratio * (coords[index + 1, 1] - coords[index, 1])
    return float(lon), float(lat)


def _duration_between(line_features, start_km, end_km):
    """TMAP 도로별 distance/time과 겹치는 비율로 Segment 시간을 배분한다."""
    duration_s = 0.0
    cursor = 0.0
    for line in line_features:
        line_km = line["distance_m"] / 1000
        line_end = cursor + line_km
        overlap = max(0.0, min(end_km, line_end) - max(start_km, cursor))
        if overlap and line_km > 0:
            duration_s += line["time_s"] * overlap / line_km
        cursor = line_end
    return duration_s


def create_or_load_segments(route, cache_path, max_segment_km=10.0):
    """고정 경로를 약 10km Segment로 만들고 CSV 캐시를 재사용한다."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        required = {"segment_id", "start_km", "end_km", "segment_distance_km", "speed_kmh"}
        if required.issubset(cached.columns):
            return cached, True

    summary = route_summary(route)
    coords = np.asarray(summary["coordinates"], dtype=float)
    leg_km = haversine_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    geometry_cumulative = np.insert(np.cumsum(leg_km), 0, 0.0)
    # 좌표로 계산한 거리와 TMAP 공식 총거리의 작은 차이를 비례 보정한다.
    geometry_cumulative *= summary["total_distance_km"] / geometry_cumulative[-1]
    boundaries = list(np.arange(0, summary["total_distance_km"], max_segment_km))
    boundaries.append(summary["total_distance_km"])
    rows = []
    for segment_id, (start_km, end_km) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        start_lon, start_lat = _point_at_distance(coords, geometry_cumulative, start_km)
        end_lon, end_lat = _point_at_distance(coords, geometry_cumulative, end_km)
        mid_lon, mid_lat = _point_at_distance(coords, geometry_cumulative, (start_km + end_km) / 2)
        duration_s = _duration_between(summary["line_features"], start_km, end_km)
        distance_km = end_km - start_km
        if duration_s <= 0:
            duration_s = distance_km / summary["total_distance_km"] * summary["total_time_s"]
        rows.append({
            "segment_id": segment_id, "start_km": start_km, "end_km": end_km,
            "segment_distance_km": distance_km, "cumulative_distance_km": end_km,
            "start_lat": start_lat, "start_lon": start_lon,
            "end_lat": end_lat, "end_lon": end_lon,
            "mid_lat": mid_lat, "mid_lon": mid_lon,
            "estimated_duration_min": duration_s / 60,
            "speed_kmh": distance_km / (duration_s / 3600),
        })
    segments = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return segments, False


def downsample_route(coordinates, max_points=1200):
    """지도 표시만 가볍게 하며 원본 JSON은 변경하지 않는다."""
    if len(coordinates) <= max_points:
        return coordinates
    indices = np.linspace(0, len(coordinates) - 1, max_points, dtype=int)
    return [coordinates[i] for i in indices]

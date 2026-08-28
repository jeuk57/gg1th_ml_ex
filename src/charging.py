from datetime import timedelta

import pandas as pd


BATTERY_KWH = 460.0
MAX_CHARGING_KW = 350.0


def build_greedy_charging_plan(segments, stops, start_soc, safety_soc,
                               charger_kw, departure_time):
    """안전 SOC를 지키면서 도달 가능한 가장 먼 휴게소를 선택해 정차 수를 줄인다."""
    if charger_kw <= 0:
        return None, "충전기 출력이 확인되지 않았고 시뮬레이션 출력도 입력되지 않았습니다."
    if stops.empty:
        return None, "고정 경로 3km 이내의 EV 충전 가능 휴게소를 찾지 못했습니다."
    effective_kw = min(float(charger_kw), MAX_CHARGING_KW)
    total_km = float(segments["end_km"].max())
    checkpoints = stops.sort_values("route_position_km").to_dict("records")
    checkpoints.append({"휴게소명": "목적지", "route_position_km": total_km})
    current_km, current_energy = 0.0, BATTERY_KWH * start_soc / 100
    elapsed_drive_h = elapsed_charge_h = 0.0
    details, soc_points = [], [{"distance_km": 0.0, "soc": start_soc, "event": "출발"}]

    def energy_between(a, b):
        mask = (segments["end_km"] > a + 1e-9) & (segments["end_km"] <= b + 1e-6)
        return float(segments.loc[mask, "segment_energy_kwh"].sum())

    while current_km < total_km - 1e-6:
        reachable = []
        for cp in checkpoints:
            pos = float(cp["route_position_km"])
            if pos <= current_km + 1e-6:
                continue
            remaining = current_energy - energy_between(current_km, pos)
            if remaining / BATTERY_KWH * 100 >= safety_soc:
                reachable.append((cp, remaining))
        if not reachable:
            return None, "현재 SOC에서 안전 SOC를 유지하며 다음 충전 지점까지 갈 수 없습니다."
        chosen, arrival_energy = reachable[-1]
        position = float(chosen["route_position_km"])
        mask = (segments["end_km"] > current_km + 1e-9) & (segments["end_km"] <= position + 1e-6)
        elapsed_drive_h += float(segments.loc[mask, "estimated_duration_min"].sum()) / 60
        arrival_soc = arrival_energy / BATTERY_KWH * 100
        soc_points.append({"distance_km": position, "soc": arrival_soc, "event": "도착"})
        current_energy = arrival_energy
        current_km = position
        if chosen["휴게소명"] == "목적지":
            break

        later = [cp for cp in checkpoints if float(cp["route_position_km"]) > current_km + 1e-6]
        # 가장 먼 다음 지점까지 안전 SOC로 도달할 최소 충전량을 탐색한다.
        target = later[0]
        for cp in later:
            required = energy_between(current_km, float(cp["route_position_km"])) + BATTERY_KWH * safety_soc / 100
            if required <= BATTERY_KWH:
                target = cp
            else:
                break
        required_energy = energy_between(current_km, float(target["route_position_km"])) + BATTERY_KWH * safety_soc / 100
        charge_energy = max(0.0, required_energy - current_energy)
        charge_hours = charge_energy / effective_kw
        after_soc = (current_energy + charge_energy) / BATTERY_KWH * 100
        arrival_time = departure_time + timedelta(hours=elapsed_drive_h + elapsed_charge_h)
        elapsed_charge_h += charge_hours
        details.append({
            "순서": len(details) + 1, "휴게소": chosen["휴게소명"],
            "노선명": chosen.get("도로노선명", ""), "방향": chosen.get("도로노선방향", ""),
            "경로 누적거리(km)": current_km, "도착 SOC(%)": arrival_soc,
            "충전 후 SOC(%)": after_soc, "충전 에너지(kWh)": charge_energy,
            "충전기 출력(kW)": effective_kw, "출력 구분": "사용자 지정 MVP 시뮬레이션 값",
            "충전시간(분)": charge_hours * 60, "도착 시각": arrival_time,
            "예상 출발시각": arrival_time + timedelta(hours=charge_hours),
            "위도": chosen.get("위도"), "경도": chosen.get("경도"),
        })
        current_energy += charge_energy
        soc_points.append({"distance_km": current_km, "soc": after_soc, "event": "충전 후"})

    plan = {
        "feasible": True, "charge_count": len(details),
        "charge_hours": elapsed_charge_h, "drive_hours": elapsed_drive_h,
        "total_hours": elapsed_drive_h + elapsed_charge_h,
        "destination_soc": current_energy / BATTERY_KWH * 100,
        "stops": pd.DataFrame(details), "soc_points": pd.DataFrame(soc_points),
    }
    return plan, None


def add_soc_with_charging(segments, start_soc, plan):
    """추천 충전을 Segment 종료지점에 반영해 상세 SOC 표를 만든다."""
    result = segments.copy()
    charges = {}
    if plan is not None and not plan["stops"].empty:
        charges = dict(zip(
            plan["stops"]["경로 누적거리(km)"].round(6),
            plan["stops"]["충전 에너지(kWh)"],
        ))
    energy = BATTERY_KWH * start_soc / 100
    cumulative, starts, arrivals, afters = 0.0, [], [], []
    for _, row in result.iterrows():
        starts.append(energy / BATTERY_KWH * 100)
        used = float(row["segment_energy_kwh"])
        cumulative += used
        energy -= used
        arrivals.append(energy / BATTERY_KWH * 100)
        energy += float(charges.get(round(float(row["end_km"]), 6), 0.0))
        afters.append(energy / BATTERY_KWH * 100)
    result["start_soc"] = starts
    result["end_soc_before_charge"] = arrivals
    result["end_soc"] = afters
    result["cumulative_energy_kwh"] = result["segment_energy_kwh"].cumsum()
    return result

# 대형 전기 화물차 충전 계획 추천 시스템

부산신항에서 인천국제공항 화물터미널까지의 고정 경로를 구간별로 분석하고 충전 계획을 추천하는 Streamlit 앱입니다.

## 실행

```bash
cd /home/aiuk/ml_ex
uv run streamlit run app.py
```

## 실행에 필요한 파일

- `models/truck_energy_model.pkl`
- `models/truck_energy_metadata.pkl`
- `data/processed/fixed_tmap_route.json`
- `data/processed/fixed_locations.json`
- `project_data/전국휴게소정보표준데이터.csv`

`data/processed/fixed_segments.csv`는 실행 속도를 위한 캐시이며, 없으면 고정 경로에서 다시 생성됩니다.

## 앱 동작

- 저장된 TMAP 경로를 약 10km 구간으로 나눕니다.
- 9 Feature XGBoost 모델로 구간별 에너지 소비량을 예측합니다.
- 구간별 SOC를 계산하고 충전 휴게소를 추천합니다.
- 날씨 기본 모드는 수동 입력입니다.
- Open-Meteo 예보 모드를 선택하고 계산 버튼을 누른 경우에만 날씨 API를 호출합니다.

## 주요 코드

- `app.py`: Streamlit 화면과 전체 실행 흐름
- `src/route_utils.py`: 고정 경로와 구간 처리
- `src/model_utils.py`: 모델 로딩과 예측
- `src/data_loader.py`: 휴게소 데이터 로딩
- `src/charging.py`: SOC와 충전 계획 계산

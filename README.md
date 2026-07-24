# 젠트리피케이션 분석

서울 상권별 원자료를 비교해 상업 젠트리피케이션 상대위험도를 보여 주는 Streamlit 대시보드입니다.

## 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

`app.py`를 Streamlit 진입점으로 지정해도 같은 대시보드가 열립니다.

AI 기능을 사용하려면 로컬 환경 변수 `OPENAI_API_KEY`를 설정하거나 `.streamlit/secrets.toml`에 아래 값을 등록합니다. 이 파일과 `api_key.md`는 `.gitignore`에 포함되어 GitHub에 게시되지 않습니다.

```toml
OPENAI_API_KEY = "발급받은 API 키"
```

## Streamlit Community Cloud 배포

- Repository: `sanghyo-net/h1-tec-aicode`
- Branch: `main`
- Main file path: `dashboard.py`
- Advanced settings > Secrets: 위와 같은 `OPENAI_API_KEY` 등록

배포에 필요한 Python 파일, `llm_instructions.md`, `market_input/` CSV가 저장소에 함께 있어야 합니다.

## 상권 입력 자료

상권별 원자료는 프로젝트 루트의 `market_input/` 폴더에 넣습니다. 탭으로 구분된 원자료와 쉼표 CSV를 모두 읽습니다.

| 파일 | 자료 |
|---|---|
| `market_input/market_rent.csv` | 임대가격지수 자료(2021년 1분기~2026년 1분기) |
| `market_input/market_processed.csv` | 전처리 상권 자료 |
| `market_input/market_vacancy.csv` | 공실률 자료 |

첫 번째 행의 열 이름은 삭제하지 않습니다. 상권명은 임대가격지수·공실률 표의 `상권별(3)` 또는 전처리 표의 `상권_코드_명`에서 읽습니다. 임대가격지수는 임대료의 절대 금액이 아니라 기준시점 대비 가격 수준의 변화 지표입니다.

임대가격지수는 2026년 1분기까지 보관하지만, 현재 점포 집계 자료의 2026년 1분기 값은 비어 있으므로 대시보드의 기본 분석 분기는 마지막 완전 공통 분기인 2025년 4분기입니다.

## 상업 젠트리피케이션 상대위험도 결합식

대시보드는 비용 압력·기존 점포 이탈 압력·프랜차이즈 대체 압력을 각각 상권 간 순위환산값(0~100)으로 만든 뒤 다음 곱셈식으로 종합 비교값을 계산합니다.

```text
종합 비교값 = 100 × (비용 압력 / 100) × (이탈 압력 / 100) × (대체 압력 / 100)
```

이 식은 한 압력만 높고 다른 압력이 낮은 상권을 과대평가하지 않기 위한 것입니다. 예를 들어 압력이 `20, 100, 100`이면 종합 비교값은 `20.0`이고, 균형적인 `70, 70, 70`이면 `34.3`입니다. 이 값 자체는 발생 확률이나 절대 위험 점수가 아니며, 모든 상권의 종합 비교값을 큰 순서로 정렬한 상대순위만 화면에 표시합니다.

각 압력은 다음처럼 구성합니다.

- 비용 압력: 임대가격지수 전년 동분기 상승률의 순위환산값
- 기존 점포 이탈 압력: 일반 점포 감소율과 폐업률 변화 순위환산값의 평균
- 프랜차이즈 대체 압력: 프랜차이즈 비율 변화의 순위환산값

AI 채팅에는 대시보드가 계산한 현재 상권별 결과만 전달됩니다. 임대가격지수로는 절대 임대료 수준, 창업 수익성, 적정 임대료를 판단할 수 없습니다.

## 테스트

```bash
python -m unittest -v
```

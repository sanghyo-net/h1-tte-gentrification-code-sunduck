"""전체 화면 대시보드와 우측 질문 패널."""

from __future__ import annotations

import csv
import re
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from llm_client import ask_ai
from model import (
    commercial_risk_product,
    get_previous_period_options,
    period_sort_key,
    predict_numeric_series,
    quarter_code_to_period,
    shift_period,
    target_period_options,
    validate_prediction_periods,
)


MARKET_INPUT_DIR = Path(__file__).with_name("market_input")


def _market_rows(filename: str) -> list[list[str]]:
    """상권 원자료를 탭 또는 쉼표 구분 형식으로 읽습니다."""
    path = MARKET_INPUT_DIR / filename
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return []
    # 헤더는 쉼표이고 데이터 행은 탭인 공공 통계 형식도 지원합니다.
    return [
        next(csv.reader([line], delimiter="\t" if "\t" in line else ","))
        for line in lines
    ]


def market_names() -> list[str]:
    wide_sets: list[set[str]] = []
    for filename in ("market_rent.csv", "market_vacancy.csv"):
        wide_sets.append({row[2].strip() for row in _market_rows(filename)[1:] if len(row) >= 3 and row[2].strip() != "소계"})
    processed = {row[1].strip() for row in _market_rows("market_processed.csv")[1:] if len(row) >= 2 and row[0].strip().isdigit()}
    return sorted(set.intersection(*wide_sets, processed) if wide_sets else set())


def _number(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip()) if value.strip() else None
    except ValueError:
        return None


def _period_from_code(code: str) -> str:
    return quarter_code_to_period(code)


def _processed_records(market: str) -> list[dict[str, str]]:
    rows = _market_rows("market_processed.csv")
    if not rows:
        return []
    header = rows[0]
    records = [dict(zip(header, row)) for row in rows[1:] if len(row) >= 2 and row[0].strip().isdigit() and row[1].strip() == market]
    periods = [record["기준_년분기_코드"] for record in records]
    if len(periods) != len(set(periods)):
        raise ValueError(f"{market} 상권에 같은 분기 자료가 중복되어 있습니다.")
    return records


def _wide_values(filename: str, market: str) -> dict[str, float]:
    rows = _market_rows(filename)
    if not rows:
        return {}
    header = rows[0]
    for row in rows[1:]:
        if len(row) >= 3 and row[2].strip() == market:
            return {header[index]: value for index in range(3, len(header)) if (value := _number(row[index] if index < len(row) else "")) is not None}
    return {}


def _average_rank(values: dict[str, float], descending: bool = False) -> dict[str, float]:
    """동점은 같은 위치들의 산술평균 순위로 처리합니다."""
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=descending)
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + (end + 1)) / 2
        for position in range(index, end + 1):
            ranks[ordered[position][0]] = rank
        index = end + 1
    return ranks


def _market_prediction(
    market: str,
    target_period: str,
    input_periods: list[str],
    available_periods: list[str],
) -> dict[str, Any] | None:
    input_periods = validate_prediction_periods(target_period, input_periods, available_periods)
    rent_index = _wide_values("market_rent.csv", market)
    records = {_period_from_code(row["기준_년분기_코드"]): row for row in _processed_records(market)}
    histories = [(period, records.get(period)) for period in input_periods]
    if not any(row for _, row in histories):
        return None

    def predict_column(column: str) -> float | None:
        return predict_numeric_series(
            [(period, _number(row.get(column, "")) if row else None) for period, row in histories],
            target_period,
        )

    predicted_rent = predict_numeric_series(
        [(period, rent_index.get(period)) for period in input_periods], target_period
    )
    predicted_general = predict_column("일반_점포_수")
    predicted_total = predict_column("총_점포_수(유사업종)")
    predicted_close = predict_column("폐업률_재계산(%)")
    predicted_franchise = predict_column("프랜차이즈비율(%)")
    predicted_turnover = predict_column("점포교체율(%)")
    predicted_open = predict_column("개업률_재계산(%)")
    if any(value is None for value in (predicted_rent, predicted_general, predicted_close, predicted_franchise)):
        return None
    latest_period = max(input_periods, key=period_sort_key)
    latest = records.get(latest_period, {})
    latest_rent = rent_index.get(latest_period)
    latest_general = _number(latest.get("일반_점포_수", ""))
    latest_close = _number(latest.get("폐업률_재계산(%)", ""))
    latest_franchise = _number(latest.get("프랜차이즈비율(%)", ""))
    latest_turnover = _number(latest.get("점포교체율(%)", ""))
    latest_open = _number(latest.get("개업률_재계산(%)", ""))
    if any(value is None for value in (latest_rent, latest_general, latest_close, latest_franchise)):
        return None
    predicted_general = max(0.0, round(predicted_general))
    predicted_total = max(0.0, round(predicted_total)) if predicted_total is not None else None
    predicted_close = max(0.0, predicted_close)
    predicted_franchise = min(100.0, max(0.0, predicted_franchise))
    predicted_turnover = min(100.0, max(0.0, predicted_turnover)) if predicted_turnover is not None else None
    predicted_open = min(100.0, max(0.0, predicted_open)) if predicted_open is not None else None
    actual = records.get(target_period)
    actual_values = None
    if actual:
        actual_values = {
            "general_stores": _number(actual.get("일반_점포_수", "")),
            "close_rate": _number(actual.get("폐업률_재계산(%)", "")),
            "franchise_ratio": _number(actual.get("프랜차이즈비율(%)", "")),
            "store_turnover": _number(actual.get("점포교체율(%)", "")),
            "rent_index": rent_index.get(target_period),
        }
    return {
        "market": market,
        "base_period": latest_period,
        "target_period": target_period,
        "input_periods": input_periods,
        "prediction_method": "최근 관측값 유지" if len(input_periods) == 1 else "선택 기간 선형 추세",
        "rent_index": predicted_rent,
        "predicted_general_stores": predicted_general,
        "predicted_close_rate": predicted_close,
        "predicted_franchise_ratio": predicted_franchise,
        "predicted_store_turnover": predicted_turnover,
        "rent_growth": (predicted_rent - latest_rent) / latest_rent * 100 if latest_rent else 0.0,
        "general_store_decline": (latest_general - predicted_general) / latest_general * 100 if latest_general else 0.0,
        "close_rate_change": predicted_close - latest_close,
        "franchise_ratio_change": predicted_franchise - latest_franchise,
        "store_turnover_change": predicted_turnover - latest_turnover if predicted_turnover is not None and latest_turnover is not None else None,
        "open_rate_change": predicted_open - latest_open if predicted_open is not None and latest_open is not None else None,
        "total_stores": predicted_total,
        "actual_values": actual_values,
        "matching_notice": latest.get("매칭_주의", "").strip(),
    }


def _risk_factor_items(predictions: dict[str, dict[str, Any]]) -> None:
    """원자료 값과 같은 분기 상권 순위로 설명 가능한 위험 근거를 만듭니다."""
    factor_meta = (
        ("rent_growth", "임대가격지수 상승률", "%"),
        ("close_rate_change", "폐업률 변화", "%p"),
        ("store_turnover_change", "점포교체율 변화", "%p"),
        ("franchise_ratio_change", "프랜차이즈비율 변화", "%p"),
    )
    factor_ranks = {
        key: _average_rank(values)
        for key, _, _ in factor_meta
        if len(values := {market: item[key] for market, item in predictions.items() if item.get(key) is not None}) >= 2
    }
    for market, item in predictions.items():
        factors: list[dict[str, Any]] = []
        for key, label, unit in factor_meta:
            if key not in factor_ranks or item.get(key) is None:
                continue
            population = sum(1 for candidate in predictions.values() if candidate.get(key) is not None)
            high_position = (population - factor_ranks[key][market] + 1) / population * 100
            factors.append({
                "label": label,
                "value": item[key],
                "unit": unit,
                "top_percent": high_position,
                "text": f"{label}이(가) 전체 상권 상위 {high_position:.1f}%입니다.",
            })
        item["risk_factors"] = sorted(factors, key=lambda factor: factor["top_percent"])[:3] if len(factors) >= 2 else []


def commercial_risk_results(
    markets: list[str],
    target_period: str,
    input_periods: list[str],
    available_periods: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    available_periods = available_periods or _available_periods(markets)
    input_periods = validate_prediction_periods(target_period, input_periods, available_periods)
    predictions = {
        market: prediction
        for market in markets
        if (prediction := _market_prediction(market, target_period, input_periods, available_periods)) is not None
    }
    metric_keys = ("rent_growth", "general_store_decline", "close_rate_change", "franchise_ratio_change")
    metric_ranks = {key: _average_rank({market: item[key] for market, item in predictions.items()}) for key in metric_keys}
    count = len(predictions)
    for market, item in predictions.items():
        item["cost_pressure"] = metric_ranks["rent_growth"][market] / count * 100
        item["exit_pressure"] = (metric_ranks["general_store_decline"][market] / count * 100 + metric_ranks["close_rate_change"][market] / count * 100) / 2
        item["replacement_pressure"] = metric_ranks["franchise_ratio_change"][market] / count * 100
        item["combined_comparison"] = commercial_risk_product(
            item["cost_pressure"], item["exit_pressure"], item["replacement_pressure"]
        )
    overall_ranks = _average_rank({market: item["combined_comparison"] for market, item in predictions.items()}, descending=True)
    area_ranks = {
        "cost_rank": _average_rank({market: item["cost_pressure"] for market, item in predictions.items()}, descending=True),
        "exit_rank": _average_rank({market: item["exit_pressure"] for market, item in predictions.items()}, descending=True),
        "replacement_rank": _average_rank({market: item["replacement_pressure"] for market, item in predictions.items()}, descending=True),
    }
    for market, item in predictions.items():
        item["comparable_count"] = count
        item["excluded_count"] = len(markets) - count
        item["overall_rank"] = overall_ranks[market]
        item["top_percent"] = overall_ranks[market] / count * 100
        for key, ranks in area_ranks.items():
            item[key] = ranks[market]
    _risk_factor_items(predictions)
    return predictions


def _available_periods(markets: list[str]) -> list[str]:
    periods = {
        _period_from_code(record["기준_년분기_코드"])
        for market in markets
        for record in _processed_records(market)
    }
    return sorted(periods, key=period_sort_key)


def risk_trend(
    markets: list[str], market: str, target_period: str, input_periods: list[str]
) -> dict[str, Any] | None:
    """선택한 예측 결과를 바로 앞 목표 분기의 동일한 최대 시차 결과와 비교합니다."""
    available = _available_periods(markets)
    previous_target = shift_period(target_period, -1)
    previous_inputs = [item["period"] for item in get_previous_period_options(previous_target, available)]
    if previous_target not in target_period_options(available) or not previous_inputs:
        return None
    previous = commercial_risk_results(markets, previous_target, previous_inputs, available).get(market)
    current = commercial_risk_results(markets, target_period, input_periods, available).get(market)
    if not previous or not current:
        return None
    difference = current["combined_comparison"] - previous["combined_comparison"]
    return {
        "previous_period": previous_target,
        "difference": difference,
        "label": "▲ 위험 증가" if difference >= 2 else "▼ 위험 감소" if difference <= -2 else "― 유지",
    }


def latest_common_comparison(markets: list[str], first: str, second: str) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """두 상권 모두에 동일한 공식으로 계산 가능한 가장 최신 분기를 찾습니다."""
    first_periods = {_period_from_code(record["기준_년분기_코드"]) for record in _processed_records(first)}
    second_periods = {_period_from_code(record["기준_년분기_코드"]) for record in _processed_records(second)}
    available = _available_periods(markets)
    for period in sorted(first_periods & second_periods, key=period_sort_key, reverse=True):
        inputs = [item["period"] for item in get_previous_period_options(period, available)]
        if not inputs:
            continue
        results = commercial_risk_results(markets, period, inputs, available)
        if first in results and second in results:
            return period, results[first], results[second]
    return None


st.set_page_config(page_title="젠트리피케이션 분석", page_icon="🏙️", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
      * { box-sizing: border-box; }
      html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: 'Noto Sans KR', sans-serif;
      }
      [data-testid="stAppViewContainer"] { background: #ffffff; color: #333; }
      [data-testid="stHeader"] { background: transparent; }
      .block-container {
        width: 86.667vw !important;
        max-width: none !important;
        height: 100dvh;
        padding: 12px 24px !important;
        padding-bottom: max(72px, calc(env(safe-area-inset-bottom) + 56px)) !important;
        margin: 0 auto !important;
        overflow-x: hidden;
        overflow-y: auto;
        overscroll-behavior-y: contain;
        scrollbar-gutter: stable;
      }
      .app-title {
        color: #4a403c;
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.05em;
        margin: 0 0 10px;
      }
      .section-heading {
        color: #4a403c;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 10px 0 10px;
      }
      [data-testid="stButton"] > button {
        border: 1px solid rgba(201, 130, 105, 0.25);
        border-radius: 999px;
        background: #faf7f2;
        color: #b8745d;
        font-family: 'Noto Sans KR', sans-serif;
        font-weight: 700;
      }
      [data-testid="stButton"] > button:hover {
        background: #c98269;
        border-color: #c98269;
        color: #ffffff;
      }
      .summary-grid, .chance-grid {
        display: grid;
        gap: 10px;
        margin-bottom: 20px;
      }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chance-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .result-box, .info-card, .chance-card {
        border: 1px solid rgba(201, 130, 105, 0.15);
        border-radius: 14px;
        background: #faf7f2;
      }
      .result-box { padding: 17px 12px; text-align: center; }
      .result-label { color: #6b6662; font-size: 0.78rem; font-weight: 700; }
      .result-value { color: #c98269; font-size: 1.55rem; font-weight: 700; margin-top: 5px; }
      .result-note { color: #6b6662; font-size: 0.78rem; margin-top: 3px; }
      .relative-grade {
        border: 1px solid rgba(201, 130, 105, 0.15);
        border-radius: 14px;
        background: #faf7f2;
        padding: 13px 14px 10px;
      }
      .relative-grade-title {
        color: #4a403c;
        font-size: 0.86rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 24px;
      }
      .grade-scale-wrap { position: relative; }
      .grade-scale {
        display: grid;
        grid-template-columns: 10fr 24fr 32fr 24fr 10fr;
        height: 16px;
        overflow: hidden;
        border-radius: 999px;
      }
      .grade-1 { background: #2e7d32; }
      .grade-2 { background: #8bc34a; }
      .grade-3 { background: #fdd835; }
      .grade-4 { background: #fb8c00; }
      .grade-5 { background: #e53935; }
      .grade-marker {
        position: absolute;
        top: -14px;
        transform: translateX(-50%);
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #1976d2;
        border: 2px solid #ffffff;
        box-shadow: 0 0 0 1px #1976d2;
      }
      .grade-marker-label {
        position: absolute;
        top: -32px;
        transform: translateX(-50%);
        color: #1976d2;
        font-size: 0.68rem;
        font-weight: 700;
        white-space: nowrap;
      }
      .grade-labels {
        display: grid;
        grid-template-columns: 10fr 24fr 32fr 24fr 10fr;
        margin-top: 7px;
      }
      .grade-labels span {
        color: #6b6662;
        font-size: 0.62rem;
        font-weight: 700;
        line-height: 1.25;
        text-align: center;
      }
      .chance-card { padding: 12px 10px; }
      .chance-label { color: #6b6662; font-size: 0.8rem; font-weight: 700; text-align: center; }
      .chance-value { color: #c98269; font-size: 1.1rem; font-weight: 700; text-align: center; margin: 4px 0 8px; }
      .meter-track { height: 9px; border-radius: 999px; background: #e7e1d8; overflow: hidden; }
      .meter-fill { height: 100%; border-radius: 999px; background: #c98269; }
      .info-card { padding: 2px 16px; }
      .info-row {
        display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px;
        align-items: center; padding: 12px 0;
        border-bottom: 1px solid rgba(201, 130, 105, 0.13);
      }
      .info-row:last-child { border-bottom: 0; }
      .info-label { color: #6b6662; font-size: 0.88rem; }
      .info-value { color: #4a403c; font-size: 0.9rem; font-weight: 700; text-align: right; }
      .explanation-panel {
        margin: 10px 0 16px;
        padding: 14px 16px;
        border: 1px solid rgba(201, 130, 105, 0.22);
        border-radius: 14px;
        background: #faf7f2;
        color: #4a403c;
        font-size: 0.9rem;
        line-height: 1.7;
      }
      .explanation-panel strong { color: #b8745d; }
      [data-testid="stExpander"] {
        border: 1px solid rgba(201, 130, 105, 0.18) !important;
        border-radius: 12px !important;
        background: #faf7f2 !important;
        margin: 6px 0 !important;
      }
      [data-testid="stExpander"] summary { color: #4a403c !important; font-weight: 700; }
      [data-testid="stVerticalBlockBorderWrapper"]:has(.analysis-panel-anchor) {
        height: calc(100dvh - 180px) !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0 !important;
        width: 100%;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        flex: 1 1 33.333% !important;
        justify-content: center !important;
        border-bottom-width: 2px !important;
        font-weight: 700;
      }
      [data-testid="stHorizontalBlock"]:has(.chat-toggle-anchor) {
        gap: 0 !important;
      }
      [data-testid="stHorizontalBlock"]:has(.chat-toggle-anchor) > [data-testid="stColumn"] {
        transition: flex-basis 360ms ease, width 360ms ease, opacity 260ms ease, padding 360ms ease;
      }
      [data-testid="stHorizontalBlock"]:has(.chat-toggle-anchor) > [data-testid="stColumn"]:has(.analysis-panel-anchor) {
        flex: 1 1 0% !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
      }
      [data-testid="stHorizontalBlock"]:has(.chat-toggle-anchor) > [data-testid="stColumn"]:has(.chat-toggle-anchor) {
        position: relative;
        flex: 0 0 34px !important;
        width: 34px !important;
        min-width: 34px !important;
        height: calc(100dvh - 180px);
        overflow: visible;
      }
      [data-testid="stHorizontalBlock"]:has(.chat-toggle-anchor) > [data-testid="stColumn"]:has(.chat-panel-anchor:not(.chat-collapsed)) {
        flex: 0 0 35% !important;
        width: 35% !important;
        opacity: 1;
      }
      [data-testid="stHorizontalBlock"]:has(.chat-toggle-anchor) > [data-testid="stColumn"]:has(.chat-panel-anchor.chat-collapsed) {
        flex: 0 0 0% !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        opacity: 0;
        overflow: hidden;
        pointer-events: none;
      }
      .chat-toggle-anchor { height: 0; }
      [data-testid="stColumn"]:has(.chat-toggle-anchor) {
        position: relative;
        overflow: visible;
      }
      [data-testid="stColumn"]:has(.chat-toggle-anchor) [data-testid="stButton"] {
        position: absolute;
        top: 50%;
        right: -17px;
        z-index: 20;
        transform: translateY(-50%);
        margin: 0 !important;
      }
      [data-testid="stColumn"]:has(.chat-toggle-anchor) [data-testid="stButton"] > button {
        width: 34px !important;
        min-width: 34px !important;
        height: 34px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
      }
      [data-testid="stColumn"]:has(.chat-toggle-anchor) [data-testid="stButton"] > button > div,
      [data-testid="stColumn"]:has(.chat-toggle-anchor) [data-testid="stButton"] > button p {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
      }
      .chat-panel-anchor { height: 0; }
      [data-testid="stColumn"]:has(.chat-panel-anchor) {
        padding: 8px 16px 12px;
        border-left: 1px solid #dedbd6;
        background: #f4f4f2;
      }
      .chat-header { display: flex; align-items: center; justify-content: space-between; }
      .chat-title { color: #4a403c; font-size: 1.08rem; font-weight: 700; margin: 2px 0 12px; }
      [data-testid="stPopover"] > button {
        min-width: auto; height: 38px; padding: 0 12px;
        border-radius: 999px; border: 1px solid rgba(201, 130, 105, 0.25);
        background: #faf7f2; color: #b8745d; font-weight: 700;
      }
      [data-testid="stColumn"]:has(.chat-panel-anchor) [data-testid="stButton"] > button {
        width: 38px !important; min-width: 38px !important; max-width: 38px !important;
        height: 38px !important; min-height: 38px !important;
        padding: 0 !important; border-radius: 50% !important;
        font-size: 1rem !important; line-height: 1 !important;
      }
      [data-testid="stTextInput"] label {
        color: #6b6662 !important; font-size: 0.82rem !important; font-weight: 700 !important;
      }
      [data-testid="stTextInput"] input {
        border: 1px solid rgba(201, 130, 105, 0.24) !important;
        border-radius: 10px !important; background: #faf7f2 !important; color: #4a403c !important;
      }
      .message-list {
        padding: 4px 2px 12px;
      }
      .message-row { display: flex; flex-direction: column; margin: 10px 0; }
      .message-row.user { align-items: flex-end; }
      .message-row.agent { align-items: flex-start; }
      .message-name { color: #6b6662; font-size: 0.74rem; margin: 0 4px 4px; }
      .message-bubble { max-width: 88%; padding: 10px 13px; border-radius: 15px; font-size: 0.9rem; line-height: 1.65; word-break: keep-word; }
      .message-row.user .message-bubble { background: #c98269; color: #ffffff; border-bottom-right-radius: 4px; }
      .message-row.agent .message-bubble { background: #faf7f2; color: #4a403c; border: 1px solid rgba(201, 130, 105, 0.15); border-bottom-left-radius: 4px; }
      .chat-disclaimer { color: #999999; font-size: 0.72rem; text-align: center; margin: 4px 0 8px; }
      [data-testid="stColumn"]:has(.chat-panel-anchor) [data-testid="stForm"] {
        border: 1px solid #dedbd6;
        border-radius: 14px;
        background: #ffffff;
        padding: 5px 7px;
      }
      [data-testid="stColumn"]:has(.chat-panel-anchor) [data-testid="stForm"] [data-testid="stTextInput"] input {
        border: 0 !important; background: transparent !important; box-shadow: none !important;
      }
      [data-testid="stColumn"]:has(.chat-panel-anchor) [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        min-height: 38px; border: 0; border-radius: 10px; background: #c98269; color: #ffffff;
      }
      @media (max-height: 760px) {
        [data-testid="stColumn"]:has(.chat-panel-anchor) {
          padding-bottom: max(48px, calc(env(safe-area-inset-bottom) + 36px));
        }
        [data-testid="stColumn"]:has(.chat-panel-anchor)
        [data-testid="stVerticalBlockBorderWrapper"]:has(.message-list) {
          height: clamp(220px, calc(100dvh - 300px), 420px) !important;
          min-height: 220px !important;
        }
      }
      @media (max-width: 850px) {
        .block-container {
          width: calc(100vw - 24px) !important;
          height: 100dvh;
          margin: 0 auto !important;
          padding: 18px 14px !important;
          padding-bottom: max(72px, calc(env(safe-area-inset-bottom) + 56px)) !important;
        }
        .app-title { font-size: 1.7rem; }
        .chance-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        [data-testid="stColumn"]:has(.chat-panel-anchor) {
          width: 100% !important;
          padding: 12px 12px max(48px, calc(env(safe-area-inset-bottom) + 36px));
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.analysis-panel-anchor) { height: calc(100dvh - 150px) !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_value(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def result_box(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="result-note">{escape(note)}</div>' if note else ""
    return (
        '<div class="result-box">'
        f'<div class="result-label">{escape(label)}</div>'
        f'<div class="result-value">{escape(value)}</div>{note_html}</div>'
    )


def relative_grade_html(top_percent: float | None) -> str:
    """상위 백분율을 5개 상대등급과 색상 막대로 표시합니다."""
    if top_percent is None:
        grade = "-"
        marker = ""
    else:
        safety_percent = 100.0 - top_percent
        grade = 1 if safety_percent <= 10 else 2 if safety_percent <= 34 else 3 if safety_percent <= 66 else 4 if safety_percent <= 90 else 5
        marker_position = max(1.0, min(99.0, safety_percent))
        marker = (
            f'<div class="grade-marker-label" style="left:{marker_position:.2f}%">내 상권</div>'
            f'<div class="grade-marker" style="left:{marker_position:.2f}%"></div>'
        )
    return (
        '<div class="relative-grade">'
        '<div class="relative-grade-title">안전 상대등급 · 5등급제</div>'
        '<div class="grade-scale-wrap">'
        f'{marker}'
        '<div class="grade-scale">'
        '<div class="grade-1"></div><div class="grade-2"></div><div class="grade-3"></div>'
        '<div class="grade-4"></div><div class="grade-5"></div>'
        '</div></div>'
        '<div class="grade-labels">'
        '<span>1등급<br>0~10%</span><span>2등급<br>10~34%</span><span>3등급<br>34~66%</span>'
        '<span>4등급<br>66~90%</span><span>5등급<br>90%~100%</span>'
        '</div></div>'
    )


def rank_display(value: float | None, count: int | None) -> str:
    return f"{count}개 중 {value:.1f}위" if value is not None and count else "계산 불가"


def recommend_markets(query: str, available_markets: list[str]) -> list[str]:
    """기존 AI 채팅 연결을 사용해 자연어 설명에 맞는 상권 후보를 찾습니다."""
    if not query.strip() or not available_markets:
        return []
    prompt = (
        "사용자가 찾는 상권 설명에 가장 가까운 후보를 아래 목록에서 최대 5개만 추천하십시오. "
        "목록에 없는 이름을 만들지 말고, 추천 상권명만 줄바꿈으로 출력하십시오.\n\n"
        f"사용자 설명: {query}\n\n후보 목록: {', '.join(available_markets)}"
    )
    answer = ask_ai(
        [{"role": "user", "content": prompt}],
        "상권 검색 사용자",
        "",
        {"available_markets": available_markets},
        {},
    )
    return [name for name in available_markets if name in answer][:5]


def message_bubble(role: str, content: str) -> str:
    is_user = role == "user"
    side = "user" if is_user else "agent"
    name = "당신" if is_user else "에이전트"
    safe_content = escape(content)
    # AI 답변의 Markdown 굵게 표시 문법(**텍스트**)을 채팅 HTML로 변환합니다.
    safe_content = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        safe_content,
        flags=re.DOTALL,
    ).replace("\n", "<br>")
    return (
        f'<div class="message-row {side}">'
        f'<div class="message-name">{name}</div>'
        f'<div class="message-bubble">{safe_content}</div></div>'
    )


def dashboard_brief_prompt(selected_market: str, metrics: dict[str, Any] | None) -> str:
    """채팅을 시작하기 전에 보여 줄 대시보드 요약 요청을 만듭니다."""
    if not metrics:
        return (
            f"{selected_market} 상권의 분석 자료가 부족합니다. 창업자에게 보여 줄 짧은 안내를 작성하십시오. "
            "반드시 5줄 이내로, 자료가 부족해 추가 확인이 필요하다는 점과 다음 행동 하나를 안내하십시오."
        )
    risk_band = (
        "이 상권은 진입하기 힘들어 보입니다"
        if metrics["top_percent"] <= 30
        else "이 상권은 좋아 보입니다"
        if metrics["top_percent"] >= 70
        else "이 상권은 추가 확인이 필요해 보입니다"
    )
    return (
        f"{selected_market} 상권의 대시보드 분석 결과를 창업자에게 먼저 안내하는 메시지를 작성하십시오. "
        "반드시 5줄 이내의 짧은 한국어 문장으로 작성하고, 목록이나 제목은 쓰지 마십시오. "
        f"첫 문장에 '{risk_band}'라는 판단을 자연스럽게 포함하십시오. "
        "상대위험도는 실제 발생 확률이 아니라 상권 간 비교라는 점을 짧게 밝히고, 다음 행동 한 가지를 제안하십시오."
    )


def render_summary_tab(market_metrics: dict[str, Any] | None, market_trend: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-heading">종합</div>', unsafe_allow_html=True)
    summary_html = '<div class="summary-grid">' + result_box(
        "선택 분기 상업 젠트리피케이션 상대위험도",
        f"{market_metrics['comparable_count']}개 상권 중 {market_metrics['overall_rank']:.1f}위" if market_metrics else "계산 불가",
        f"상위 {market_metrics['top_percent']:.1f}%" if market_metrics else "자료 부족",
    ) + relative_grade_html(market_metrics.get("top_percent") if market_metrics else None) + "</div>"
    st.markdown(summary_html, unsafe_allow_html=True)
    if market_metrics:
        st.caption(
            f"예측 대상: {market_metrics['target_period']} · "
            f"입력: {', '.join(market_metrics['input_periods'])} · "
            f"예측 방식: {market_metrics['prediction_method']}"
        )

    st.markdown('<div class="section-heading">위험도 추세</div>', unsafe_allow_html=True)
    if market_trend:
        difference = market_trend["difference"]
        st.markdown(
            '<div class="info-card"><div class="info-row">'
            f'<div class="info-label">{market_trend["previous_period"]} 대비</div>'
            f'<div class="info-value">{market_trend["label"]} · {difference:+.1f}점</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="info-card"><div class="info-row"><div class="info-label">추세</div><div class="info-value">추세 산정 불가</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">주요 위험 요인</div>', unsafe_allow_html=True)
    risk_factors = market_metrics.get("risk_factors", []) if market_metrics else []
    if risk_factors:
        factor_columns = st.columns(len(risk_factors), gap="small")
        for column, factor in zip(factor_columns, risk_factors):
            with column:
                st.markdown(
                    '<div class="chance-card">'
                    f'<div class="chance-label">{factor["label"]}</div>'
                    f'<div class="chance-value">{factor["value"]:.2f}{factor["unit"]}</div>'
                    f'<div class="result-note">{factor["text"]}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="info-card"><div class="info-row"><div class="info-label">근거 데이터</div><div class="info-value">부족</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">영역별 상대순위</div>', unsafe_allow_html=True)
    chance_values = [
        ("임대가격 상승 압력", market_metrics.get("cost_rank") if market_metrics else None),
        ("기존 점포 이탈 압력", market_metrics.get("exit_rank") if market_metrics else None),
        ("프랜차이즈 대체 압력", market_metrics.get("replacement_rank") if market_metrics else None),
    ]
    area_columns = st.columns(3, gap="small")
    for column, (key, (label, value)) in zip(area_columns, zip(("cost", "exit", "replacement"), chance_values)):
        with column:
            if st.button(
                f"{label}\n{rank_display(value, market_metrics.get('comparable_count') if market_metrics else None)}",
                key=f"area_card_{key}",
                use_container_width=True,
            ):
                st.session_state.explanation_key = key
    explanations = {
        "cost": "임대가격 상승 압력",
        "exit": "기존 점포 이탈 압력",
        "replacement": "프랜차이즈 대체 압력",
    }
    if st.session_state.explanation_key in explanations:
        st.markdown(
            f'<div class="explanation-panel"><strong>{explanations[st.session_state.explanation_key]}</strong><br>(fill)</div>',
            unsafe_allow_html=True,
        )
    st.caption("이 결과는 상권 간 상대순위 예측이며 실제 발생 확률이나 인과관계를 의미하지 않습니다.")
    st.caption("종합 비교값은 `100 × (비용 압력 ÷ 100) × (이탈 압력 ÷ 100) × (대체 압력 ÷ 100)`으로 계산합니다.")


def render_forecast_tab(market_metrics: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-heading">선택 분기 예상 변화</div>', unsafe_allow_html=True)
    labels = {
        "rent_growth": "임대가격지수 전년 동분기 상승률(%)", "general_store_decline": "일반 점포 전년 동분기 감소율(%)",
        "close_rate_change": "폐업률 전년 동분기 변화(%p)", "franchise_ratio_change": "프랜차이즈 비율 전년 동분기 변화(%p)",
        "store_turnover_change": "점포교체율 전년 동분기 변화(%p)", "open_rate_change": "개업률 전년 동분기 변화(%p)",
        "total_stores": "예상 총 점포 수", "rent_index": "예상 임대가격지수",
        "predicted_general_stores": "예상 일반 점포 수", "predicted_close_rate": "예상 폐업률(%)",
        "predicted_franchise_ratio": "예상 프랜차이즈 비율(%)", "predicted_store_turnover": "예상 점포교체율(%)",
    }
    metric_order = (
        "rent_growth", "general_store_decline", "close_rate_change", "franchise_ratio_change",
        "rent_index", "predicted_general_stores", "predicted_close_rate", "predicted_franchise_ratio", "predicted_store_turnover",
        "total_stores", "store_turnover_change", "open_rate_change",
    )
    displayed_metrics = [
        (name, (market_metrics or {}).get(name))
        for name in metric_order
        if (market_metrics or {}).get(name) is not None
    ]
    if not displayed_metrics:
        st.markdown('<div class="info-card"><div class="info-row"><div class="info-label">모형 계산 상태</div><div class="info-value">자료 부족</div></div></div>', unsafe_allow_html=True)
    elif market_metrics:
        st.caption(
            f"기준 분기 비교 가능 상권: {market_metrics['comparable_count']}개 "
            f"(임대가격지수 또는 점포 자료 부족으로 제외: {market_metrics['excluded_count']}개)"
        )
    for name, value in displayed_metrics:
        with st.expander(f"{labels[name]} · {display_value(value)}"):
            st.markdown("(fill)")
    if market_metrics and market_metrics.get("actual_values"):
        st.markdown('<div class="section-heading">실제 관측값과 비교</div>', unsafe_allow_html=True)
        comparison_items = (
            ("일반 점포 수", "predicted_general_stores", "general_stores"),
            ("폐업률", "predicted_close_rate", "close_rate"),
            ("프랜차이즈 비율", "predicted_franchise_ratio", "franchise_ratio"),
            ("점포교체율", "predicted_store_turnover", "store_turnover"),
            ("임대가격지수", "rent_index", "rent_index"),
        )
        rows = ["| 지표 | 예측값 | 실제값 | 오차 | 오차율 |", "| --- | ---: | ---: | ---: | ---: |"]
        for label, predicted_key, actual_key in comparison_items:
            predicted = market_metrics.get(predicted_key)
            actual = market_metrics["actual_values"].get(actual_key)
            if predicted is None or actual is None:
                continue
            error = predicted - actual
            error_rate = error / actual * 100 if actual else None
            rate_text = f"{error_rate:+.2f}%" if error_rate is not None else "계산 불가"
            rows.append(f"| {label} | {predicted:.2f} | {actual:.2f} | {error:+.2f} | {rate_text} |")
        if len(rows) > 2:
            st.markdown("\n".join(rows))


def _better_market(first_market: str, second_market: str, key: str, first_value: float, second_value: float) -> str:
    if first_value == second_value:
        return "동일"
    if key == "overall_rank":
        return first_market if first_value > second_value else second_market
    return first_market if first_value < second_value else second_market


def render_comparison_tab(names: list[str], selected_market: str | None) -> None:
    st.markdown('<div class="section-heading">상권 간 비교</div>', unsafe_allow_html=True)
    comparison_options = [market for market in names if market != selected_market]
    if st.session_state.get("comparison_market") not in comparison_options:
        st.session_state.comparison_market = None
    comparison_market = st.selectbox(
        "비교할 상권",
        comparison_options,
        index=None,
        placeholder="비교할 다른 상권을 선택하세요",
        key="comparison_market",
        disabled=not selected_market,
        label_visibility="collapsed",
    )
    if not selected_market or not comparison_market:
        return
    comparison = latest_common_comparison(names, selected_market, comparison_market)
    if not comparison:
        st.info("두 상권을 같은 공식으로 비교할 수 있는 공통 분기가 없습니다.")
        return
    comparison_period, first, second = comparison
    comparison_metrics = (
        ("위험 점수", "combined_comparison", "점"),
        ("상대위험도 순위", "overall_rank", "위"),
        ("임대가격지수 상승률", "rent_growth", "%"),
        ("폐업률 변화", "close_rate_change", "%p"),
        ("점포교체율 변화", "store_turnover_change", "%p"),
        ("프랜차이즈비율 변화", "franchise_ratio_change", "%p"),
    )
    table_lines = [
        f"| 항목 | {selected_market} | {comparison_market} | 차이({selected_market}-{comparison_market}) | 더 좋은 조건 |",
        "| --- | ---: | ---: | ---: | :--- |",
    ]
    differences: list[tuple[float, str]] = []
    for label, key, unit in comparison_metrics:
        first_value, second_value = first.get(key), second.get(key)
        if first_value is None or second_value is None:
            table_lines.append(f"| {label} | 자료 없음 | 자료 없음 | 자료 없음 | 판단 불가 |")
            continue
        better = _better_market(selected_market, comparison_market, key, first_value, second_value)
        table_lines.append(
            f"| {label} | {first_value:.2f}{unit} | {second_value:.2f}{unit} | {first_value - second_value:+.2f}{unit} | {better} |"
        )
        if key in {"rent_growth", "close_rate_change", "store_turnover_change", "franchise_ratio_change"}:
            differences.append((abs(first_value - second_value), label))
    st.caption(f"두 상권 모두 계산 가능한 최신 공통 분기: {comparison_period}")
    st.markdown("\n".join(table_lines))
    drivers = ", ".join(label for _, label in sorted(differences, reverse=True)[:2])
    if drivers:
        st.caption(f"위험도 차이에 크게 기여한 지표: {drivers}")


def close_market_section() -> None:
    """상권을 고른 직후 선택 영역을 접습니다."""
    st.session_state.market_section_open = False


if "chat_open" not in st.session_state:
    st.session_state.chat_open = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommended_markets" not in st.session_state:
    st.session_state.recommended_markets = []
if "explanation_key" not in st.session_state:
    st.session_state.explanation_key = ""
if "market_candidates" not in st.session_state:
    st.session_state.market_candidates = []
if "market_ai_searched" not in st.session_state:
    st.session_state.market_ai_searched = False
if "selected_market" not in st.session_state:
    st.session_state.selected_market = None
if "market_section_open" not in st.session_state:
    st.session_state.market_section_open = True
if "chat_greeting_sent" not in st.session_state:
    st.session_state.chat_greeting_sent = False
if "chat_suggestions_sent" not in st.session_state:
    st.session_state.chat_suggestions_sent = False
if "dashboard_brief_key" not in st.session_state:
    st.session_state.dashboard_brief_key = ""
if "dashboard_brief_pending" not in st.session_state:
    st.session_state.dashboard_brief_pending = False
if "prediction_config" not in st.session_state:
    st.session_state.prediction_config = None

st.markdown('<div class="app-title">상업 젠트리피케이션 상대위험도 예측</div>', unsafe_allow_html=True)
names = market_names()
with st.expander("분석할 상권 찾기", expanded=st.session_state.market_section_open):
    st.markdown('<div class="subsection-heading">AI로 찾기</div>', unsafe_allow_html=True)
    with st.form("market_search_form", border=False, clear_on_submit=False):
        search_column, search_button_column = st.columns([1, 0.18], gap="small")
        with search_column:
            market_query = st.text_input(
                "상권 설명 검색",
                placeholder="찾으시려는 지역의 특징을 입력하세요. 특징에 맞는 상권을 특정해 드립니다. (예시: 광역형 유동인구 많은 상권 ➡️ 강남대로) 서울 지역 상권을 지원합니다.",
                label_visibility="collapsed",
            )
        with search_button_column:
            find_market = st.form_submit_button("AI 찾기", use_container_width=True)
    if find_market and market_query.strip():
        st.session_state.market_ai_searched = True
        with st.spinner("관련 상권을 찾는 중입니다..."):
            try:
                candidates = recommend_markets(market_query, names)
            except Exception:
                candidates = []
        if not candidates:
            keyword = market_query.replace(" ", "")
            candidates = [name for name in names if keyword in name.replace(" ", "")]
        st.session_state.market_candidates = candidates
        if st.session_state.selected_market not in candidates:
            st.session_state.selected_market = None
        if not candidates:
            st.warning("관련 상권을 찾지 못했습니다. 다른 설명으로 다시 검색해 보십시오.")

    candidate_names = st.session_state.market_candidates or names
    if st.session_state.selected_market not in candidate_names:
        st.session_state.selected_market = None
    if st.session_state.market_candidates:
        st.caption(f"AI가 찾은 후보 {len(candidate_names)}개입니다. 아래에서 선택하십시오.")
    selected_market = st.selectbox(
        "분석 상권",
        candidate_names,
        index=None,
        placeholder="분석할 상권을 선택하세요",
        key="selected_market",
        on_change=close_market_section,
        label_visibility="collapsed",
    )
    if st.session_state.market_ai_searched:
        st.markdown('<div class="direct-market-label">또는 직접 검색하시겠어요?</div>', unsafe_allow_html=True)
        if st.button("필터 초기화", use_container_width=False):
            st.session_state.market_candidates = []
            st.session_state.market_ai_searched = False
            st.rerun()
available_periods = _available_periods(names) if names else []
selectable_targets = target_period_options(available_periods)
selected_input_periods: list[str] = []
target_period: str | None = None
with st.expander("예측 설정", expanded=True):
    if not selectable_targets:
        st.warning("예측 가능한 분기 자료가 부족합니다.")
    else:
        target_period = st.selectbox(
            "예측할 분기",
            options=selectable_targets,
            index=len(selectable_targets) - 1,
            format_func=lambda period: period.replace(".", "년 ").replace("/4", "분기"),
        )
        st.markdown("예측에 사용할 데이터")
        for option in get_previous_period_options(target_period, available_periods):
            if st.checkbox(
                option["label"],
                value=True,
                key=f"input_period_{target_period}_{option['lag']}",
            ):
                selected_input_periods.append(option["period"])
        selected_input_periods.sort(key=period_sort_key)
        if len(selected_input_periods) < 2:
            st.warning("추세 예측을 위해 이전 분기를 두 개 이상 선택하세요.")
        if st.button(
            "선택한 기간으로 예측하기",
            type="primary",
            disabled=len(selected_input_periods) < 2,
        ):
            st.session_state.prediction_config = {
                "target_period": target_period,
                "input_periods": selected_input_periods,
            }
calculation_error = ""
market_trend: dict[str, Any] | None = None
config = st.session_state.prediction_config
try:
    all_results = (
        commercial_risk_results(
            names,
            config["target_period"],
            config["input_periods"],
            available_periods,
        )
        if names and config
        else {}
    )
    if selected_market and selected_market in all_results and config:
        market_trend = risk_trend(
            names, selected_market, config["target_period"], config["input_periods"]
        )
except ValueError as error:
    all_results = {}
    calculation_error = str(error)
market_metrics = all_results.get(selected_market)
chat_ratio = 0.35
dashboard_column, toggle_column, chat_column = st.columns([0.62, 0.03, chat_ratio], gap="small")

with toggle_column:
    st.markdown('<div class="chat-toggle-anchor"></div>', unsafe_allow_html=True)
    if st.button("↔", key="toggle_chat", help="채팅창 접기 또는 펼치기"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

with dashboard_column:
    simulation_input = {"selected_market": selected_market, "commercial_risk_prediction": market_metrics or {}, "available_markets": names}
    analysis_scroll = st.container(height=720, border=False)
    with analysis_scroll:
        st.markdown('<div class="analysis-panel-anchor"></div>', unsafe_allow_html=True)
        if calculation_error:
            st.error(f"자료 확인 필요: {calculation_error}")
        summary_tab, forecast_tab, comparison_tab = st.tabs(["요약", "다음 분기 예상", "상권 비교"])
        with summary_tab:
            render_summary_tab(market_metrics, market_trend)
        with forecast_tab:
            render_forecast_tab(market_metrics)
        with comparison_tab:
            render_comparison_tab(names, selected_market)

if chat_column is not None:
    with chat_column:
        chat_panel_state = "" if st.session_state.chat_open else " chat-collapsed"
        st.markdown(f'<div class="chat-panel-anchor{chat_panel_state}"></div>', unsafe_allow_html=True)
        if not st.session_state.chat_greeting_sent:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "안녕하세요! 대화를 통해 가게를 준비하고 계신 창업자님의 상황에 맞추어 최적의 행동 방안을 제안해 드립니다.",
                }
            )
            st.session_state.chat_greeting_sent = True
        if not st.session_state.chat_suggestions_sent:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "다음과 같은 질문을 하실 수 있습니다.\n예시:\n"
                        '- "현재 이 상권의 성장세와 젠트리피케이션 진행 단계를 고려할 때, 지금 당장 진입하는 것이 좋은 타이밍일까요?"\n'
                        '- "이 상권의 최근 임대가격 상승 압력과 점포 구조 변화를 바탕으로, 입점 전 추가로 확인할 자료는 무엇인가요?"'
                    ),
                }
            )
            st.session_state.chat_suggestions_sent = True
        if selected_market:
            dashboard_brief_key = (
                f"{selected_market}|{(market_metrics or {}).get('overall_rank')}|"
                f"{(market_metrics or {}).get('top_percent')}"
            )
            if dashboard_brief_key != st.session_state.dashboard_brief_key:
                st.session_state.dashboard_brief_key = dashboard_brief_key
                st.session_state.dashboard_brief_pending = True
                st.rerun()
        else:
            st.session_state.dashboard_brief_key = ""
            st.session_state.dashboard_brief_pending = False

        st.markdown('<div class="chat-title">채팅</div>', unsafe_allow_html=True)

        messages_html = "".join(
            message_bubble(message["role"], message["content"])
            for message in st.session_state.messages
        )
        with st.container(height=420, border=False):
            messages_placeholder = st.empty()
            messages_placeholder.markdown(f'<div class="message-list">{messages_html}</div>', unsafe_allow_html=True)
        with st.form("chat_composer", clear_on_submit=True, border=False):
            input_column, send_column = st.columns([1, 0.18], gap="small")
            with input_column:
                prompt = st.text_input(
                    "질문",
                    placeholder="질문을 입력하세요",
                    label_visibility="collapsed",
                    disabled=st.session_state.dashboard_brief_pending,
                )
            with send_column:
                submitted = st.form_submit_button(
                    "전송",
                    use_container_width=True,
                    disabled=st.session_state.dashboard_brief_pending,
                )
        st.markdown('<div class="chat-disclaimer">실수할 수 있습니다. 중요한 정보는 검증하십시오.</div>', unsafe_allow_html=True)
        if st.session_state.dashboard_brief_pending:
            with st.spinner("대시보드를 분석해 안내를 작성하는 중입니다..."):
                try:
                    brief = ask_ai(
                        [{"role": "user", "content": dashboard_brief_prompt(selected_market, market_metrics)}],
                        "창업 준비자",
                        "",
                        simulation_input,
                        market_metrics or {},
                    )
                    brief = "\n".join(line for line in brief.splitlines() if line.strip())[:1200]
                    brief = "\n".join(brief.splitlines()[:5])
                except Exception:
                    brief = "대시보드 분석 안내를 불러오지 못했습니다. 현재 지표를 확인한 뒤 질문해 주십시오."
                st.session_state.messages.append({"role": "assistant", "content": brief})
                st.session_state.dashboard_brief_pending = False
            st.rerun()
        if submitted and prompt.strip():
            st.session_state.messages.append({"role": "user", "content": prompt})
            # 사용자 메시지를 먼저 그려서 전송 직후 채팅창에 즉시 표시합니다.
            messages_placeholder.markdown(
                '<div class="message-list">'
                + "".join(
                    message_bubble(message["role"], message["content"])
                    for message in st.session_state.messages
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            try:
                answer = ask_ai(
                    st.session_state.messages,
                    "창업 준비자",
                    "",
                    simulation_input,
                    market_metrics or {},
                )
            except Exception:
                answer = "AI 응답을 불러오지 못했습니다. API 키와 네트워크 연결을 확인하세요."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

"""상권별 상대위험도 계산에 사용하는 핵심 결합식."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def quarter_code_to_period(code: Any) -> str:
    """분기 코드(예: 20251)를 표준 표기(2025.1/4)로 바꾼다."""
    text = str(code).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 5 or not text.isdigit() or text[-1] not in "1234":
        raise ValueError(f"올바르지 않은 분기 코드입니다: {code}")
    return f"{text[:4]}.{text[4]}/4"


def period_to_quarter_code(period: str) -> int:
    year, quarter = _period_parts(period)
    return int(f"{year}{quarter}")


def _period_parts(period: str) -> tuple[int, int]:
    try:
        year_text, quarter_text = period.split(".", 1)
        year = int(year_text)
        quarter = int(quarter_text.split("/", 1)[0])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"올바르지 않은 분기 형식입니다: {period}") from error
    if quarter not in range(1, 5):
        raise ValueError(f"분기는 1~4여야 합니다: {period}")
    return year, quarter


def period_sort_key(period: str) -> int:
    year, quarter = _period_parts(period)
    return year * 4 + quarter


def shift_period(period: str, offset: int) -> str:
    year, quarter = _period_parts(period)
    shifted = year * 4 + (quarter - 1) + offset
    shifted_year, shifted_quarter = divmod(shifted, 4)
    return f"{shifted_year}.{shifted_quarter + 1}/4"


def target_period_options(available_periods: Iterable[str]) -> list[str]:
    """가장 이른 분기는 이전 자료가 없으므로 예측 대상에서 제외한다."""
    periods = sorted(set(available_periods), key=period_sort_key)
    return periods[1:]


def get_previous_period_options(
    target_period: str,
    available_periods: Iterable[str],
    max_lag: int = 4,
) -> list[dict[str, Any]]:
    available = set(available_periods)
    return [
        {"lag": lag, "period": period, "label": f"전 {lag}분기 ({period})"}
        for lag in range(1, max_lag + 1)
        if (period := shift_period(target_period, -lag)) in available
    ]


def validate_prediction_periods(
    target_period: str,
    input_periods: Iterable[str],
    available_periods: Iterable[str],
) -> list[str]:
    """입력 분기를 검증하고 중복을 제거한 시간순 목록을 반환한다."""
    available = set(available_periods)
    selected = sorted(set(input_periods), key=period_sort_key)
    if len(selected) < 2:
        raise ValueError("추세 예측을 위해 이전 분기를 두 개 이상 선택해야 합니다.")
    allowed = {shift_period(target_period, -lag) for lag in range(1, 5)}
    for period in selected:
        if period_sort_key(period) >= period_sort_key(target_period):
            raise ValueError("목표 분기와 같거나 이후인 분기는 입력으로 사용할 수 없습니다.")
        if period not in allowed:
            raise ValueError(f"{period}은 목표 분기 {target_period}의 전 1~4분기가 아닙니다.")
        if period not in available:
            raise ValueError(f"실제 데이터가 없는 분기입니다: {period}")
    return selected


def predict_numeric_series(
    period_value_pairs: Sequence[tuple[str, float | int | None]],
    target_period: str,
) -> float | None:
    """실제 분기 간격을 사용한 최소제곱 선형 추세로 목표 값을 예측한다."""
    valid: list[tuple[int, float]] = []
    for period, value in period_value_pairs:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        valid.append((period_sort_key(period), number))
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0][1]
    mean_x = sum(x for x, _ in valid) / len(valid)
    mean_y = sum(y for _, y in valid) / len(valid)
    denominator = sum((x - mean_x) ** 2 for x, _ in valid)
    if denominator == 0:
        return valid[-1][1]
    slope = sum((x - mean_x) * (y - mean_y) for x, y in valid) / denominator
    return mean_y + slope * (period_sort_key(target_period) - mean_x)


def commercial_risk_product(
    cost_pressure: float,
    exit_pressure: float,
    replacement_pressure: float,
) -> float:
    """세 상업 압력의 동시 충족 정도를 곱셈식으로 결합한다.

    각 입력은 상권 간 순위환산값(0~100)이며, 반환값은 발생 확률이 아니다.
    """
    pressures = (cost_pressure, exit_pressure, replacement_pressure)
    if any(not 0 <= pressure <= 100 for pressure in pressures):
        raise ValueError("비용·이탈·대체 압력은 모두 0~100 범위여야 합니다.")
    return 100 * (cost_pressure / 100) * (exit_pressure / 100) * (replacement_pressure / 100)

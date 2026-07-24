"""사용자가 원자료 형식 그대로 붙여 넣는 상권 파일을 읽는 모듈."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from model import period_sort_key, quarter_code_to_period


MARKET_INPUT_DIR = Path(__file__).with_name("market_input")


def _rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        dialect = csv.excel
        dialect.delimiter = ","
    return list(csv.reader(text.splitlines(), dialect))


def market_names() -> list[str]:
    names: set[str] = set()
    for filename in ("market_rent.csv", "market_vacancy.csv"):
        rows = _rows(MARKET_INPUT_DIR / filename)
        for row in rows[1:]:
            if len(row) >= 3 and row[2].strip() and row[2].strip() != "소계":
                names.add(row[2].strip())
    rows = _rows(MARKET_INPUT_DIR / "market_processed.csv")
    for row in rows[1:]:
        if len(row) >= 2 and row[1].strip():
            names.add(row[1].strip())
    return sorted(names)


def _wide_snapshot(filename: str, market: str) -> dict[str, Any]:
    rows = _rows(MARKET_INPUT_DIR / filename)
    if not rows:
        return {}
    header = rows[0]
    periods = sorted(
        [column for column in header[3:] if "." in column and column.endswith("/4")],
        key=period_sort_key,
    )
    if not periods:
        return {}
    period_indexes = {period: header.index(period) for period in periods}
    for row in rows[1:]:
        if len(row) >= 3 and row[2].strip() == market:
            values = {period: row[index].strip() if index < len(row) else "" for period, index in period_indexes.items()}
            latest_period = periods[-1]
            return {"market": market, "values": values, "latest_period": latest_period, "latest_value": values[latest_period]}
    return {}


def market_snapshot(market: str) -> dict[str, Any]:
    result = {"market": market}
    result["rent"] = _wide_snapshot("market_rent.csv", market)
    result["vacancy"] = _wide_snapshot("market_vacancy.csv", market)
    rows = _rows(MARKET_INPUT_DIR / "market_processed.csv")
    if rows:
        header = rows[0]
        name_index = header.index("상권_코드_명") if "상권_코드_명" in header else 1
        matching = [row for row in rows[1:] if len(row) > name_index and row[0].strip().isdigit() and row[name_index].strip() == market]
        if matching:
            row = max(matching, key=lambda item: period_sort_key(quarter_code_to_period(item[0])))
            result["processed"] = {key: row[index] if index < len(row) else "" for index, key in enumerate(header)}
    return result

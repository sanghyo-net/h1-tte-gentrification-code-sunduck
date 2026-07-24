"""OpenAI Responses API와 대시보드 채팅을 연결한다."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


MODEL = "gpt-5.6-luna"
PROJECT_DIR = Path(__file__).resolve().parent
API_KEY_PATH = PROJECT_DIR / "api_key.md"
INSTRUCTIONS_PATH = PROJECT_DIR / "llm_instructions.md"


def load_api_key(path: Path = API_KEY_PATH) -> str:
    """환경 변수, Streamlit Secrets, 로컬 파일 순으로 API 키를 찾는다."""
    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        import streamlit as st

        secret_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""
    if secret_key:
        return secret_key

    if path.exists():
        text = path.read_text(encoding="utf-8")
        match = re.search(r"sk-[A-Za-z0-9_-]{20,}", text)
        if match:
            return match.group(0)

    raise RuntimeError(
        "OPENAI_API_KEY를 Streamlit Secrets 또는 환경 변수에 등록해 주세요."
    )


def load_instructions(path: Path = INSTRUCTIONS_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError("llm_instructions.md를 찾을 수 없습니다.")
    instructions = path.read_text(encoding="utf-8").strip()
    if not instructions:
        raise ValueError("llm_instructions.md가 비어 있습니다.")
    return instructions


def simulation_context(
    user_role: str,
    current_region: str,
    simulation_input: dict[str, Any],
    result: dict[str, Any],
) -> str:
    payload = {
        "User_Role": user_role,
        "Current_Region": current_region,
        "simulation_input": simulation_input,
        "simulation_result": result,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def ask_ai(
    messages: Iterable[dict[str, str]],
    user_role: str,
    current_region: str,
    simulation_input: dict[str, Any],
    result: dict[str, Any],
    uploaded_data: str = "",
) -> str:
    instructions = load_instructions()
    context = simulation_context(user_role, current_region, simulation_input, result)
    client = OpenAI(api_key=load_api_key())
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "high"},
        instructions=(
            f"{instructions}\n\n"
            "아래 시뮬레이션 결과와 업로드 원자료만 현재 상태를 판단하는 근거로 사용하세요. "
            "제공되지 않은 수치나 사실을 만들어내지 마세요.\n\n"
            f"[현재 시뮬레이션 결과]\n{context}\n\n"
            f"[업로드 원자료]\n{uploaded_data}"
        ),
        input=list(messages),
    )
    return response.output_text


def recommend_markets(query: str, available_markets: list[str]) -> list[str]:
    """자연어 지역 설명을 등록된 상권명 후보로 좁힙니다."""
    if not query.strip() or not available_markets:
        return []
    client = OpenAI(api_key=load_api_key())
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "high"},
        instructions=(
            "당신은 한국 상권명 검색 보조 도구입니다. 사용자의 자연어 설명과 후보 상권명만 보고 "
            "가장 관련 있는 후보를 최대 5개 고르십시오. 후보에 없는 상권명을 만들지 마십시오. "
            "반드시 JSON 배열만 출력하십시오. 예: [\"홍대/합정\", \"동교/연남\"]"
        ),
        input=(
            f"[사용자 설명]\n{query}\n\n"
            f"[선택 가능한 상권명]\n{json.dumps(available_markets, ensure_ascii=False)}"
        ),
    )
    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [name for name in parsed if isinstance(name, str) and name in available_markets][:5] if isinstance(parsed, list) else []


def ask_uploaded_data_analysis(user_role: str, uploaded_data: str) -> str:
    """형식이 제각각인 CSV 원자료를 AI가 읽어, 가능한 범위만 분석한다."""
    return ask_ai(
        messages=[{"role": "user", "content": "업로드한 통계자료를 분석해 현재 확인 가능한 변화와 계산 불가 항목을 구분해 설명해 주세요."}],
        user_role=user_role,
        current_region="",
        simulation_input={},
        result={},
        uploaded_data=uploaded_data,
    )


def normalize_uploaded_tables(uploaded_data: str) -> dict[str, list[dict[str, Any]]]:
    """서로 다른 한국어·영어 열 이름의 CSV 표를 계산용 공통 열로 정규화한다."""
    client = OpenAI(api_key=load_api_key())
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "high"},
        instructions=(
            "당신은 통계 표 정규화 도구입니다. 업로드 원자료에서 실제로 존재하는 값만 사용하세요. "
            "열 이름은 한국어·영어·약어가 달라도 의미를 해석하세요. 예: 생활물가지수/소비자물가지수/CPI는 cpi, "
            "전입은 inflow, 전출은 outflow, 제조업 사업체 수는 manufacturing_count입니다. "
            "서로 다른 파일에 있는 열은 year를 기준으로 결합할 수 있습니다. 값이나 연도를 추정하거나 만들어내지 마세요. "
            "반드시 Markdown 없이 유효한 JSON 객체만 반환하세요. 키는 rent_cpi, manufacturing, migration, population이며, "
            "각 값은 행 객체 배열입니다. rent_cpi 행은 year와 rent 또는 cpi 중 실제 있는 값만, manufacturing 행은 year와 manufacturing_count, "
            "migration 행은 year와 inflow 또는 outflow, population 행은 year와 population을 포함합니다. "
            "파일이 해당 자료가 아니면 무시하세요."
        ),
        input=f"[업로드 원자료]\n{uploaded_data}",
    )
    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("AI가 업로드 표를 계산용 형식으로 변환하지 못했습니다.") from error
    if not isinstance(parsed, dict):
        raise ValueError("AI 표 변환 결과 형식이 올바르지 않습니다.")
    return {key: value for key, value in parsed.items() if key in {"rent_cpi", "manufacturing", "migration", "population"} and isinstance(value, list)}

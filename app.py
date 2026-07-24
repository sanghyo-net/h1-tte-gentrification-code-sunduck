"""Streamlit Community Cloud용 대시보드 진입점."""

import streamlit as st

try:
    import dashboard  # noqa: F401
except Exception as error:
    st.error("앱을 시작하지 못했습니다. 아래 오류 내용을 확인하세요.")
    st.exception(error)

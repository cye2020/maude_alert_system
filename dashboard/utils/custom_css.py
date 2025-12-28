# custom_css.py
"""대시보드 커스텀 CSS 스타일"""

import streamlit as st


def apply_custom_css():
    """커스텀 CSS 적용"""
    st.markdown("""
    <style>
    /* 메트릭 카드 테두리 및 스타일 */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    [data-testid="stMetric"]:hover {
        border-color: #adb5bd;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }

    /* 메트릭 라벨 스타일 */
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #495057;
    }

    /* 메트릭 값 스타일 */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #212529;
    }

    /* 메트릭 델타 스타일 */
    [data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

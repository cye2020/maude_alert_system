# overview_tab.py
import streamlit as st
from utils.filter_manager import create_sidebar

# overview_tab.py
def show(filters=None):
    st.title("📊 Overview")

    # 필터 값 사용
    selected_date = filters.get("date")
    window = filters.get("window", 1)
    
    # KPI 메트릭 (3열 레이아웃)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📁 총 이상 사례 보고 건수",
            value="1,234,567",
            delta="12.5%",
            delta_arrow='down',
            delta_color='inverse'
        )

    with col2:
        st.metric(
            label="⚙️ 파이프라인 상태",
            value="정상",
            delta="100% Uptime"
        )

    with col3:
        st.metric(
            label="🤖 모델 정확도",
            value="94.2%",
            delta="↑ 2.3%"
        )

    st.markdown("---")
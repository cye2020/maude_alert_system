"""
Streamlit 멀티페이지 대시보드 - 메인 홈페이지
"""
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from millify import millify
import overview_tab as o_tab
import eda_tab as e_tab
import cluster_tab as c_tab
from utils.filter_manager import create_sidebar


# 프로젝트 루트 경로 설정
root_path = Path(__file__).parent
sys.path.append(str(root_path))

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="MAUDE 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 세션 상태 초기화
if 'TODAY' not in st.session_state:
    st.session_state.TODAY = datetime.now()

TODAY = st.session_state.TODAY

# ==================== 탭 선택 (세그먼트 컨트롤) ====================

# 탭 옵션 정의
tab_options = {
    "📊 Overview": "overview",
    "📈 Detailed Analysis": "eda",
    "🔍 Clustering Reports": "cluster"
}

# 세그먼트 컨트롤로 탭 선택
selected_tab_display = st.segmented_control(
    label="대시보드 선택",
    options=list(tab_options.keys()),
    default="📊 Overview",
    label_visibility="collapsed",
    selection_mode="single",
    key="selected_tab_key"
)

# None인 경우 기본값 사용 (선택 해제 시 이전 값 유지를 위해 rerun)
if selected_tab_display is None:
    selected_tab_display = "📊 Overview"
    st.rerun()

current_tab = tab_options[selected_tab_display]

# ==================== 사이드바 ====================
# 선택된 탭에 맞는 사이드바 렌더링
filters = create_sidebar(current_tab)

# ==================== 메인 콘텐츠 ====================

# 선택된 탭의 콘텐츠 표시
if current_tab == "overview":
    o_tab.show(filters)
elif current_tab == "eda":
    e_tab.show(filters)
elif current_tab == "cluster":
    c_tab.show(filters)


# ==================== 시스템 상태 ====================
st.subheader("🖥️ 시스템 상태")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**데이터 파이프라인**")
    st.progress(0.95)
    st.caption("95% - 정상 작동 중")

with col2:
    st.markdown("**모델 서빙**")
    st.progress(1.0)
    st.caption("100% - 정상")

with col3:
    st.markdown("**데이터베이스**")
    st.progress(0.87)
    st.caption("87% - 여유 공간")

# ==================== 푸터 ====================
st.markdown("---")
st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 버전: 1.0.0")
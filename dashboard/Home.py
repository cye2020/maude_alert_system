"""
Streamlit 멀티페이지 대시보드 - 메인 홈페이지
"""

import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from millify import millify
import overview_tab as o_tab
import eda_tab as e_tab
import cluster_tab as c_tab


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="MAUDE 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 초기화 시 한 번만 TODAY 설정
if 'TODAY' not in st.session_state:
    st.session_state.TODAY = datetime.now()

TODAY = st.session_state.TODAY

# ==================== 사이드바 ====================
with st.sidebar:
    st.image("dashboard/assets/logo.png", width='stretch')
    
    # 프로젝트 정보
    st.markdown("### 📌 프로젝트 정보")
    st.info("""
    **버전**: v1.0.0  
    **업데이트**: 2025-12-24  
    **환경**: Development
    """)
    
    st.markdown('---')

    with st.container(horizontal=True):
        year_range = 3
        year = st.selectbox(
            "년도",
            range(TODAY.year - year_range + 1, TODAY.year+1),
            index=year_range - 1,
            format_func=lambda x: f"{x}년",
            width="stretch",
            key="sidebar_year"
        )
        st.space(1)  # 간격 추가
        month = st.selectbox(
            "월",
            range(1, 13),
            format_func=lambda x: f"{x:02d}월",
            width="stretch",
            key="sidebar_month"
        )

    selected_date = datetime(year, month, 1)
    st.write(f"선택된 년월: {selected_date.strftime('%Y년 %m월')}")
    
    # 선택한 년월을 session_state에 저장 (YYYY-MM 형식)
    st.session_state.selected_year_month = selected_date.strftime('%Y-%m')
    
    window = st.selectbox(
        label='관측 기간',
        options = [1, 3],
        index = 0,
        format_func=lambda op: f'{op}개월',
        key="sidebar_window"
    )
    
    # 선택한 window를 session_state에 저장
    st.session_state.selected_window = window
    
    st.markdown("---")
    
    # 빠른 링크
    st.markdown("### 🔗 빠른 링크")
    st.markdown("""
    - [데이터 개요](#data-overview)
    - [분석 대시보드](#analytics)
    - [모델 성능](#model-performance)
    """)

# ==================== 메인 콘텐츠 ====================

# 헤더
# st.title("🏠 홈 대시보드")
# st.markdown("데이터 파이프라인과 ML 모델 모니터링을 위한 통합 대시보드입니다.")

# 메인 영역 상단의 탭
overview_tab, eda_tab, cluster_tab = st.tabs([
    "Overview", 
    "Detailed Analysis", 
    "Clustering Reports"
])

# 탭 내용
with overview_tab:
    o_tab.show()

with eda_tab:
    e_tab.show()
    

with cluster_tab:
    c_tab.show()

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
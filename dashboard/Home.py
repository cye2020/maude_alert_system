"""
Streamlit 멀티페이지 대시보드 - 메인 홈페이지
"""
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from millify import millify
import polars as pl
import overview_tab as o_tab
import dashboard.eda_tab as e_tab
import cluster_tab as c_tab
import spike_tab as s_tab
from dashboard.utils.sidebar_manager import create_sidebar
from utils.dashboard_config import get_config


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


# ==================== 데이터 로딩 ====================

@st.cache_data
def load_maude_data(cache_key: str) -> pl.DataFrame:
    """Silver Stage3 (클러스터링) 데이터 로드

    매월 1일에 자동 갱신 (cache_key가 변경되면 캐시 무효화)

    Args:
        cache_key: 캐시 키 (예: "2025-01") - 월이 바뀌면 자동 갱신
    """
    config = get_config()
    data_path = config.get_silver_stage3_path(dataset='maude')

    if not data_path.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        st.stop()

    return pl.scan_parquet(data_path)

# 세션 상태 초기화
if 'TODAY' not in st.session_state:
    st.session_state.TODAY = datetime.now()

# 매월 1일 기준으로 캐시 키 생성 (예: "2025-01")
# 월이 바뀌면 cache_key가 달라져서 자동으로 새 데이터 로드
cache_key = st.session_state.TODAY.strftime("%Y-%m")

if 'data' not in st.session_state:
    with st.spinner("데이터 로딩 중..."):
        st.session_state.data = load_maude_data(cache_key)

TODAY = st.session_state.TODAY
maude_lf = st.session_state.data

# ==================== 탭 선택 (세그먼트 컨트롤) ====================

# 탭 옵션 정의
tab_options = {
    "📊 Overview": "overview",
    "📈 Detailed Analytics": 'eda',
    "🚨 Spike Detection": "spike",
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
# cluster 탭의 경우 동적 옵션 전달
if current_tab == "cluster":
    # cluster 탭: available_clusters를 미리 계산 (전체 데이터 기준)
    from utils.analysis_cluster import get_available_clusters
    from utils.constants import ColumnNames
    from utils.data_utils import get_year_month_expr

    # year_month 표현식 생성
    year_month_expr = get_year_month_expr(maude_lf, ColumnNames.DATE_RECEIVED)

    # available_clusters 계산 (전체 데이터 기준)
    available_clusters = get_available_clusters(
        _lf=maude_lf,
        cluster_col=ColumnNames.CLUSTER,
        date_col=ColumnNames.DATE_RECEIVED,
        selected_dates=None,  # 전체 기간
        selected_manufacturers=None,
        selected_products=None,
        exclude_minus_one=True,
        _year_month_expr=year_month_expr
    )

    # 동적 옵션으로 사이드바 렌더링
    from dashboard.utils.sidebar_manager import SidebarManager
    manager = SidebarManager(current_tab)
    dynamic_options = {
        "selected_cluster": available_clusters
    }
    filters = manager.render_sidebar(dynamic_options=dynamic_options)
else:
    filters = create_sidebar(current_tab)

# ==================== 메인 콘텐츠 ====================

# 선택된 탭의 콘텐츠 표시
if current_tab == "overview":
    o_tab.show(filters, maude_lf)
elif current_tab == 'eda':
    e_tab.show(filters, maude_lf)
elif current_tab == 'spike':
    s_tab.show(filters, maude_lf)
elif current_tab == "cluster":
    c_tab.show(filters, maude_lf)

# ==================== 푸터 ====================
st.markdown("---")
st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 버전: 1.0.0")
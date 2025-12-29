"""
Streamlit 멀티페이지 대시보드 - 메인 홈페이지
"""
# 1. 표준 라이브러리
import sys
from pathlib import Path
from datetime import datetime

# 2. 서드파티 라이브러리
import streamlit as st
import polars as pl

# 3. 프로젝트 내부 탭 모듈
import overview_tab as o_tab
import eda_tab as e_tab
import cluster_tab as c_tab
import spike_tab as s_tab

# 4. 프로젝트 유틸 / 설정
from utils.dashboard_config import get_config
from utils.constants import DisplayNames
from dashboard.utils.custom_css import apply_custom_css

# 커스텀 CSS 적용
apply_custom_css()


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
    
    print('='*50)
    print(data_path)
    print('='*50)

    # S3 사용 여부에 따라 storage_options 설정
    storage_options = config.get_s3_storage_options()

    if storage_options:
        # S3 경로: 존재 체크 없이 바로 로드
        return pl.scan_parquet(str(data_path), storage_options=storage_options)
    else:
        # 로컬 경로: 존재 체크 후 로드
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

# 탭 옵션 정의 (한글 표준화)
tab_options = {
    DisplayNames.FULL_TITLE_OVERVIEW: "overview",
    DisplayNames.FULL_TITLE_EDA: "eda",
    DisplayNames.FULL_TITLE_SPIKE: "spike",
    DisplayNames.FULL_TITLE_CLUSTER: "cluster"
}

# 세그먼트 컨트롤로 탭 선택
selected_tab_display = st.segmented_control(
    label="대시보드 선택",
    options=list(tab_options.keys()),
    default=DisplayNames.FULL_TITLE_OVERVIEW,
    label_visibility="collapsed",
    selection_mode="single",
    key="selected_tab_key"
)

# None인 경우 기본값 사용 (선택 해제 시 이전 값 유지를 위해 rerun)
if selected_tab_display is None:
    selected_tab_display = DisplayNames.FULL_TITLE_OVERVIEW
    st.rerun()

current_tab = tab_options[selected_tab_display]

# ==================== 사이드바 ====================
# 선택된 탭에 맞는 사이드바 렌더링
from dashboard.utils.sidebar_manager import SidebarManager
from utils.constants import ColumnNames
from utils.data_utils import get_year_month_expr

# year_month 표현식 생성 (공통)
year_month_expr = get_year_month_expr(maude_lf, ColumnNames.DATE_RECEIVED)

# 공통 필터 옵션 로드 (모든 탭에서 사용)
from dashboard.utils.filter_helpers import (
    get_available_filters,
    get_available_clusters
)

# 1. 제조사, 제품군 (전체 데이터 기준)
_, available_manufacturers, available_products = get_available_filters(
    maude_lf,
    date_col=ColumnNames.DATE_RECEIVED,
    _year_month_expr=year_month_expr
)

# 2. 클러스터 (전체 데이터 기준, -1 제외)
available_clusters = get_available_clusters(
    maude_lf,
    cluster_col=ColumnNames.CLUSTER,
    exclude_minus_one=True
)

# 3. 결함 유형 (전체 데이터 기준)
from dashboard.utils.filter_helpers import get_available_defect_types
available_defect_types = get_available_defect_types(
    maude_lf,
    date_col=ColumnNames.DATE_RECEIVED,
    _year_month_expr=year_month_expr
)

# 4. 기기는 cascade이므로 빈 리스트로 초기화 (사용자 선택에 따라 업데이트)
available_devices = []

# 공통 동적 옵션 구성
common_dynamic_options = {
    "manufacturers": available_manufacturers,
    "products": available_products,
    "devices": available_devices,
    "defect_types": available_defect_types,
    "clusters": available_clusters,
    "_cascade_config": {
        "products": {
            "depends_on": "manufacturers",
            "data_source": maude_lf
        },
        "devices": {
            "depends_on": ["manufacturers", "products"],
            "data_source": maude_lf
        }
    }
}

# 탭별 추가 동적 옵션 (필요 시)
if current_tab == "cluster":
    # Cluster 탭: selected_cluster 옵션 추가
    from utils.analysis_cluster import get_available_clusters as get_clusters_with_dates

    tab_available_clusters = get_clusters_with_dates(
        _lf=maude_lf,
        cluster_col=ColumnNames.CLUSTER,
        date_col=ColumnNames.DATE_RECEIVED,
        selected_dates=None,
        selected_manufacturers=None,
        selected_products=None,
        exclude_minus_one=True,
        _year_month_expr=year_month_expr
    )

    common_dynamic_options["selected_cluster"] = tab_available_clusters

# 사이드바 렌더링
manager = SidebarManager(current_tab)
filters = manager.render_sidebar(dynamic_options=common_dynamic_options)

# ==================== 필터 변경 감지 및 캐시 무효화 ====================
# TODO: 캐싱 전략 개선 필요
# 현재는 필터 변경 시 모든 캐시를 클리어하지만 (st.cache_data.clear()),
# 이는 너무 aggressive한 방식입니다.
# 개선 방안:
# 1. 필터별로 독립적인 캐시 키 사용 (cache_key에 필터 값 포함)
# 2. 특정 함수의 캐시만 선택적으로 무효화
# 3. 필터 변경 시 영향받는 데이터만 재계산
#
# 필터 값을 문자열로 변환하여 이전 값과 비교
import json
current_filter_state = json.dumps({
    "manufacturers": filters.get("manufacturers", []),
    "products": filters.get("products", []),
}, sort_keys=True)

# 세션 스테이트에 이전 필터 상태 저장
if "prev_filter_state" not in st.session_state:
    st.session_state.prev_filter_state = current_filter_state
elif st.session_state.prev_filter_state != current_filter_state:
    # 필터가 변경되었으면 캐시 클리어 및 페이지 재실행
    st.cache_data.clear()  # TODO: 전체 클리어 대신 필터 관련 캐시만 선택적으로 무효화
    st.session_state.prev_filter_state = current_filter_state
    st.rerun()

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
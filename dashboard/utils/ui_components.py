# ui_components.py
"""
대시보드 공통 UI 컴포넌트
모든 탭에서 재사용 가능한 UI 함수들을 제공
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple

from dashboard.utils.constants import DisplayNames, HarmColors, ChartStyles
from dashboard.utils.dashboard_config import get_config


# ==================== 필터 관련 ====================

def render_filter_summary_badge(
    date_range: Optional[Tuple[datetime, datetime]] = None,
    manufacturers: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    **kwargs
) -> None:
    """필터 요약 배지 표시 (모든 탭 공통)

    Args:
        date_range: (start, end) datetime tuple
        manufacturers: 선택된 제조사 리스트
        products: 선택된 제품 리스트
        **kwargs: 추가 필터
            - segment: 분석 기준 (컬럼명)
            - cluster: 클러스터 번호
            - defect_type: 결함 유형
            - top_n: 상위 N개
            - min_cases: 최소 케이스 수

    Example:
        >>> render_filter_summary_badge(
        ...     date_range=(start_dt, end_dt),
        ...     manufacturers=["Manufacturer A"],
        ...     segment="product_code"
        ... )
    """
    cfg = get_config()
    filter_config = cfg.defaults.get('filter_summary', {})

    if not filter_config.get('enabled', True):
        return

    badges = []
    separator = filter_config.get('format', {}).get('item_separator', ' · ')

    # 날짜 범위
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        if isinstance(start, datetime) and isinstance(end, datetime):
            badges.append(f"📅 {start.strftime('%Y-%m')} ~ {end.strftime('%Y-%m')}")

    # Segment (분석 기준) - 한글 매핑
    segment = kwargs.get('segment')
    if segment:
        segment_map = {
            'manufacturer_name': '제조사',
            'product_code': '제품군',
            'udi_di': '기기',
            'cluster': '클러스터',
            'defect_type': '결함 유형'
        }
        segment_label = segment_map.get(segment, segment)
        badges.append(f"🎯 분석 기준: {segment_label}")

    # 제조사
    if manufacturers and len(manufacturers) > 0:
        badges.append(f"🏭 {len(manufacturers)}개 제조사")

    # 제품
    if products and len(products) > 0:
        badges.append(f"📦 {len(products)}개 제품")

    # 클러스터
    cluster = kwargs.get('cluster')
    if cluster is not None:
        badges.append(f"🔍 클러스터 {cluster}")

    # 결함 유형
    defect_type = kwargs.get('defect_type')
    if defect_type:
        badges.append(f"🔧 {defect_type}")

    # Top N
    top_n = kwargs.get('top_n')
    if top_n:
        badges.append(f"📊 상위 {top_n}개")

    # 최소 케이스 수
    min_cases = kwargs.get('min_cases')
    if min_cases:
        badges.append(f"📉 최소 {min_cases}건")

    # 배지가 없으면 기본 텍스트
    if not badges:
        default_text = filter_config.get('format', {}).get('default_text', '🌐 전체 데이터')
        badges.append(default_text)

    # 표시
    st.markdown(f"**적용된 필터:** {separator.join(badges)}")


def render_spike_filter_summary(
    as_of_month: str = None,
    window: int = None,
    min_c_recent: int = None,
    z_threshold: float = None,
    **kwargs
) -> None:
    """Spike Detection 탭 전용 필터 요약

    Args:
        as_of_month: 기준 월 (예: "2025-11")
        window: 윈도우 크기 (1 또는 3)
        min_c_recent: 최소 최근 케이스 수
        z_threshold: Z-score 임계값
        **kwargs: 추가 파라미터
            - alpha: 유의수준
            - correction: 다중검정 보정
            - min_methods: 앙상블 최소 방법 수
    """
    badges = []

    # 기준 월
    if as_of_month:
        badges.append(f"📅 기준 월: {as_of_month}")

    # 윈도우 크기
    if window:
        window_label = f"{window}개월" if window > 1 else f"{window}개월"
        badges.append(f"📊 윈도우: {window_label}")

    # 최소 케이스 수
    if min_c_recent:
        badges.append(f"📈 최소 케이스: {min_c_recent}건")

    # Z-score 임계값
    if z_threshold:
        badges.append(f"📉 Z-score ≥ {z_threshold:.2f}σ")

    # 유의수준
    alpha = kwargs.get('alpha')
    if alpha:
        badges.append(f"⚡ α = {alpha}")

    # 다중검정 보정
    correction = kwargs.get('correction')
    if correction:
        correction_map = {
            'fdr_bh': 'FDR (Benjamini-Hochberg)',
            'bonferroni': 'Bonferroni',
            'sidak': 'Sidak'
        }
        correction_label = correction_map.get(correction, correction)
        badges.append(f"🔧 보정: {correction_label}")

    # 앙상블 최소 방법 수
    min_methods = kwargs.get('min_methods')
    if min_methods:
        badges.append(f"🎯 앙상블: {min_methods}개 이상")

    # 배지가 없으면 기본값
    if not badges:
        badges.append("🌐 기본 설정")

    # 표시
    st.markdown(f"**분석 설정:** {' · '.join(badges)}")


def convert_date_range_to_months(date_range: Optional[Tuple]) -> List[str]:
    """날짜 범위를 년-월 문자열 리스트로 변환 (모든 탭 공통)

    Args:
        date_range: (start_date, end_date) tuple
                   각 요소는 datetime 또는 str

    Returns:
        List[str]: 년-월 리스트 (예: ['2024-11', '2024-12', '2025-01'])

    Example:
        >>> start = datetime(2024, 11, 1)
        >>> end = datetime(2025, 1, 1)
        >>> convert_date_range_to_months((start, end))
        ['2024-11', '2024-12', '2025-01']
    """
    if not date_range or len(date_range) != 2:
        return []

    start_val, end_val = date_range

    # datetime 객체로 변환
    if isinstance(start_val, str):
        start = datetime.strptime(start_val, "%Y-%m")
    else:
        start = start_val

    if isinstance(end_val, str):
        end = datetime.strptime(end_val, "%Y-%m")
    else:
        end = end_val

    # 월 리스트 생성
    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    return months


# ==================== 차트 생성 ====================

def create_pie_chart(
    data: Dict[str, int] = None,
    labels: List[str] = None,
    values: List[int] = None,
    colors: List[str] = None,
    height: int = 400,
    show_legend: bool = True,
    hole: float = 0.4,
    textinfo: str = 'label+percent',
    texttemplate: str = '%{label}<br>%{percent}'
) -> Optional[go.Figure]:
    """범용 파이/도넛 차트 생성 (공통)

    Args:
        data: 딕셔너리 형식의 데이터 {label: value, ...}
              labels/values와 상호 배타적
        labels: 라벨 리스트 (data 대신 사용 가능)
        values: 값 리스트 (data 대신 사용 가능)
        colors: 색상 리스트 (선택, 없으면 기본 색상 사용)
        height: 차트 높이 (px)
        show_legend: 범례 표시 여부
        hole: 도넛 구멍 크기 (0: 파이 차트, 0~1: 도넛 차트)
        textinfo: 텍스트 정보 표시 ('label+percent', 'value', etc.)
        texttemplate: 텍스트 템플릿

    Returns:
        Plotly Figure 객체 또는 None (데이터 없을 때)

    Examples:
        >>> # 방법 1: 딕셔너리로 전달
        >>> data = {'Category A': 100, 'Category B': 200, 'Category C': 150}
        >>> fig = create_pie_chart(data=data)

        >>> # 방법 2: 리스트로 전달
        >>> labels = ['A', 'B', 'C']
        >>> values = [100, 200, 150]
        >>> colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        >>> fig = create_pie_chart(labels=labels, values=values, colors=colors)
    """
    # 데이터 파싱
    original_labels = None
    if data is not None:
        # 딕셔너리 형식
        original_labels = list(data.keys())
        filtered_data = [(k, v) for k, v in data.items() if v > 0]
        if not filtered_data:
            return None
        labels = [item[0] for item in filtered_data]
        values = [item[1] for item in filtered_data]
    elif labels is not None and values is not None:
        # 리스트 형식
        original_labels = labels.copy()
        original_values = values.copy()
        filtered_pairs = [(l, v, i) for i, (l, v) in enumerate(zip(labels, values)) if v > 0]
        if not filtered_pairs:
            return None
        labels = [item[0] for item in filtered_pairs]
        values = [item[1] for item in filtered_pairs]
        original_indices = [item[2] for item in filtered_pairs]
    else:
        return None

    # 색상이 지정되지 않으면 기본 색상 사용
    if colors is None:
        default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                         '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#ABEBC6']
        colors = [default_colors[i % len(default_colors)] for i in range(len(labels))]
    else:
        # 색상 필터링 (값이 0인 항목 제거에 따라)
        if data is not None:
            # 딕셔너리 형식
            filtered_keys = [k for k, v in data.items() if v > 0]
            colors = [colors[list(data.keys()).index(k)] for k in filtered_keys]
        else:
            # 리스트 형식 - 원본 인덱스 기반으로 색상 매핑
            colors = [colors[i] for i in original_indices]

    # 파이 차트 생성
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=hole,
        marker=dict(
            colors=colors,
            line=dict(color='#FFFFFF', width=2)
        ),
        textinfo=textinfo,
        texttemplate=texttemplate,
        hovertemplate='<b>%{label}</b><br>건수: %{value:,}<br>비율: %{percent}<extra></extra>'
    )])

    # 레이아웃
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ) if show_legend else None
    )

    return fig


def create_harm_pie_chart(
    harm_summary: Dict[str, int],
    height: int = 400,
    show_legend: bool = True
) -> Optional[go.Figure]:
    """환자 피해 분포 파이 차트 생성 (호환성 래퍼)

    Args:
        harm_summary: 피해 요약 딕셔너리
            - total_deaths: 사망 건수
            - total_serious_injuries: 중증 부상 건수
            - total_minor_injuries: 경증 부상 건수
            - total_no_injuries: 부상 없음 건수
            - total_unknown: 알 수 없음 건수 (선택)
        height: 차트 높이 (px)
        show_legend: 범례 표시 여부

    Returns:
        Plotly Figure 객체 또는 None (데이터 없을 때)

    Example:
        >>> harm_summary = {
        ...     'total_deaths': 10,
        ...     'total_serious_injuries': 50,
        ...     'total_minor_injuries': 100,
        ...     'total_no_injuries': 200
        ... }
        >>> fig = create_harm_pie_chart(harm_summary)
        >>> st.plotly_chart(fig)
    """
    # 라벨, 값, 색상 준비
    labels = ['사망', '중증 부상', '경증 부상', '부상 없음', 'Unknown']
    values = [
        harm_summary.get('total_deaths', 0),
        harm_summary.get('total_serious_injuries', 0),
        harm_summary.get('total_minor_injuries', 0),
        harm_summary.get('total_no_injuries', 0),
        harm_summary.get('total_unknown', 0)
    ]
    colors = [
        HarmColors.DEATH,
        HarmColors.SERIOUS_INJURY,
        HarmColors.MINOR_INJURY,
        HarmColors.NO_HARM,
        HarmColors.UNKNOWN
    ]

    return create_pie_chart(
        labels=labels,
        values=values,
        colors=colors,
        height=height,
        show_legend=show_legend
    )


def create_defect_confirmed_pie_chart(
    defect_confirmed_df: pl.DataFrame,
    defect_col: str = 'defect_confirmed',
    count_col: str = 'count',
    height: int = 400,
    show_legend: bool = True
) -> Optional[go.Figure]:
    """결함 확정 분포 파이 차트 생성 (전용)

    Args:
        defect_confirmed_df: 결함 확정 데이터 DataFrame
        defect_col: 결함 확정 컬럼명
        count_col: 건수 컬럼명
        height: 차트 높이 (px)
        show_legend: 범례 표시 여부

    Returns:
        Plotly Figure 객체 또는 None (데이터 없을 때)

    Example:
        >>> fig = create_defect_confirmed_pie_chart(defect_confirmed_df)
        >>> st.plotly_chart(fig)
    """
    if defect_confirmed_df is None or len(defect_confirmed_df) == 0:
        return None

    # 데이터 추출
    labels = defect_confirmed_df[defect_col].to_list()
    values = defect_confirmed_df[count_col].to_list()

    # 색상 매핑
    color_map = {
        '결함 있음': ChartStyles.DANGER_COLOR,
        '결함 없음': ChartStyles.SUCCESS_COLOR,
        '알 수 없음': '#CCCCCC'
    }
    colors = [color_map.get(label, '#808080') for label in labels]

    return create_pie_chart(
        labels=labels,
        values=values,
        colors=colors,
        height=height,
        show_legend=show_legend,
        textinfo='percent+label',
        texttemplate='%{label}<br>%{percent}'
    )


def create_horizontal_bar_chart(
    df: pl.DataFrame,
    category_col: str,
    count_col: str = 'count',
    ratio_col: str = 'ratio',
    top_n: int = 10,
    title: Optional[str] = "",
    xaxis_title: str = "발생 건수",
    yaxis_title: Optional[str] = "",
    colorscale: str = 'Blues'
) -> Optional[go.Figure]:
    """수평 막대 차트 생성 (공통 - 부품/결함유형 등에 사용)

    Args:
        df: 데이터 DataFrame
        category_col: 카테고리 컬럼명 (부품명, 결함유형 등)
        count_col: 건수 컬럼
        ratio_col: 비율 컬럼
        top_n: 상위 N개
        title: 차트 제목 (None이면 제목 없음)
        xaxis_title: x축 제목
        yaxis_title: y축 제목 (None이면 제목 없음)
        colorscale: 색상 스케일

    Returns:
        Plotly Figure 객체 또는 None (데이터 없을 때)

    Example:
        >>> from dashboard.utils.terminology import get_term_manager
        >>> term = get_term_manager()
        >>> fig = create_horizontal_bar_chart(
        ...     df=component_df,
        ...     category_col='problem_components',
        ...     xaxis_title=term.korean.metrics.report_count,
        ...     colorscale='Blues'
        ... )
        >>> st.plotly_chart(fig)
    """
    if df is None or len(df) == 0:
        return None

    # Top N 추출
    top_df = df.head(top_n)

    # 데이터 준비
    categories = top_df[category_col].to_list()
    counts = top_df[count_col].to_list()
    ratios = top_df[ratio_col].to_list()

    # 막대 차트 생성
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts,
        y=categories,
        orientation='h',
        marker=dict(
            color=counts,
            colorscale=colorscale,
            showscale=False,
            line=dict(color='rgba(0,0,0,0.2)', width=1)
        ),
        text=[f"{r:.2f}%" for r in ratios],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>건수: %{x:,}<br>비율: %{text}<extra></extra>'
    ))

    # 레이아웃
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title if yaxis_title else "",
        height=max(400, len(top_df) * 35),
        margin=dict(l=20, r=20, t=40 if title else 20, b=40),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )

    return fig


def create_component_bar_chart(
    component_df: pl.DataFrame,
    component_col: str,
    count_col: str = 'count',
    ratio_col: str = 'ratio',
    top_n: int = 10,
    title: Optional[str] = None,
    xaxis_title: str = "발생 건수",
    yaxis_title: Optional[str] = None
) -> Optional[go.Figure]:
    """문제 부품 막대 차트 생성 (하위 호환성 유지용 래퍼)

    Args:
        component_df: 부품 데이터 DataFrame
        component_col: 부품명 컬럼
        count_col: 건수 컬럼
        ratio_col: 비율 컬럼
        top_n: 상위 N개
        title: 차트 제목
        xaxis_title: x축 제목
        yaxis_title: y축 제목

    Returns:
        Plotly Figure 객체 또는 None (데이터 없을 때)
    """
    return create_horizontal_bar_chart(
        df=component_df,
        category_col=component_col,
        count_col=count_col,
        ratio_col=ratio_col,
        top_n=top_n,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        colorscale='Blues'
    )


# ==================== 메트릭 표시 ====================

def render_metrics_row(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """메트릭 행 렌더링 (공통)

    Args:
        metrics: 메트릭 딕셔너리 리스트
            - label: 라벨
            - value: 값
            - delta: 변화량 (선택)
            - delta_color: 'normal', 'inverse', 'off' (선택)
            - help: 도움말 텍스트 (선택)
        columns: 컬럼 수

    Example:
        >>> metrics = [
        ...     {"label": "총 보고 건수", "value": "1,000건", "delta": "+10%"},
        ...     {"label": "사망률", "value": "5.2%", "delta": "-0.5%p", "delta_color": "inverse"}
        ... ]
        >>> render_metrics_row(metrics, columns=4)
    """
    cols = st.columns(columns)

    for i, metric in enumerate(metrics[:columns]):  # 최대 columns 개까지만
        with cols[i]:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", "N/A"),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
                help=metric.get("help")
            )


# ==================== 데이터 다운로드 ====================

def render_download_button(
    data: pl.DataFrame,
    filename_prefix: str = "data",
    label: str = None,
    key: str = None
) -> None:
    """CSV 다운로드 버튼 렌더링 (공통)

    Args:
        data: Polars DataFrame
        filename_prefix: 파일명 접두사
        label: 버튼 라벨 (기본값: "📥 CSV 다운로드")
        key: Streamlit 위젯 키

    Example:
        >>> render_download_button(
        ...     data=result_df,
        ...     filename_prefix="total_reports",
        ...     key="download_total"
        ... )
    """
    import pandas as pd
    from datetime import datetime

    if data is None or len(data) == 0:
        return

    # Polars → Pandas 변환
    if isinstance(data, pl.DataFrame):
        pdf = data.to_pandas()
    elif isinstance(data, pl.LazyFrame):
        pdf = data.collect().to_pandas()
    else:
        pdf = data

    # CSV 생성
    csv_data = pdf.to_csv(index=False, encoding='utf-8-sig')

    # 파일명
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"

    # 버튼 라벨
    if label is None:
        label = "📥 CSV 다운로드"

    # 다운로드 버튼
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=key
    )


# ==================== 북마크 관리 ====================

def apply_pending_bookmark(tab_name: str) -> dict:
    """사이드바 렌더링 전에 pending 북마크를 반환

    Args:
        tab_name: 탭 이름 (예: "eda", "spike")

    Returns:
        pending 북마크 데이터 (없으면 빈 dict)

    Note:
        Home.py에서 사이드바 렌더링 전에 호출하여 dynamic_options로 전달
    """
    pending_key = f"{tab_name}_pending_bookmark"
    if pending_key in st.session_state:
        bookmark_data = st.session_state[pending_key]
        del st.session_state[pending_key]
        return bookmark_data
    return {}


def render_bookmark_manager(
    tab_name: str,
    current_filters: dict,
    filter_keys: list
) -> None:
    """북마크 관리 UI (모든 탭 공통)

    Args:
        tab_name: 탭 이름 (예: "eda", "spike")
        current_filters: 현재 사이드바 필터 상태
        filter_keys: 북마크할 필터 키 리스트
            예: ["date_range", "manufacturers", "products", "top_n"]

    Example:
        >>> render_bookmark_manager(
        ...     tab_name="eda",
        ...     current_filters=filters,
        ...     filter_keys=["date_range", "manufacturers", "products", "top_n", "min_cases"]
        ... )
    """
    # 북마크 저장소 초기화
    bookmark_key = f"{tab_name}_bookmarks"
    if bookmark_key not in st.session_state:
        st.session_state[bookmark_key] = {}

    with st.expander("🔖 필터 북마크 관리"):
        col1, col2 = st.columns([3, 1])

        with col1:
            bookmark_name = st.text_input(
                "북마크 이름",
                key=f"{tab_name}_bookmark_name",
                placeholder="예: 2024년 제조사A 분석"
            )

        with col2:
            st.write("")  # 정렬용
            st.write("")
            if st.button("💾 저장", key=f"{tab_name}_save_bookmark", use_container_width=True):
                if bookmark_name:
                    # 현재 필터 상태를 북마크로 저장
                    bookmark_data = {key: current_filters.get(key) for key in filter_keys}
                    st.session_state[bookmark_key][bookmark_name] = bookmark_data
                    st.success(f"'{bookmark_name}' 저장 완료!")
                else:
                    st.warning("북마크 이름을 입력해주세요.")

        # 저장된 북마크 목록
        if st.session_state[bookmark_key]:
            st.markdown("---")
            st.markdown("**📚 저장된 북마크**")

            for name in list(st.session_state[bookmark_key].keys()):
                col_name, col_load, col_del = st.columns([4, 1, 1])

                with col_name:
                    # 북마크 정보 표시
                    bookmark = st.session_state[bookmark_key][name]
                    info_parts = []

                    # 날짜 범위
                    if "date_range" in bookmark and bookmark["date_range"]:
                        date_range = bookmark["date_range"]
                        if isinstance(date_range, tuple) and len(date_range) == 2:
                            start, end = date_range
                            if hasattr(start, 'strftime'):
                                info_parts.append(f"📅 {start.strftime('%Y-%m')}~{end.strftime('%Y-%m')}")

                    # 기준 월 (Spike)
                    if "as_of_month" in bookmark and bookmark["as_of_month"]:
                        info_parts.append(f"📅 {bookmark['as_of_month']}")

                    # 제조사/제품 수
                    if "manufacturers" in bookmark and bookmark["manufacturers"]:
                        info_parts.append(f"🏭 {len(bookmark['manufacturers'])}개")
                    if "products" in bookmark and bookmark["products"]:
                        info_parts.append(f"📦 {len(bookmark['products'])}개")

                    # 기타 파라미터
                    param_map = {
                        "top_n": "Top",
                        "min_cases": "Min",
                        "window": "Window",
                        "z_threshold": "Z",
                        "min_methods": "Methods"
                    }
                    for key, label in param_map.items():
                        if key in bookmark and bookmark[key] is not None:
                            info_parts.append(f"{label}={bookmark[key]}")

                    info_text = " · ".join(info_parts) if info_parts else "(정보 없음)"
                    st.markdown(f"**{name}**  \n`{info_text}`")

                with col_load:
                    if st.button("📂", key=f"{tab_name}_load_{name}", help="불러오기"):
                        # 위젯이 렌더링되기 전에 값을 설정하기 위해 먼저 session_state에 저장
                        bookmark_to_load = st.session_state[bookmark_key][name]

                        # 임시 플래그 설정 (다음 rerun 시 적용하기 위함)
                        st.session_state[f"{tab_name}_pending_bookmark"] = bookmark_to_load
                        st.success(f"'{name}' 불러오기 완료!")
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"{tab_name}_delete_{name}", help="삭제"):
                        del st.session_state[bookmark_key][name]
                        st.rerun()


# ==================== 섹션 헤더 ====================

def render_section_header(
    title: str,
    icon: str = "",
    caption: str = None,
    divider: bool = True
) -> None:
    """섹션 헤더 렌더링 (공통)

    Args:
        title: 제목
        icon: 아이콘 이모지
        caption: 캡션 (작은 설명)
        divider: 구분선 표시 여부

    Example:
        >>> render_section_header(
        ...     title="누적 보고서 수",
        ...     icon="📊",
        ...     caption="최근 12개월 데이터"
        ... )
    """
    full_title = f"{icon} {title}" if icon else title
    st.subheader(full_title)

    if caption:
        st.caption(caption)

    if divider:
        st.markdown("---")


# ==================== HTML 차트 ====================

def create_html_bar_chart(
    data: pl.DataFrame,
    item_col: str,
    value_col: str,
    ratio_col: str = None,
    top_n: int = 10,
    height_per_item: int = 55
) -> str:
    """HTML 스타일 막대 차트 생성 (Config 기반)

    Args:
        data: 데이터프레임
        item_col: 항목 컬럼명
        value_col: 값 컬럼명
        ratio_col: 비율 컬럼명 (선택)
        top_n: 상위 N개
        height_per_item: 항목당 높이 (px)

    Returns:
        HTML 문자열

    Example:
        >>> html = create_html_bar_chart(
        ...     data=df,
        ...     item_col='manufacturer_name',
        ...     value_col='count',
        ...     ratio_col='ratio',
        ...     top_n=10
        ... )
        >>> st.markdown(html, unsafe_allow_html=True)
    """
    cfg = get_config()
    ui_standards = cfg.ui_standards

    # 스타일 설정 가져오기
    bar_styles = ui_standards.get('html_chart_styles', {}).get('bar_chart', {})
    container_styles = ui_standards.get('html_chart_styles', {}).get('scrollable_container', {})

    # 기본값
    bar_height = bar_styles.get('bar_height', 45)
    border_radius = bar_styles.get('border_radius', 20)
    gradient_start = bar_styles.get('gradient_start', '#3B82F6')
    gradient_end = bar_styles.get('gradient_end', '#2563EB')
    background = bar_styles.get('background', '#F3F4F6')
    text_color = bar_styles.get('text_color', '#374151')
    hover_transform = bar_styles.get('hover_transform', 'translateX(3px)')
    shadow = bar_styles.get('shadow', '0 2px 4px rgba(59, 130, 246, 0.3)')

    max_visible = container_styles.get('max_visible_items', 10)
    scrollbar_width = container_styles.get('scrollbar_width', 8)
    scrollbar_color = container_styles.get('scrollbar_color', '#888')
    scrollbar_hover = container_styles.get('scrollbar_hover', '#555')
    scrollbar_track = container_styles.get('scrollbar_track', '#f1f1f1')

    # 데이터 준비
    top_data = data.head(top_n)

    if len(top_data) == 0:
        return "<p>데이터가 없습니다.</p>"

    # 최대값 계산
    max_value = top_data[value_col].max()

    # HTML 생성
    html_parts = [f"""
    <style>
        .html-bar-container {{
            max-height: {max_visible * height_per_item}px;
            overflow-y: auto;
            padding-right: 10px;
        }}
        .html-bar-container::-webkit-scrollbar {{
            width: {scrollbar_width}px;
        }}
        .html-bar-container::-webkit-scrollbar-track {{
            background: {scrollbar_track};
            border-radius: 10px;
        }}
        .html-bar-container::-webkit-scrollbar-thumb {{
            background: {scrollbar_color};
            border-radius: 10px;
        }}
        .html-bar-container::-webkit-scrollbar-thumb:hover {{
            background: {scrollbar_hover};
        }}
        .html-bar-item {{
            height: {height_per_item}px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            position: relative;
        }}
        .html-bar {{
            height: {bar_height}px;
            background: linear-gradient(90deg, {gradient_start}, {gradient_end});
            border-radius: {border_radius}px;
            transition: all 0.3s ease;
            box-shadow: {shadow};
            display: flex;
            align-items: center;
            padding: 0 15px;
            position: relative;
        }}
        .html-bar:hover {{
            transform: {hover_transform};
            box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
        }}
        .html-bar-label {{
            position: absolute;
            left: 15px;
            color: white;
            font-weight: 600;
            font-size: 14px;
            z-index: 2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 70%;
        }}
        .html-bar-value {{
            position: absolute;
            right: 15px;
            color: white;
            font-weight: 700;
            font-size: 13px;
            z-index: 2;
        }}
        .html-bar-background {{
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: {background};
            border-radius: {border_radius}px;
            z-index: 0;
        }}
    </style>
    <div class="html-bar-container">
    """]

    # 각 항목에 대한 막대 생성
    for row in top_data.iter_rows(named=True):
        item = row[item_col]
        value = row[value_col]
        ratio = row.get(ratio_col, 0) if ratio_col else 0

        # 퍼센트 계산
        percent = (value / max_value * 100) if max_value > 0 else 0

        # 값 표시
        if ratio_col and ratio > 0:
            value_text = f"{value:,}건 ({ratio:.2f}%)"
        else:
            value_text = f"{value:,}건"

        html_parts.append(f"""
        <div class="html-bar-item">
            <div class="html-bar-background" style="width: 100%;"></div>
            <div class="html-bar" style="width: {percent}%;">
                <span class="html-bar-label">{item}</span>
                <span class="html-bar-value">{value_text}</span>
            </div>
        </div>
        """)

    html_parts.append("</div>")

    return "".join(html_parts)

# spike_tab.py
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional

from dashboard.utils.analysis import perform_spike_detection, get_spike_time_series
from dashboard.utils.constants import ColumnNames
from dashboard.utils.ui_components import render_filter_summary_badge, render_spike_filter_summary, render_bookmark_manager


def show(filters=None, lf: pl.LazyFrame = None):
    """
    Spike Detection 탭

    Args:
        filters: SidebarManager에서 생성된 필터 딕셔너리
            - date_range: (start_date, end_date) 튜플
            - as_of_month: 기준 월 (예: "2025-11")
            - window: 윈도우 크기 (1 또는 3)
            - min_c_recent: 최소 최근 케이스 수
            - z_threshold: Z-score 임계값
            - eps: Epsilon 값
            - alpha: 유의수준
            - correction: 다중검정 보정 방법
            - min_methods: 앙상블 최소 방법 수
        lf: MAUDE 데이터 LazyFrame
    """
    from dashboard.utils.constants import DisplayNames

    st.title(DisplayNames.FULL_TITLE_SPIKE)

    if lf is None:
        st.warning("데이터가 로드되지 않았습니다.")
        return

    # 필터가 없으면 기본값 사용
    if filters is None:
        filters = {}

    # 필터 값 추출 - 급증 탐지 파라미터
    as_of_month = filters.get('as_of_month', '2025-11')
    window = filters.get('window', 1)
    min_c_recent = filters.get('min_c_recent', 20)
    z_threshold = filters.get('z_threshold', 2.0)
    eps = filters.get('eps', 0.1)
    alpha = filters.get('alpha', 0.05)
    correction = filters.get('correction', 'fdr_bh')
    min_methods = filters.get('min_methods', 2)

    # 공통 필터 추출
    manufacturers = filters.get("manufacturers", [])
    products = filters.get("products", [])
    devices = filters.get("devices", [])
    defect_types = filters.get("defect_types", [])
    clusters = filters.get("clusters", [])

    # ==================== 북마크 관리 ====================
    render_bookmark_manager(
        tab_name="spike",
        current_filters=filters,
        filter_keys=[
            "as_of_month", "window", "min_c_recent", "z_threshold",
            "eps", "alpha", "correction", "min_methods",
            "manufacturers", "products", "devices", "defect_types", "clusters"
        ]
    )

    # ==================== 필터 요약 배지 (Spike 전용) ====================
    render_filter_summary_badge(
        manufacturers=manufacturers,
        products=products,
        devices=devices,
        defect_types=defect_types,
        clusters=clusters
    )
    
    render_spike_filter_summary(
        as_of_month=as_of_month,
        window=window,
        min_c_recent=min_c_recent,
        z_threshold=z_threshold,
        alpha=alpha,
        correction=correction,
        min_methods=min_methods
    )
    st.markdown("---")

    # 공통 필터 적용
    from dashboard.utils.filter_helpers import apply_common_filters
    filtered_lf = apply_common_filters(
        lf,
        manufacturers=manufacturers,
        products=products,
        devices=devices,
        defect_types=defect_types,
        clusters=clusters
    )

    # 스파이크 탐지 수행 (필터링된 데이터로 계산)
    with st.spinner("급증 탐지 분석 중..."):
        result_df = outlier_detect_check(
            lf=filtered_lf,
            window=window,
            min_c_recent=min_c_recent,
            z_threshold=z_threshold,
            eps=eps,
            alpha=alpha,
            correction=correction,
            min_methods=min_methods,
            month=as_of_month,
            manufacturers=tuple(manufacturers) if manufacturers else (),
            products=tuple(products) if products else ()
        )

    if result_df is None or len(result_df) == 0:
        st.info("분석할 데이터가 없습니다.")
        return

    # 급증 키워드만 필터링 (앙상블 기준)
    spike_df = result_df.filter(pl.col("is_spike_ensemble") == True)

    # ========================================
    # 💡 SECTION 0: 핵심 인사이트 (최상단 배치) - terminology 기반
    # ========================================
    from dashboard.utils.terminology import get_term_manager

    term = get_term_manager()
    st.subheader("💡 핵심 인사이트")

    if len(spike_df) > 0:
        # 1️⃣ 가장 위험한 급증 (3개 방법 모두 동의)
        critical_spikes = spike_df.filter(pl.col("n_methods") == 3).sort("ratio", descending=True)

        if len(critical_spikes) > 0:
            top_critical = critical_spikes.head(1)
            keyword = top_critical["keyword"][0]
            ratio = top_critical["ratio"][0]
            c_recent = top_critical["C_recent"][0]
            c_base = top_critical["C_base"][0]

            st.error(term.format_message('spike_critical',
                                        keyword=keyword,
                                        c_base=c_base,
                                        c_recent=c_recent,
                                        ratio=ratio))

        # 2️⃣ 새롭게 등장한 급증 (이전 기간엔 없었던 키워드)
        new_spikes = spike_df.filter(pl.col("C_base") < 5)  # 기준 기간에 거의 없었던 키워드
        if len(new_spikes) > 0:
            new_count = len(new_spikes)
            new_keywords = new_spikes.head(3)["keyword"].to_list()
            st.warning(term.format_message('spike_new',
                                          new_count=new_count,
                                          examples=', '.join(new_keywords)))

        # 3️⃣ 패턴별 요약
        severe_count = len(spike_df.filter(pl.col("pattern") == "severe"))
        if severe_count > 0:
            st.warning(term.format_message('spike_severe_pattern', severe_count=severe_count))
        else:
            alert_count = len(spike_df.filter(pl.col("pattern") == "alert"))
            if alert_count > 0:
                st.info(term.format_message('spike_alert_pattern', alert_count=alert_count))
    else:
        st.success(term.messages.get('spike_none'))
        st.info(term.messages.get('spike_none_good'))

    st.markdown("---")

    # ========================================
    # 🚨 SECTION 1: 급증 탐지 요약
    # ========================================
    st.subheader("🚨 급증 탐지 요약")

    # 주요 메트릭
    col_main1, col_main2, col_main3 = st.columns([2, 2, 3])

    with col_main1:
        st.metric(
            label="⚠️ 탐지된 급증",
            value=f"{len(spike_df)}개",
            delta=f"전체 {len(result_df)}개 중",
            help="앙상블 방법으로 탐지된 급증 키워드 수"
        )

    with col_main2:
        if len(spike_df) > 0:
            avg_methods = spike_df["n_methods"].mean()
            st.metric(
                label="📊 평균 탐지 방법 수",
                value=f"{avg_methods:.2f}개",
                help="Ratio/Z-score/Poisson 중 몇 개의 방법이 급증으로 판정했는지"
            )
        else:
            st.metric(label="📊 평균 탐지 방법 수", value="N/A")

    with col_main3:
        if len(spike_df) > 0:
            max_ratio_row = spike_df.sort("ratio", descending=True).head(1)
            max_keyword = max_ratio_row["keyword"][0]
            max_ratio = max_ratio_row["ratio"][0]
            st.metric(
                label="🔥 최대 급증 키워드",
                value=max_keyword,
                delta=f"{max_ratio:.2f}x 증가",
                help="기준 기간 대비 가장 많이 증가한 키워드"
            )
        else:
            st.metric(label="🔥 최대 급증 키워드", value="없음")

    # 패턴별 분포
    st.markdown("**패턴별 분포**")
    pattern_counts = result_df.group_by("pattern").agg(pl.len().alias("count")).sort("count", descending=True)

    col1, col2, col3, col4 = st.columns(4)
    pattern_map = {
        "severe": ("🔴 심각", col1),
        "alert": ("🟠 경고", col2),
        "attention": ("🟡 주의", col3),
        "general": ("🟢 일반", col4)
    }

    for pattern, (label, col) in pattern_map.items():
        count = pattern_counts.filter(pl.col("pattern") == pattern)
        count_val = count["count"][0] if len(count) > 0 else 0
        col.metric(label, count_val)

    st.markdown("---")

    # 시계열 데이터 준비 (12개월)
    end_date = datetime.strptime(as_of_month, "%Y-%m")
    start_date = end_date - relativedelta(months=11)
    start_month = start_date.strftime("%Y-%m")

    # ========================================
    # 📈 SECTION 2: 시계열 차트 (시각화)
    # ========================================
    st.subheader("📈 키워드 비율 추이 (Anomaly Detection)")

    # 전체 키워드 목록 (ratio 기준 내림차순)
    all_keywords = result_df.sort("ratio", descending=True)["keyword"].to_list()
    spike_keywords = spike_df.sort("ratio", descending=True)["keyword"].to_list() if len(spike_df) > 0 else []
    severe_keywords = result_df.filter(pl.col("pattern") == "severe").sort("ratio", descending=True)["keyword"].to_list()
    alert_keywords = result_df.filter(pl.col("pattern") == "alert").sort("ratio", descending=True)["keyword"].to_list()

    # # 🔍 디버그: 날짜-제조사-제품군-키워드 매핑
    # st.write("### 🔍 디버그: 날짜-제조사-제품군-키워드 매핑")
    # debug_mapping = (
    #     filtered_lf
    #     .select([
    #         pl.col("date_received").dt.truncate("1mo").alias("month"),
    #         pl.col(ColumnNames.MANUFACTURER).alias("manufacturer"),
    #         pl.col(ColumnNames.PRODUCT_CODE).alias("product"),
    #         pl.col(ColumnNames.DEFECT_TYPE).alias("keyword")
    #     ])
    #     .filter(pl.col("keyword").is_in(all_keywords[:20]))  # 상위 20개 키워드만
    #     .group_by(["month", "manufacturer", "product", "keyword"])
    #     .agg(pl.len().alias("count"))
    #     .sort(["month", "keyword", "count"], descending=[True, False, True])
    #     .head(100)
    #     .collect()
    # )
    # st.write(f"**매핑 데이터 (상위 100행)**:")
    # st.dataframe(debug_mapping, height=300)

    # 빠른 선택 버튼
    st.markdown("**🔘 빠른 선택**")
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)

    # 세션 스테이트 초기화 (초기값만 설정, default 파라미터 사용 안 함)
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = all_keywords[:min(5, len(all_keywords))]

    with col_btn1:
        severe_count = len(severe_keywords)
        if st.button(f"🔴 심각 ({severe_count})", use_container_width=True, help=f"심각 패턴 키워드 {severe_count}개 중 최대 10개 선택"):
            st.session_state.selected_keywords = severe_keywords[:10]
            st.rerun()

    with col_btn2:
        alert_count = len(alert_keywords)
        if st.button(f"🟠 경고 ({alert_count})", use_container_width=True, help=f"경고 패턴 키워드 {alert_count}개 중 최대 10개 선택"):
            st.session_state.selected_keywords = alert_keywords[:10]
            st.rerun()

    with col_btn3:
        spike_count = len(spike_keywords)
        if st.button(f"⚠️ 급증 ({spike_count})", use_container_width=True, help=f"급증으로 탐지된 키워드 {spike_count}개 중 최대 10개 선택"):
            st.session_state.selected_keywords = spike_keywords[:10]
            st.rerun()

    with col_btn4:
        if st.button("🔝 Top 10", use_container_width=True, help="비율 상위 10개 키워드 선택"):
            st.session_state.selected_keywords = all_keywords[:10]
            st.rerun()

    with col_btn5:
        if st.button("🔄 초기화", use_container_width=True, help="기본값(상위 5개)으로 초기화"):
            st.session_state.selected_keywords = all_keywords[:5]
            st.rerun()

    # 키워드 멀티셀렉트 (세션 스테이트와 연동, default 제거하여 경고 방지)
    selected_keywords = st.multiselect(
        "🔍 표시할 키워드 선택",
        options=all_keywords,
        key="selected_keywords",
        help="차트에 표시할 키워드를 선택하세요"
    )

    # 선택된 키워드로 시계열 데이터 가져오기
    if len(selected_keywords) > 0:
        ts_df_filtered = get_spike_time_series(
            _lf=filtered_lf,
            keywords=selected_keywords,
            start_month=start_month,
            end_month=as_of_month,
            window=window
        )

        if len(ts_df_filtered) > 0:
            fig = create_spike_chart(ts_df_filtered, z_threshold, as_of_month, window)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("선택한 키워드에 대한 시계열 데이터가 없습니다.")
    else:
        st.info("차트를 표시할 키워드를 선택해주세요.")

    # ========================================
    # 📋 SECTION 3: 상세 테이블 (상세 정보)
    # ========================================
    st.subheader("📋 전체 분석 결과")

    # 필터 도움말
    with st.expander("ℹ️ 필터 사용 방법", expanded=False):
        st.markdown("""
        **📊 패턴 필터**: 표시할 패턴 유형 선택 (심각/경고/주의/일반)

        **⚠️ 급증만 체크박스**:
        - ✅ 체크: 앙상블 방법으로 **급증 판정**된 키워드만 표시
        - ☐ 미체크: 선택한 패턴의 **전체 키워드** 표시 (급증 아닌 것도 포함)

        **🔘 빠른 필터 프리셋**:
        - 🔴 **Critical만**: 심각 패턴 + 급증만 (가장 위험한 항목)
        - ⚠️ **주의 필요**: 심각+경고 패턴 전체 (급증 아닌 것도 포함)
        - 📊 **전체 급증**: 모든 패턴의 급증만 (패턴 무관하게 급증 판정된 것)
        - 🔄 **초기화**: 기본 설정 (심각+경고+주의 패턴 전체)
        """)

    # 빠른 프리셋 버튼
    st.markdown("**🔘 빠른 필터**")
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)

    # 세션 상태 초기화
    if 'table_pattern_filter' not in st.session_state:
        st.session_state.table_pattern_filter = ["severe", "alert", "attention"]
    if 'table_spike_only' not in st.session_state:
        st.session_state.table_spike_only = False

    with col_preset1:
        if st.button("🔴 Critical만", use_container_width=True, help="심각 패턴 + 급증만 표시"):
            st.session_state.table_pattern_filter = ["severe"]
            st.session_state.table_spike_only = True
            st.rerun()

    with col_preset2:
        if st.button("⚠️ 주의 필요", use_container_width=True, help="심각 + 경고 패턴 전체"):
            st.session_state.table_pattern_filter = ["severe", "alert"]
            st.session_state.table_spike_only = False
            st.rerun()

    with col_preset3:
        if st.button("📊 전체 급증", use_container_width=True, help="모든 패턴의 급증만"):
            st.session_state.table_pattern_filter = ["severe", "alert", "attention", "general"]
            st.session_state.table_spike_only = True
            st.rerun()

    with col_preset4:
        if st.button("🔄 초기화", use_container_width=True, help="기본 설정으로 복원"):
            st.session_state.table_pattern_filter = ["severe", "alert", "attention"]
            st.session_state.table_spike_only = False
            st.rerun()

    # 테이블 필터 (세션 상태 연동)
    col_pattern, col_spike_only, col_topn = st.columns([3, 1, 1])
    with col_pattern:
        pattern_filter = st.multiselect(
            "📊 패턴 필터",
            options=["severe", "alert", "attention", "general"],
            default=st.session_state.table_pattern_filter,
            format_func=lambda x: {
                "severe": "🔴 심각",
                "alert": "🟠 경고",
                "attention": "🟡 주의",
                "general": "🟢 일반"
            }[x],
            key="pattern_filter_table"
        )
        # 선택값을 세션 상태에 저장
        st.session_state.table_pattern_filter = pattern_filter

    with col_spike_only:
        show_spike_only = st.checkbox(
            "⚠️ 급증만",
            value=st.session_state.table_spike_only,
            help="앙상블 급증으로 판정된 키워드만 표시",
            key="spike_only_checkbox"
        )
        # 선택값을 세션 상태에 저장
        st.session_state.table_spike_only = show_spike_only

    with col_topn:
        top_n_table = st.number_input(
            "표시 행 수",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="top_n_table"
        )

    # 필터링된 결과 테이블
    filtered_result = result_df.filter(pl.col("pattern").is_in(pattern_filter))
    if show_spike_only:
        filtered_result = filtered_result.filter(pl.col("is_spike_ensemble") == True)

    display_all_df = prepare_spike_table(filtered_result.head(top_n_table))

    if len(display_all_df) > 0:
        st.dataframe(display_all_df, width='stretch', height=600)
    else:
        st.info("필터 조건에 맞는 데이터가 없습니다.")

    # ========================================
    # 📥 SECTION 4: 데이터 다운로드
    # ========================================
    st.markdown("---")
    col_download1, col_download2 = st.columns(2)

    with col_download1:
        st.markdown("**📥 전체 분석 결과 다운로드**")
        csv_all = result_df.write_csv()
        st.download_button(
            label="전체 결과 CSV 다운로드",
            data=csv_all,
            file_name=f"spike_detection_all_{as_of_month}_w{window}.csv",
            mime="text/csv"
        )

    with col_download2:
        if len(spike_df) > 0:
            st.markdown("**📥 급증만 다운로드**")
            csv_spike = spike_df.write_csv()
            st.download_button(
                label="급증만 CSV 다운로드",
                data=csv_spike,
                file_name=f"spike_detection_spikes_{as_of_month}_w{window}.csv",
                mime="text/csv"
            )

def outlier_detect_check(
    lf: pl.LazyFrame,
    window: int = 1,
    min_c_recent: int = 20,
    z_threshold: float = 2.0,
    eps: float = 0.1,
    alpha: float = 0.05,
    correction: str = 'fdr_bh',
    min_methods: int = 2,
    month: str = "2025-11",
    manufacturers: tuple = None,
    products: tuple = None,
) -> Optional[pl.DataFrame]:
    """
    스파이크 탐지 분석 수행

    Args:
        lf: MAUDE 데이터 LazyFrame (이미 공통 필터 적용됨)
        window: 윈도우 크기 (1 또는 3)
        min_c_recent: 최소 최근 케이스 수
        z_threshold: Z-score 임계값
        eps: Epsilon 값 (z_log 계산용)
        alpha: 유의수준 (Poisson 검정용)
        correction: 다중검정 보정 방법 ('bonferroni', 'sidak', 'fdr_bh', None)
        min_methods: 앙상블 스파이크 판정 최소 방법 수
        month: 기준 월 (예: "2025-11")
        manufacturers: 제조사 필터 (캐시 키용)
        products: 제품군 필터 (캐시 키용)

    Returns:
        스파이크 탐지 결과 DataFrame
        컬럼: keyword, C_recent, C_base, ratio, z_log, score_pois,
              is_spike, is_spike_z, is_spike_p, n_methods, is_spike_ensemble, pattern
    """
    result_df = perform_spike_detection(
        _lf=lf,
        as_of_month=month,
        window=window,
        min_c_recent=min_c_recent,
        z_threshold=z_threshold,
        eps=eps,
        alpha=alpha,
        correction=correction,
        min_methods=min_methods,
        manufacturers=manufacturers,
        products=products,
    )

    return result_df


def create_spike_chart(
    ts_df: pl.DataFrame,
    z_threshold: float,
    as_of_month: str,
    window: int
) -> go.Figure:
    """
    스파이크 시계열 차트 생성

    Args:
        ts_df: 시계열 데이터 (columns: month, keyword, count, ratio)
        z_threshold: Z-score 임계값 (표시용)
        as_of_month: 기준 월
        window: 윈도우 크기

    Returns:
        Plotly Figure 객체
    """
    import plotly.express as px
    from src import BaselineAggregator

    fig = go.Figure()

    # 키워드별로 라인 추가
    keywords = ts_df["keyword"].unique().to_list()

    # 동적 색상 생성 (Plotly의 qualitative 색상 팔레트 사용)
    n_colors = len(keywords)
    if n_colors <= 10:
        colors = px.colors.qualitative.Plotly
    elif n_colors <= 24:
        colors = px.colors.qualitative.Dark24
    else:
        # 많은 키워드의 경우 색상 반복
        colors = px.colors.qualitative.Dark24 * ((n_colors // 24) + 1)

    for i, keyword in enumerate(keywords):
        keyword_data = ts_df.filter(pl.col("keyword") == keyword).sort("month")

        # 월 문자열을 월 중간 날짜로 변환 (예: "2024-11" → "2024-11-15")
        months = keyword_data["month"].to_list()
        month_mid_dates = [f"{m}-15" for m in months]

        fig.add_trace(go.Scatter(
            x=month_mid_dates,
            y=keyword_data["ratio"].to_list(),
            mode='lines+markers',
            name=keyword,
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Ratio: %{y:.2f}x<br>' +
                         '<extra></extra>'
        ))

    # y축 범위 계산
    max_ratio = ts_df["ratio"].max() if len(ts_df) > 0 else z_threshold
    y_max = max(max_ratio, z_threshold) + 0.5

    # 레이아웃 설정
    fig.update_layout(
        title=f"급증 탐지 - 키워드 비율 추이 (윈도우: {window}개월, 임계값: {z_threshold:.2f}σ)",
        xaxis_title="월",
        yaxis_title="비율 (배수)",
        yaxis=dict(range=[0, y_max]),
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=50, r=150, t=80, b=50)
    )

    # z-score 임계값 표시
    fig.add_hline(
        y=z_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Z-score 임계값 ({z_threshold}σ)",
        annotation_position="right",
        annotation_font=dict(color="red", size=10)
    )

    # 기준 구간과 비교 구간 시각적 표시
    if len(ts_df) > 0:
        all_months = sorted(ts_df["month"].unique().to_list())

        # BaselineAggregator를 사용하여 구간 계산
        # BaselineAggregator 인스턴스 생성 (더미)
        dummy_agg = BaselineAggregator(ts_df.lazy())

        # _get_window_months() 메서드로 구간 계산
        recent_months, baseline_months = dummy_agg._get_window_months(as_of_month, window)

        # 실제 데이터에 있는 월만 필터링
        baseline_months_in_data = [m for m in baseline_months if m in all_months]
        comparison_months_in_data = [m for m in recent_months if m in all_months]

        # 기준 구간 (파란색)
        if baseline_months_in_data:
            baseline_sorted = sorted(baseline_months_in_data)

            # months[0] ~ months[-1] + 1m
            x0_month = baseline_sorted[0]
            last_date = datetime.strptime(baseline_sorted[-1], '%Y-%m')
            x1_date = last_date + relativedelta(months=1)
            x1_month = x1_date.strftime('%Y-%m')

            fig.add_vrect(
                x0=x0_month,
                x1=x1_month,
                fillcolor="lightblue",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="기준" if window == 1 else "기준 구간",
                annotation_position="top left",
                annotation_font=dict(size=10, color="blue")
            )

        # 비교 구간 (주황색)
        if comparison_months_in_data:
            comparison_sorted = sorted(comparison_months_in_data)

            # months[0] ~ months[-1] + 1m
            x0_month = comparison_sorted[0]
            last_date = datetime.strptime(comparison_sorted[-1], '%Y-%m')
            x1_date = last_date + relativedelta(months=1)
            x1_month = x1_date.strftime('%Y-%m')

            fig.add_vrect(
                x0=x0_month,
                x1=x1_month,
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="비교" if window == 1 else "비교 구간",
                annotation_position="top right",
                annotation_font=dict(size=10, color="darkorange")
            )

    return fig


def prepare_spike_table(spike_df: pl.DataFrame) -> pl.DataFrame:
    """
    스파이크 테이블 표시용 데이터 준비 (중요 컬럼 우선 배치)

    Args:
        spike_df: 스파이크 탐지 결과 DataFrame

    Returns:
        표시용 DataFrame
    """
    # 패턴에 이모지 추가
    pattern_emoji = (
        pl.when(pl.col("pattern") == "severe").then(pl.lit("🔴 심각"))
        .when(pl.col("pattern") == "alert").then(pl.lit("🟠 경고"))
        .when(pl.col("pattern") == "attention").then(pl.lit("🟡 주의"))
        .otherwise(pl.lit("🟢 일반"))
    )

    # 증감 계산 (signed int로 명시적 캐스팅하여 음수 오버플로우 방지)
    increase = (pl.col("C_recent").cast(pl.Int64) - pl.col("C_base").cast(pl.Int64))

    # 컬럼 순서: 중요한 정보 우선 (키워드 → 패턴 → 비율 → 증감 → 방법 수 → 상세)
    display_df = spike_df.select([
        pl.col("keyword").alias("키워드"),
        pattern_emoji.alias("패턴"),
        pl.col("ratio").round(2).alias("비율 (배수)"),
        pl.col("C_recent").alias("최근 보고수"),
        increase.alias("증감"),
        pl.col("C_base").alias("기준 보고수"),
        pl.col("n_methods").alias("탐지방법수"),
        pl.col("is_spike").alias("✓Ratio"),
        pl.col("is_spike_z").alias("✓Z-score"),
        pl.col("is_spike_p").alias("✓Poisson"),
    ])

    return display_df
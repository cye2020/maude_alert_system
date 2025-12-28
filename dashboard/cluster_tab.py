# cluster_tab.py (전면 개선 버전)
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.analysis_cluster import cluster_check, get_available_clusters
from utils.constants import ColumnNames, Defaults, ChartStyles, DisplayNames, HarmColors, Terms
from utils.data_utils import get_year_month_expr
from dashboard.utils.ui_components import (
    render_filter_summary_badge,
    convert_date_range_to_months,
    create_harm_pie_chart,
    create_defect_confirmed_pie_chart,
    create_horizontal_bar_chart
)


def show(filters=None, lf: pl.LazyFrame = None):
    """클러스터 분석 탭 메인 함수 (전면 개선)

    Args:
        filters: 사이드바 필터 값
        lf: LazyFrame 데이터
    """
    from utils.constants import DisplayNames

    st.title(DisplayNames.FULL_TITLE_CLUSTER)

    # 데이터 확인
    if lf is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    # ==================== 사이드바 필터 추출 ====================
    date_range = filters.get("date_range", None)

    # 공통 필터 추출
    manufacturers = filters.get("manufacturers", [])
    products = filters.get("products", [])
    devices = filters.get("devices", [])
    defect_types = filters.get("defect_types", [])
    clusters = filters.get("clusters", [])

    # date_range를 문자열 리스트로 변환 (공통 함수 사용)
    selected_dates = convert_date_range_to_months(date_range)

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

    # year_month 표현식 생성 (재사용)
    year_month_expr = get_year_month_expr(filtered_lf, ColumnNames.DATE_RECEIVED)

    # ==================== 사용 가능한 클러스터 목록 가져오기 ====================
    with st.spinner("클러스터 목록 로딩 중..."):
        available_clusters = get_available_clusters(
            _lf=filtered_lf,
            cluster_col=ColumnNames.CLUSTER,
            date_col=ColumnNames.DATE_RECEIVED,
            selected_dates=selected_dates if selected_dates else None,
            selected_manufacturers=None,
            selected_products=None,
            exclude_minus_one=True,
            _year_month_expr=year_month_expr
        )

    if not available_clusters:
        st.warning("선택한 기간에 해당하는 클러스터가 없습니다.")
        return

    # ==================== 필터 요약 배지 (공통 함수 사용) ====================
    render_filter_summary_badge(
        date_range=date_range,
        manufacturers=manufacturers,
        products=products,
        devices=devices,
        defect_types=defect_types,
        clusters=clusters
    )
    st.markdown("---")

    # ==================== 핵심 인사이트 (상단 배치) ====================
    render_cluster_insights(
        filtered_lf,
        available_clusters,
        selected_dates,
        year_month_expr,
        manufacturers,
        products
    )
    st.markdown("---")

    # ==================== 탭 구조 (3개) ====================
    tab1, tab2, tab3 = st.tabs([
        "📊 개별 분석",
        "⚖️ 클러스터 비교",
        "🔍 전체 개요"
    ])

    # ==================== 탭 1: 개별 클러스터 상세 분석 ====================
    with tab1:
        render_individual_cluster_analysis(
            filtered_lf,
            available_clusters,
            selected_dates,
            year_month_expr,
            manufacturers,
            products
        )

    # ==================== 탭 2: 클러스터 간 비교 ====================
    with tab2:
        render_cluster_comparison(
            filtered_lf,
            available_clusters,
            selected_dates,
            year_month_expr,
            manufacturers,
            products
        )

    # ==================== 탭 3: 전체 클러스터 개요 ====================
    with tab3:
        render_cluster_overview(
            filtered_lf,
            available_clusters,
            selected_dates,
            year_month_expr,
            manufacturers,
            products
        )


def render_individual_cluster_analysis(lf, available_clusters, selected_dates, year_month_expr, manufacturers, products):
    """개별 클러스터 상세 분석"""
    st.markdown("### 🔍 개별 클러스터 상세 분석")
    st.caption(f"특정 클러스터의 {Terms.KOREAN.PATIENT_HARM}, {Terms.KOREAN.PROBLEM_COMPONENT}, 시계열 추이를 분석합니다")

    # 설명 추가
    with st.expander("ℹ️ 개별 클러스터 분석이란?", expanded=False):
        st.markdown(f"""
        **개별 클러스터 분석**은 특정 클러스터(문제 유형 그룹)에 대한 상세 정보를 제공합니다.

        **구성 요소**:
        - **요약 메트릭**: 전체 케이스 수, {Terms.KOREAN.CFR}, 사망/부상 통계
        - **{Terms.KOREAN.PATIENT_HARM} 분포**: 사망, 중증/경증 부상, 부상 없음의 비율을 파이 차트로 표시
        - **상위 {Terms.KOREAN.PROBLEM_COMPONENT}**: 해당 클러스터에서 가장 빈번하게 보고된 {Terms.KOREAN.PROBLEM_COMPONENT} 순위
        - **시계열 추이**: 월별 케이스 수 변화를 통해 증가/감소 트렌드 파악

        **인사이트**:
        - {Terms.KOREAN.CFR}이 높은 클러스터는 우선적으로 안전 조치가 필요합니다
        - 특정 부품이 압도적으로 많이 보고된다면 해당 부품의 품질 개선이 시급합니다
        - 시계열에서 급증하는 구간은 특정 사건이나 리콜과 연관될 수 있습니다
        """)


    # 클러스터 선택 및 Top N 설정
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_cluster = st.selectbox(
            "클러스터 선택",
            options=available_clusters,
            index=0,
            format_func=lambda x: f"Cluster {x}",
            key="individual_cluster_selectbox"
        )

    with col2:
        top_n = st.number_input(
            "Top N 개수",
            min_value=5,
            max_value=50,
            value=Defaults.TOP_N,
            step=5,
            key="individual_top_n"
        )

    st.markdown("---")

    # 클러스터 분석 실행
    with st.spinner(f"Cluster {selected_cluster} 분석 중..."):
        cluster_data = cluster_check(
            _lf=lf,
            cluster_name=selected_cluster,
            cluster_col=ColumnNames.CLUSTER,
            component_col=ColumnNames.PROBLEM_COMPONENTS,
            event_col=ColumnNames.PATIENT_HARM,
            date_col=ColumnNames.DATE_RECEIVED,
            selected_dates=selected_dates,
            selected_manufacturers=None,
            selected_products=None,
            top_n=top_n,
            _year_month_expr=year_month_expr,
            manufacturers=tuple(manufacturers) if manufacturers else (),
            products=tuple(products) if products else ()
        )

    # ==================== 1. 전체 요약 메트릭 ====================
    st.subheader(f"📊 Cluster {selected_cluster} 요약")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("전체 케이스", f"{cluster_data['total_count']:,}")
    with col2:
        # 치명률 (사망 + 중증부상)
        death_count = cluster_data['harm_summary']['total_deaths']
        serious_count = cluster_data['harm_summary']['total_serious_injuries']
        severe_harm_count = death_count + serious_count
        cfr = (severe_harm_count / cluster_data['total_count'] * 100) if cluster_data['total_count'] > 0 else 0
        st.metric("치명률 (CFR)", f"{cfr:.2f}%",
                  delta=f"{severe_harm_count:,}건", delta_color="inverse")
    with col3:
        death_rate = (death_count / cluster_data['total_count'] * 100) if cluster_data['total_count'] > 0 else 0
        st.metric(Terms.KOREAN.DEATH_COUNT, f"{death_count:,}",
                  delta=f"{death_rate:.2f}%", delta_color="inverse")
    with col4:
        serious_rate = (serious_count / cluster_data['total_count'] * 100) if cluster_data['total_count'] > 0 else 0
        st.metric(Terms.KOREAN.SERIOUS_INJURY, f"{serious_count:,}",
                  delta=f"{serious_rate:.2f}%", delta_color="inverse")
    with col5:
        minor_count = cluster_data['harm_summary']['total_minor_injuries']
        minor_rate = (minor_count / cluster_data['total_count'] * 100) if cluster_data['total_count'] > 0 else 0
        st.metric(Terms.KOREAN.MINOR_INJURY, f"{minor_count:,}",
                  delta=f"{minor_rate:.2f}%", delta_color="inverse")

    st.markdown("---")

    # ==================== 2. 환자 피해 분포 + 상위 부품 (좌우 배치) ====================
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"#### 🎯 {Terms.KOREAN.PATIENT_HARM} 분포")

        harm_summary = cluster_data['harm_summary']

        # 공통 함수 사용
        fig_pie = create_harm_pie_chart(harm_summary, height=400, show_legend=True)
    
        # 라벨 위치 조정 (선택사항)
        fig_pie.update_traces(
            textposition='inside',  # 라벨을 파이 안쪽에 배치
            textinfo='percent+label'  # 퍼센트와 라벨 표시
        )

        if fig_pie:
            st.plotly_chart(fig_pie, width='stretch', config={'displayModeBar': False})
        else:
            st.info(f"{Terms.KOREAN.PATIENT_HARM} 데이터가 없습니다.")

    with col_right:
        st.markdown(f"#### 🔧 상위 {top_n}개 {Terms.KOREAN.PROBLEM_COMPONENT}")

        top_components = cluster_data['top_components']

        # 공통 함수 사용
        if len(top_components) > 0:
            fig_bar = create_horizontal_bar_chart(
                df=top_components,
                category_col=ColumnNames.PROBLEM_COMPONENTS,
                count_col='count',
                ratio_col='ratio',
                top_n=top_n,
                xaxis_title=Terms.KOREAN.REPORT_COUNT,
                yaxis_title=None,  # y축 제목 없음 (부품명이 이미 y축에 표시됨)
                colorscale='Blues'
            )

            if fig_bar:
                st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

            # 상세 데이터 - 컬럼명 한글로 변경
            with st.expander(f"📋 {Terms.KOREAN.DATA_TABLE}"):
                # 컬럼명을 한글로 변경
                top_components_display = top_components.rename({
                    ColumnNames.PROBLEM_COMPONENTS: Terms.KOREAN.PROBLEM_COMPONENT,
                    'count': Terms.KOREAN.REPORT_COUNT,
                    'ratio': f"{Terms.KOREAN.RATIO} (%)"
                })

                # 소수점 2자리 표시 포맷 적용
                st.dataframe(
                    top_components_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(
                            f"{Terms.KOREAN.RATIO} (%)",
                            format="%.2f"
                        )
                    }
                )
        else:
            st.info(f"해당 클러스터에는 {Terms.KOREAN.COMPONENT} 정보가 없습니다.")

    st.markdown("---")

    # ==================== 3. 결함 유형 및 결함 확정 분포 ====================
    col_confirmed, col_defect = st.columns([1, 1])

    with col_defect:
        st.markdown(f"#### 🔍 상위 {top_n}개 {Terms.KOREAN.DEFECT_TYPE}")

        defect_types = cluster_data['defect_types']

        if len(defect_types) > 0:
            # 공통 함수 사용
            fig_defect = create_horizontal_bar_chart(
                df=defect_types,
                category_col=ColumnNames.DEFECT_TYPE,
                count_col='count',
                ratio_col='ratio',
                top_n=top_n,
                xaxis_title=Terms.KOREAN.REPORT_COUNT,
                yaxis_title=None,  # y축 제목 없음
                colorscale='Oranges'
            )

            if fig_defect:
                st.plotly_chart(fig_defect, width='stretch', config={'displayModeBar': False})

            with st.expander(f"📋 {Terms.KOREAN.DATA_TABLE}"):
                # 컬럼명 한글로 변경
                defect_types_display = defect_types.rename({
                    ColumnNames.DEFECT_TYPE: Terms.KOREAN.DEFECT_TYPE,
                    'count': Terms.KOREAN.REPORT_COUNT,
                    'ratio': f"{Terms.KOREAN.RATIO} (%)"
                })
                # 결함 유형 컬럼을 문자열로 변환 (Arrow 직렬화 에러 방지)
                defect_types_display = defect_types_display.with_columns(
                    pl.col(Terms.KOREAN.DEFECT_TYPE).cast(pl.Utf8)
                )
                st.dataframe(
                    defect_types_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(f"{Terms.KOREAN.RATIO} (%)", format="%.2f")
                    }
                )
        else:
            st.info(f"{Terms.KOREAN.DEFECT_TYPE} 데이터가 없습니다.")

    with col_confirmed:
        st.markdown(f"#### ✅ {Terms.KOREAN.DEFECT_CONFIRMED} 분포")

        defect_confirmed = cluster_data['defect_confirmed']

        if len(defect_confirmed) > 0:
            # 전용 파이 차트 함수 사용
            fig_confirmed = create_defect_confirmed_pie_chart(
                defect_confirmed_df=defect_confirmed,
                defect_col=ColumnNames.DEFECT_CONFIRMED,
                count_col='count',
                height=400,
                show_legend=True
            )

            if fig_confirmed:
                st.plotly_chart(fig_confirmed, width='stretch', config={'displayModeBar': False})

            with st.expander("📋 상세 데이터"):
                # 컬럼명 한글로 변경
                defect_confirmed_display = defect_confirmed.rename({
                    ColumnNames.DEFECT_CONFIRMED: Terms.KOREAN.DEFECT_CONFIRMED,
                    'count': Terms.KOREAN.REPORT_COUNT,
                    'ratio': f"{Terms.KOREAN.RATIO} (%)"
                })
                st.dataframe(
                    defect_confirmed_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(f"{Terms.KOREAN.RATIO} (%)", format="%.2f")
                    }
                )
        else:
            st.info(f"{Terms.KOREAN.DEFECT_CONFIRMED} 데이터가 없습니다.")

    st.markdown("---")

    # ==================== 4. 시계열 분석 ====================
    st.markdown("#### 📈 월별 발생 추이")

    time_series = cluster_data['time_series']

    if len(time_series) > 0:
        fig_line = px.line(
            time_series,
            x='year_month',
            y='count',
            markers=True,
            labels={'year_month': '년-월', 'count': '발생 건수'}
        )

        fig_line.update_traces(
            line_color=ChartStyles.PRIMARY_COLOR,
            line_width=3,
            marker=dict(size=8)
        )

        fig_line.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=80),
            hovermode='x unified',
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_line, width='stretch', config={'displayModeBar': False})

        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("평균 월별 발생", f"{time_series['count'].mean():.2f}")
        with col2:
            st.metric("최대 월별 발생", f"{time_series['count'].max()}")
        with col3:
            st.metric("최소 월별 발생", f"{time_series['count'].min()}")
        with col4:
            std_dev = time_series['count'].std()
            st.metric("표준편차", f"{std_dev:.2f}" if std_dev is not None else "N/A")
    else:
        st.info("시계열 데이터가 없습니다.")


def render_cluster_comparison(lf, available_clusters, selected_dates, year_month_expr, manufacturers, products):
    """클러스터 간 비교 분석"""
    st.markdown("### ⚖️ 클러스터 간 비교")
    st.caption("두 클러스터의 특성을 나란히 비교합니다")

    # 설명 추가
    with st.expander("ℹ️ 클러스터 비교란?", expanded=False):
        st.markdown(f"""
        **클러스터 비교**는 두 개의 클러스터(문제 유형 그룹)를 직접 대조하여 차이점을 분석합니다.

        **비교 항목**:
        - **핵심 메트릭**: 전체 케이스 수, {Terms.KOREAN.CFR}, 사망/부상 건수 비교
        - **{Terms.KOREAN.PATIENT_HARM} 분포**: 두 클러스터의 피해 심각도 패턴 차이
        - **상위 {Terms.KOREAN.PROBLEM_COMPONENT}**: 각 클러스터에서 주로 보고되는 부품의 차이
        - **시계열 추이**: 시간에 따른 보고 건수 변화 패턴 비교

        **인사이트**:
        - 케이스 수는 많지만 {Terms.KOREAN.CFR}이 낮은 클러스터 vs. 케이스는 적지만 {Terms.KOREAN.CFR}이 높은 클러스터를 구분할 수 있습니다
        - {Terms.KOREAN.PROBLEM_COMPONENT}이 겹치는 클러스터는 공통 원인이 있을 가능성이 있습니다
        - 시계열 추이가 유사하다면 동일한 외부 요인(예: 리콜, 규제 변화)의 영향을 받을 수 있습니다
        """)

    if len(available_clusters) < 2:
        st.warning("비교를 위해서는 최소 2개 이상의 클러스터가 필요합니다.")
        return

    # 클러스터 선택
    col1, col2 = st.columns(2)

    with col1:
        cluster_a = st.selectbox(
            "클러스터 A",
            options=available_clusters,
            index=0,
            format_func=lambda x: f"Cluster {x}",
            key="compare_cluster_a"
        )

    with col2:
        cluster_b = st.selectbox(
            "클러스터 B",
            options=available_clusters,
            index=min(1, len(available_clusters) - 1),
            format_func=lambda x: f"Cluster {x}",
            key="compare_cluster_b"
        )

    if cluster_a == cluster_b:
        st.warning("⚠️ 서로 다른 클러스터를 선택해주세요")
        return

    top_n = st.slider("Top N 개수", 5, 20, 10, key="compare_top_n")

    st.markdown("---")

    # 두 클러스터 데이터 로드
    with st.spinner("클러스터 비교 데이터 로딩 중..."):
        data_a = cluster_check(
            _lf=lf, cluster_name=cluster_a, cluster_col=ColumnNames.CLUSTER,
            component_col=ColumnNames.PROBLEM_COMPONENTS, event_col=ColumnNames.PATIENT_HARM,
            date_col=ColumnNames.DATE_RECEIVED, selected_dates=selected_dates,
            selected_manufacturers=None, selected_products=None,
            top_n=top_n, _year_month_expr=year_month_expr,
            manufacturers=tuple(manufacturers) if manufacturers else (),
            products=tuple(products) if products else ()
        )

        data_b = cluster_check(
            _lf=lf, cluster_name=cluster_b, cluster_col=ColumnNames.CLUSTER,
            component_col=ColumnNames.PROBLEM_COMPONENTS, event_col=ColumnNames.PATIENT_HARM,
            date_col=ColumnNames.DATE_RECEIVED, selected_dates=selected_dates,
            selected_manufacturers=None, selected_products=None,
            top_n=top_n, _year_month_expr=year_month_expr,
            manufacturers=tuple(manufacturers) if manufacturers else (),
            products=tuple(products) if products else ()
        )

    # ==================== 1. 요약 비교 ====================
    st.markdown("#### 📊 요약 비교")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Cluster {cluster_a}**")
        st.metric(Terms.KOREAN.TOTAL_CASES, f"{data_a['total_count']:,}")
        st.metric(Terms.KOREAN.DEATH_COUNT, f"{data_a['harm_summary']['total_deaths']:,}")
        st.metric(Terms.KOREAN.SERIOUS_INJURY, f"{data_a['harm_summary']['total_serious_injuries']:,}")

    with col2:
        st.markdown(f"**Cluster {cluster_b}**")
        st.metric(Terms.KOREAN.TOTAL_CASES, f"{data_b['total_count']:,}")
        st.metric(Terms.KOREAN.DEATH_COUNT, f"{data_b['harm_summary']['total_deaths']:,}")
        st.metric(Terms.KOREAN.SERIOUS_INJURY, f"{data_b['harm_summary']['total_serious_injuries']:,}")

    st.markdown("---")

    # ==================== 2. 환자 피해 비교 (나란히) ====================
    st.markdown(f"#### 🎯 {Terms.KOREAN.PATIENT_HARM} 분포 비교")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Cluster {cluster_a}", f"Cluster {cluster_b}"),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )

    # Cluster A 파이 차트
    harm_a = data_a['harm_summary']
    labels_a = [Terms.KOREAN.DEATH_COUNT, Terms.KOREAN.SERIOUS_INJURY, Terms.KOREAN.MINOR_INJURY, Terms.KOREAN.NO_HARM]
    values_a = [
        harm_a['total_deaths'],
        harm_a['total_serious_injuries'],
        harm_a['total_minor_injuries'],
        harm_a['total_no_injuries']
    ]

    fig.add_trace(go.Pie(
        labels=labels_a,
        values=values_a,
        name=f"Cluster {cluster_a}",
        marker=dict(colors=[ChartStyles.DANGER_COLOR, ChartStyles.WARNING_COLOR, '#ffd700', ChartStyles.SUCCESS_COLOR])
    ), row=1, col=1)

    # Cluster B 파이 차트
    harm_b = data_b['harm_summary']
    values_b = [
        harm_b['total_deaths'],
        harm_b['total_serious_injuries'],
        harm_b['total_minor_injuries'],
        harm_b['total_no_injuries']
    ]

    fig.add_trace(go.Pie(
        labels=labels_a,
        values=values_b,
        name=f"Cluster {cluster_b}",
        marker=dict(colors=[ChartStyles.DANGER_COLOR, ChartStyles.WARNING_COLOR, '#ffd700', ChartStyles.SUCCESS_COLOR])
    ), row=1, col=2)

    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # ==================== 3. 상위 부품 비교 ====================
    st.markdown("#### 🔧 상위 부품 비교")

    components_a = data_a['top_components'].to_pandas()
    components_b = data_b['top_components'].to_pandas()

    if len(components_a) > 0 and len(components_b) > 0:
        # 공통 부품 찾기
        common_components = set(components_a[ColumnNames.PROBLEM_COMPONENTS]) & set(components_b[ColumnNames.PROBLEM_COMPONENTS])

        if common_components:
            st.info(f"🔍 **공통 부품**: {len(common_components)}개 발견 - {', '.join(list(common_components)[:5])}" +
                   (f" 외 {len(common_components) - 5}개" if len(common_components) > 5 else ""))

        # 나란히 비교
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Cluster {cluster_a} 상위 부품**")
            comp_a_display = components_a.head(10).rename(columns={
                ColumnNames.PROBLEM_COMPONENTS: Terms.KOREAN.PROBLEM_COMPONENT,
                'count': Terms.KOREAN.REPORT_COUNT,
                'ratio': f"{Terms.KOREAN.RATIO} (%)"
            })
            if f"{Terms.KOREAN.RATIO} (%)" in comp_a_display.columns:
                st.dataframe(
                    comp_a_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(
                            f"{Terms.KOREAN.RATIO} (%)",
                            format="%.2f"
                        )
                    }
                )
            else:
                st.dataframe(comp_a_display, width='stretch', hide_index=True)

        with col2:
            st.markdown(f"**Cluster {cluster_b} 상위 부품**")
            comp_b_display = components_b.head(10).rename(columns={
                ColumnNames.PROBLEM_COMPONENTS: Terms.KOREAN.PROBLEM_COMPONENT,
                'count': Terms.KOREAN.REPORT_COUNT,
                'ratio': f"{Terms.KOREAN.RATIO} (%)"
            })
            if f"{Terms.KOREAN.RATIO} (%)" in comp_b_display.columns:
                st.dataframe(
                    comp_b_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(
                            f"{Terms.KOREAN.RATIO} (%)",
                            format="%.2f"
                        )
                    }
                )
            else:
                st.dataframe(comp_b_display, width='stretch', hide_index=True)
    else:
        st.info("부품 데이터가 부족합니다.")

    st.markdown("---")

    # ==================== 4. 결함 유형 비교 ====================
    st.markdown(f"#### 🔍 {Terms.KOREAN.DEFECT_TYPE} 비교")

    defect_a = data_a['defect_types'].to_pandas()
    defect_b = data_b['defect_types'].to_pandas()

    if len(defect_a) > 0 and len(defect_b) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Cluster {cluster_a} 상위 {Terms.KOREAN.DEFECT_TYPE}**")
            defect_a_display = defect_a.head(10).rename(columns={
                'defect_type': Terms.KOREAN.DEFECT_TYPE,
                'count': Terms.KOREAN.REPORT_COUNT,
                'ratio': f"{Terms.KOREAN.RATIO} (%)"
            })
            # 결함 유형 컬럼을 문자열로 변환 (Arrow 직렬화 에러 방지)
            if Terms.KOREAN.DEFECT_TYPE in defect_a_display.columns:
                defect_a_display[Terms.KOREAN.DEFECT_TYPE] = defect_a_display[Terms.KOREAN.DEFECT_TYPE].astype(str)
            if f"{Terms.KOREAN.RATIO} (%)" in defect_a_display.columns:
                st.dataframe(
                    defect_a_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(f"{Terms.KOREAN.RATIO} (%)", format="%.2f")
                    }
                )
            else:
                st.dataframe(defect_a_display, width='stretch', hide_index=True)

        with col2:
            st.markdown(f"**Cluster {cluster_b} 상위 {Terms.KOREAN.DEFECT_TYPE}**")
            defect_b_display = defect_b.head(10).rename(columns={
                'defect_type': Terms.KOREAN.DEFECT_TYPE,
                'count': Terms.KOREAN.REPORT_COUNT,
                'ratio': f"{Terms.KOREAN.RATIO} (%)"
            })
            # 결함 유형 컬럼을 문자열로 변환 (Arrow 직렬화 에러 방지)
            if Terms.KOREAN.DEFECT_TYPE in defect_b_display.columns:
                defect_b_display[Terms.KOREAN.DEFECT_TYPE] = defect_b_display[Terms.KOREAN.DEFECT_TYPE].astype(str)
            if f"{Terms.KOREAN.RATIO} (%)" in defect_b_display.columns:
                st.dataframe(
                    defect_b_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(f"{Terms.KOREAN.RATIO} (%)", format="%.2f")
                    }
                )
            else:
                st.dataframe(defect_b_display, width='stretch', hide_index=True)
    else:
        st.info(f"{Terms.KOREAN.DEFECT_TYPE} 데이터가 부족합니다.")

    st.markdown("---")

    # ==================== 5. 결함 확정 비교 ====================
    st.markdown(f"#### ✅ {Terms.KOREAN.DEFECT_CONFIRMED} 비교")

    confirmed_a = data_a['defect_confirmed'].to_pandas()
    confirmed_b = data_b['defect_confirmed'].to_pandas()

    if len(confirmed_a) > 0 and len(confirmed_b) > 0:
        fig_confirmed = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Cluster {cluster_a}", f"Cluster {cluster_b}"),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )

        # Cluster A
        fig_confirmed.add_trace(go.Pie(
            labels=confirmed_a[ColumnNames.DEFECT_CONFIRMED],
            values=confirmed_a['count'],
            name=f"Cluster {cluster_a}",
            marker=dict(colors=['#d62728', '#2ca02c', '#CCCCCC'])
        ), row=1, col=1)

        # Cluster B
        fig_confirmed.add_trace(go.Pie(
            labels=confirmed_b[ColumnNames.DEFECT_CONFIRMED],
            values=confirmed_b['count'],
            name=f"Cluster {cluster_b}",
            marker=dict(colors=['#d62728', '#2ca02c', '#CCCCCC'])
        ), row=1, col=2)

        fig_confirmed.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_confirmed, width='stretch', config={'displayModeBar': False})

        # 비율 비교 테이블
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Cluster {cluster_a} {Terms.KOREAN.RATIO}**")
            confirmed_a_display = confirmed_a.rename(columns={
                'defect_confirmed': Terms.KOREAN.DEFECT_CONFIRMED,
                'count': Terms.KOREAN.REPORT_COUNT,
                'ratio': f"{Terms.KOREAN.RATIO} (%)"
            })
            st.dataframe(
                confirmed_a_display,
                width='stretch',
                hide_index=True,
                column_config={
                    f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(f"{Terms.KOREAN.RATIO} (%)", format="%.2f")
                }
            )

        with col2:
            st.markdown(f"**Cluster {cluster_b} {Terms.KOREAN.RATIO}**")
            confirmed_b_display = confirmed_b.rename(columns={
                'defect_confirmed': Terms.KOREAN.DEFECT_CONFIRMED,
                'count': Terms.KOREAN.REPORT_COUNT,
                'ratio': f"{Terms.KOREAN.RATIO} (%)"
            })
            st.dataframe(
                confirmed_b_display,
                width='stretch',
                hide_index=True,
                column_config={
                    f"{Terms.KOREAN.RATIO} (%)": st.column_config.NumberColumn(f"{Terms.KOREAN.RATIO} (%)", format="%.2f")
                }
            )
    else:
        st.info(f"{Terms.KOREAN.DEFECT_CONFIRMED} 데이터가 부족합니다.")


def render_cluster_overview(lf, available_clusters, selected_dates, year_month_expr, manufacturers, products):
    """전체 클러스터 개요"""
    st.markdown("### 🌐 전체 클러스터 개요")
    st.caption("모든 클러스터의 전체적인 분포와 특성을 한눈에 확인합니다")

    # 설명 추가
    with st.expander("ℹ️ 전체 클러스터 개요란?", expanded=False):
        st.markdown("""
        **전체 클러스터 개요**는 모든 클러스터를 한눈에 비교하고 전체 패턴을 파악합니다.

        **시각화 구성**:
        - **클러스터별 케이스 분포**: 각 클러스터의 보고 건수를 막대 차트로 비교
        - **클러스터별 치명률 비교**: CFR(사망+중증부상 비율)을 막대 차트로 표시
        - **케이스 수 vs 치명률 산점도**: 보고 건수와 치명률의 관계를 버블 차트로 시각화 (버블 크기 = 사망 건수)
        - **전체 통계 테이블**: 모든 클러스터의 주요 지표를 한 테이블에 정리

        **인사이트**:
        - 케이스 수가 많은 클러스터는 빈도가 높은 문제이므로 전반적인 품질 개선이 필요합니다
        - 치명률이 높은 클러스터는 심각도가 높은 문제이므로 즉각적인 안전 조치가 필요합니다
        - 산점도에서 오른쪽 위(고빈도+고위험)에 위치한 클러스터가 최우선 대응 대상입니다
        - 왼쪽 위(저빈도+고위험)에 위치한 클러스터는 발생 시 치명적이므로 예방 조치가 중요합니다
        """)


    # 모든 클러스터 데이터 수집
    with st.spinner("전체 클러스터 데이터 로딩 중..."):
        all_cluster_data = []

        for cluster_id in available_clusters:
            data = cluster_check(
                _lf=lf, cluster_name=cluster_id, cluster_col=ColumnNames.CLUSTER,
                component_col=ColumnNames.PROBLEM_COMPONENTS, event_col=ColumnNames.PATIENT_HARM,
                date_col=ColumnNames.DATE_RECEIVED, selected_dates=selected_dates,
                selected_manufacturers=None, selected_products=None,
                top_n=5, _year_month_expr=year_month_expr,
                manufacturers=tuple(manufacturers) if manufacturers else (),
                products=tuple(products) if products else ()
            )

            # Defect Confirmed 통계
            defect_confirmed = data['defect_confirmed']
            confirmed_yes = defect_confirmed.filter(pl.col(ColumnNames.DEFECT_CONFIRMED) == '결함 있음')['count'].sum() if len(defect_confirmed) > 0 else 0
            confirmed_no = defect_confirmed.filter(pl.col(ColumnNames.DEFECT_CONFIRMED) == '결함 없음')['count'].sum() if len(defect_confirmed) > 0 else 0
            confirmed_unknown = defect_confirmed.filter(pl.col(ColumnNames.DEFECT_CONFIRMED) == '알 수 없음')['count'].sum() if len(defect_confirmed) > 0 else 0

            # Defect Type 통계 - 상위 5개 결함 유형 추출
            defect_types = data['defect_types']
            defect_type_dict = {}
            for row in defect_types.iter_rows(named=True):
                defect_type_dict[row[ColumnNames.DEFECT_TYPE]] = row['count']

            all_cluster_data.append({
                'cluster': cluster_id,
                'total_count': data['total_count'],
                'deaths': data['harm_summary']['total_deaths'],
                'serious_injuries': data['harm_summary']['total_serious_injuries'],
                'minor_injuries': data['harm_summary']['total_minor_injuries'],
                'no_harm': data['harm_summary']['total_no_injuries'],
                'defect_confirmed_yes': confirmed_yes,
                'defect_confirmed_no': confirmed_no,
                'defect_confirmed_unknown': confirmed_unknown,
                'defect_type_dict': defect_type_dict
            })

    overview_df = pd.DataFrame(all_cluster_data)
    # 치명률 = (사망 + 중증부상) / 총 건수 × 100
    overview_df['cfr'] = ((overview_df['deaths'] + overview_df['serious_injuries']) / overview_df['total_count'] * 100).round(2)
    overview_df['cluster_label'] = overview_df['cluster'].apply(lambda x: f"Cluster {x}")

    # ==================== 1. 클러스터별 케이스 수 비교 ====================
    st.markdown("#### 📊 클러스터별 케이스 분포")

    fig_bar = px.bar(
        overview_df,
        x='cluster_label',
        y='total_count',
        text='total_count',
        labels={'cluster_label': '클러스터', 'total_count': '케이스 수'},
        color='total_count',
        color_continuous_scale='Blues'
    )

    fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
    max_count = overview_df['total_count'].max()
    fig_bar.update_layout(
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, max_count * 1.15])  # 텍스트 표시 공간 확보
    )

    st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # ==================== 2. 클러스터별 환자 피해 분포 (적층 바) ====================
    st.markdown(f"#### 🎯 클러스터별 {Terms.KOREAN.PATIENT_HARM} 분포")

    fig_stacked = go.Figure()

    fig_stacked.add_trace(go.Bar(
        name=Terms.KOREAN.DEATH_COUNT,
        x=overview_df['cluster_label'],
        y=overview_df['deaths'],
        marker_color=ChartStyles.DANGER_COLOR
    ))

    fig_stacked.add_trace(go.Bar(
        name=Terms.KOREAN.SERIOUS_INJURY,
        x=overview_df['cluster_label'],
        y=overview_df['serious_injuries'],
        marker_color=ChartStyles.WARNING_COLOR
    ))

    fig_stacked.add_trace(go.Bar(
        name=Terms.KOREAN.MINOR_INJURY,
        x=overview_df['cluster_label'],
        y=overview_df['minor_injuries'],
        marker_color='#ffd700'
    ))

    fig_stacked.add_trace(go.Bar(
        name=Terms.KOREAN.NO_HARM,
        x=overview_df['cluster_label'],
        y=overview_df['no_harm'],
        marker_color=ChartStyles.SUCCESS_COLOR
    ))

    # 적층 바의 최대값 계산 (각 클러스터의 전체 합)
    max_stacked = overview_df['total_count'].max()

    fig_stacked.update_layout(
        barmode='stack',
        xaxis_title="클러스터",
        yaxis_title="케이스 수",
        yaxis=dict(range=[0, max_stacked * 1.1]),  # 10% 여유 공간
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_stacked, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # ==================== 3. 클러스터별 치명률 ====================
    st.markdown("#### 💀 클러스터별 치명률")

    fig_cfr = px.scatter(
        overview_df,
        x='cluster_label',
        y='cfr',
        size='total_count',
        color='cfr',
        color_continuous_scale='Reds',
        labels={'cluster_label': '클러스터', 'cfr': '치명률 (%)'},
        hover_data={'total_count': ':,', 'deaths': True, 'serious_injuries': True}
    )

    fig_cfr.update_layout(height=400)
    st.plotly_chart(fig_cfr, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # ==================== 4. 클러스터별 결함 유형 분포 ====================
    st.markdown(f"#### 🔧 클러스터별 {Terms.KOREAN.DEFECT_TYPE} 분포")

    # 모든 결함 유형 수집
    all_defect_types = set()
    for cluster_data in all_cluster_data:
        all_defect_types.update(cluster_data['defect_type_dict'].keys())

    # 발생 빈도 순으로 정렬 (전체 데이터에서 가장 많이 나타나는 것들)
    defect_type_totals = {}
    for defect_type in all_defect_types:
        total = sum(cluster_data['defect_type_dict'].get(defect_type, 0) for cluster_data in all_cluster_data)
        defect_type_totals[defect_type] = total

    sorted_defect_types = sorted(defect_type_totals.items(), key=lambda x: x[1], reverse=True)
    all_defect_type_names = [dt[0] for dt in sorted_defect_types]

    # 결함 유형 다중 선택 필터
    st.markdown("**🔍 표시할 결함 유형 선택**")
    default_selection = all_defect_type_names[:min(5, len(all_defect_type_names))]
    selected_defect_types = st.multiselect(
        label="결함 유형",
        options=all_defect_type_names,
        default=default_selection,
        key="defect_type_filter",
        label_visibility="collapsed",
        help="클러스터별로 비교할 결함 유형을 선택하세요"
    )

    if len(selected_defect_types) > 0:
        # 그룹형 막대 차트 (Grouped Bar) - 비율 기준
        fig_defect_type = go.Figure()

        # 색상 팔레트 정의
        defect_type_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#ABEBC6']

        for idx, defect_type in enumerate(selected_defect_types):
            # 각 클러스터의 전체 케이스 수 대비 비율 계산
            ratios = [
                (cluster_data['defect_type_dict'].get(defect_type, 0) / cluster_data['total_count'] * 100) if cluster_data['total_count'] > 0 else 0
                for cluster_data in all_cluster_data
            ]
            fig_defect_type.add_trace(go.Bar(
                name=defect_type,
                x=overview_df['cluster_label'],
                y=ratios,
                marker_color=defect_type_colors[idx % len(defect_type_colors)],
                text=ratios,
                textposition='outside',
                texttemplate='%{text:.1f}%'
            ))

        # 최대 비율 계산 (모든 선택된 결함 유형에서)
        all_ratios = []
        for defect_type in selected_defect_types:
            ratios = [
                (cluster_data['defect_type_dict'].get(defect_type, 0) / cluster_data['total_count'] * 100) if cluster_data['total_count'] > 0 else 0
                for cluster_data in all_cluster_data
            ]
            all_ratios.extend(ratios)
        max_ratio = max(all_ratios) if all_ratios else 100

        fig_defect_type.update_layout(
            barmode='group',  # Grouped bar
            xaxis_title="클러스터",
            yaxis_title="비율 (%)",
            yaxis=dict(range=[0, max_ratio * 1.2]),  # 텍스트 표시 공간 확보 (20% 여유)
            height=450,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)  # 범례 공간 확보
        )

        st.plotly_chart(fig_defect_type, width='stretch', config={'displayModeBar': False})
    else:
        st.info("비교할 결함 유형을 선택해주세요.")

    st.markdown("---")

    # ==================== 5. 클러스터별 결함 확정률 ====================
    st.markdown(f"#### ✅ 클러스터별 {Terms.KOREAN.DEFECT_CONFIRMED}률")

    # 결함 확정률 계산
    overview_df['defect_confirmed_rate'] = (
        (overview_df['defect_confirmed_yes'] / overview_df['total_count'] * 100).round(2)
    )

    fig_confirmed_rate = px.bar(
        overview_df,
        x='cluster_label',
        y='defect_confirmed_rate',
        text='defect_confirmed_rate',
        labels={'cluster_label': '클러스터', 'defect_confirmed_rate': '확정률 (%)'},
        color='defect_confirmed_rate',
        color_continuous_scale='Reds'
    )

    fig_confirmed_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    max_confirmed_rate = overview_df['defect_confirmed_rate'].max()
    fig_confirmed_rate.update_layout(
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, max_confirmed_rate * 1.15])  # 텍스트 표시 공간 확보
    )

    st.plotly_chart(fig_confirmed_rate, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # 요약 테이블
    with st.expander("📋 전체 클러스터 요약 테이블"):
        display_df = overview_df[[
            'cluster_label', 'total_count', 'deaths',
            'serious_injuries', 'minor_injuries', 'no_harm', 'cfr',
            'defect_confirmed_yes', 'defect_confirmed_rate'
        ]].rename(columns={
            'cluster_label': '클러스터',
            'total_count': '전체 케이스',
            'deaths': '사망',
            'serious_injuries': '중증 부상',
            'minor_injuries': '경증 부상',
            'no_harm': '부상 없음',
            'cfr': '치명률 (%)',
            'defect_confirmed_yes': '결함 확정',
            'defect_confirmed_rate': '확정률 (%)'
        })

        # 소수점 2자리 표시 포맷 적용
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "치명률 (%)": st.column_config.NumberColumn("치명률 (%)", format="%.2f"),
                "확정률 (%)": st.column_config.NumberColumn("확정률 (%)", format="%.2f")
            }
        )


def render_cluster_insights(lf, available_clusters, selected_dates, year_month_expr, manufacturers, products):
    """자동 인사이트 생성 (terminology 기반)"""
    from dashboard.utils.terminology import get_term_manager

    term = get_term_manager()
    st.subheader("💡 핵심 인사이트")

    insights = []

    with st.spinner(term.messages.get('analyzing', '분석 중...')):
        # 모든 클러스터 데이터 수집
        all_data = []
        for cluster_id in available_clusters:
            data = cluster_check(
                _lf=lf, cluster_name=cluster_id, cluster_col=ColumnNames.CLUSTER,
                component_col=ColumnNames.PROBLEM_COMPONENTS, event_col=ColumnNames.PATIENT_HARM,
                date_col=ColumnNames.DATE_RECEIVED, selected_dates=selected_dates,
                selected_manufacturers=None, selected_products=None,
                top_n=10, _year_month_expr=year_month_expr,
                manufacturers=tuple(manufacturers) if manufacturers else (),
                products=tuple(products) if products else ()
            )
            all_data.append((cluster_id, data))

        # 1. 가장 큰 클러스터
        largest_cluster = max(all_data, key=lambda x: x[1]['total_count'])
        insights.append({
            "type": "info",
            "text": term.format_message('cluster_most_cases',
                                       cluster_id=largest_cluster[0],
                                       count=largest_cluster[1]['total_count'])
        })

        # 2. 가장 위험한 클러스터 (치명률 기준: 사망 + 중증부상)
        cfr_rates = [(c_id,
                      (data['harm_summary']['total_deaths'] + data['harm_summary']['total_serious_injuries']) / data['total_count'] * 100 if data['total_count'] > 0 else 0,
                      data['harm_summary']['total_deaths'] + data['harm_summary']['total_serious_injuries'])
                     for c_id, data in all_data]
        highest_cfr = max(cfr_rates, key=lambda x: x[1])

        if highest_cfr[1] > 0:
            insights.append({
                "type": "error",
                "text": term.format_message('cluster_highest_cfr',
                                           cluster_id=highest_cfr[0],
                                           cfr=highest_cfr[1],
                                           severe_count=highest_cfr[2])
            })

        # 3. 가장 안전한 클러스터
        lowest_cfr = min(cfr_rates, key=lambda x: x[1])
        insights.append({
            "type": "success",
            "text": term.format_message('cluster_lowest_cfr',
                                       cluster_id=lowest_cfr[0],
                                       cfr=lowest_cfr[1])
        })

        # 4. 공통 문제 부품
        all_components = []
        for c_id, data in all_data:
            if len(data['top_components']) > 0:
                top_3 = data['top_components'].head(3)[ColumnNames.PROBLEM_COMPONENTS].to_list()
                all_components.extend(top_3)

        if all_components:
            from collections import Counter
            most_common = Counter(all_components).most_common(3)
            common_parts = ", ".join([f"{part} ({count}개 클러스터)" for part, count in most_common if count > 1])

            if common_parts:
                insights.append({
                    "type": "warning",
                    "text": term.format_message('cluster_common_components', parts=common_parts)
                })

    # 인사이트 표시
    if insights:
        for insight in insights:
            if insight["type"] == "warning":
                st.warning(insight["text"])
            elif insight["type"] == "error":
                st.error(insight["text"])
            elif insight["type"] == "success":
                st.success(insight["text"])
            else:
                st.info(insight["text"])
    else:
        st.info("특이사항이 감지되지 않았습니다")

    st.markdown("---")

    # 권장 사항 (terminology 기반)
    st.markdown("### 🎯 권장 사항")

    recommendations = []

    # 치명률 높은 클러스터에 대한 권장
    if highest_cfr[1] > 5.0:
        recommendations.append(
            term.format_message('cluster_recommendation_high_cfr', cluster_id=highest_cfr[0])
        )

    # 케이스 수 많은 클러스터
    if largest_cluster[1]['total_count'] > 100:
        recommendations.append(
            term.format_message('cluster_recommendation_large', cluster_id=largest_cluster[0])
        )

    # 공통 부품
    if all_components:
        recommendations.append(term.messages.get('cluster_recommendation_common_parts'))

    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.markdown(term.messages.get('cluster_recommendation_none'))

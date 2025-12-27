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
    create_component_bar_chart
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
    st.caption("특정 클러스터의 환자 피해, 문제 부품, 시계열 추이를 분석합니다")

    # 설명 추가
    with st.expander("ℹ️ 개별 클러스터 분석이란?", expanded=False):
        st.markdown("""
        **개별 클러스터 분석**은 특정 클러스터(문제 유형 그룹)에 대한 상세 정보를 제공합니다.

        **구성 요소**:
        - **요약 메트릭**: 전체 케이스 수, 치명률(CFR), 사망/부상 통계
        - **환자 피해 분포**: 사망, 중증/경증 부상, 부상 없음의 비율을 파이 차트로 표시
        - **상위 문제 부품**: 해당 클러스터에서 가장 빈번하게 보고된 문제 부품 순위
        - **시계열 추이**: 월별 케이스 수 변화를 통해 증가/감소 트렌드 파악

        **인사이트**:
        - 치명률이 높은 클러스터는 우선적으로 안전 조치가 필요합니다
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
            "상위 부품 개수",
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
        st.markdown("#### 🎯 환자 피해 분포")

        harm_summary = cluster_data['harm_summary']

        # 공통 함수 사용
        fig_pie = create_harm_pie_chart(harm_summary, height=400, show_legend=True)

        if fig_pie:
            st.plotly_chart(fig_pie, width='stretch', config={'displayModeBar': False})
        else:
            st.info("환자 피해 데이터가 없습니다.")

    with col_right:
        st.markdown(f"#### 🔧 상위 {top_n}개 문제 부품")

        top_components = cluster_data['top_components']

        # 공통 함수 사용
        if len(top_components) > 0:
            fig_bar = create_component_bar_chart(
                component_df=top_components,
                component_col=ColumnNames.PROBLEM_COMPONENTS,
                count_col='count',
                ratio_col='ratio',
                top_n=top_n
            )

            if fig_bar:
                st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

            # 상세 데이터
            with st.expander("📋 상세 데이터"):
                # 소수점 2자리 표시 포맷 적용
                if 'ratio' in top_components.columns:
                    st.dataframe(
                        top_components,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "ratio": st.column_config.NumberColumn(
                                "ratio",
                                format="%.2f"
                            )
                        }
                    )
                else:
                    st.dataframe(top_components, width='stretch', hide_index=True)
        else:
            st.info("해당 클러스터에는 부품 정보가 없습니다.")

    st.markdown("---")

    # ==================== 3. 시계열 분석 ====================
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
        st.markdown("""
        **클러스터 비교**는 두 개의 클러스터(문제 유형 그룹)를 직접 대조하여 차이점을 분석합니다.

        **비교 항목**:
        - **핵심 메트릭**: 전체 케이스 수, 치명률, 사망/부상 건수 비교
        - **환자 피해 분포**: 두 클러스터의 피해 심각도 패턴 차이
        - **상위 문제 부품**: 각 클러스터에서 주로 보고되는 부품의 차이
        - **시계열 추이**: 시간에 따른 보고 건수 변화 패턴 비교

        **인사이트**:
        - 케이스 수는 많지만 치명률이 낮은 클러스터 vs. 케이스는 적지만 치명률이 높은 클러스터를 구분할 수 있습니다
        - 문제 부품이 겹치는 클러스터는 공통 원인이 있을 가능성이 있습니다
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

    top_n = st.slider("상위 부품 개수", 5, 20, 10, key="compare_top_n")

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
    st.markdown("#### 🎯 환자 피해 분포 비교")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Cluster {cluster_a}", f"Cluster {cluster_b}"),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )

    # Cluster A 파이 차트
    harm_a = data_a['harm_summary']
    labels_a = ['Death', 'Serious Injury', 'Minor Injury', 'No Harm']
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
            # 소수점 2자리 표시 포맷 적용
            comp_a_display = components_a.head(10)
            if 'ratio' in comp_a_display.columns:
                st.dataframe(
                    comp_a_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "ratio": st.column_config.NumberColumn(
                            "ratio",
                            format="%.2f"
                        )
                    }
                )
            else:
                st.dataframe(comp_a_display, width='stretch', hide_index=True)

        with col2:
            st.markdown(f"**Cluster {cluster_b} 상위 부품**")
            # 소수점 2자리 표시 포맷 적용
            comp_b_display = components_b.head(10)
            if 'ratio' in comp_b_display.columns:
                st.dataframe(
                    comp_b_display,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "ratio": st.column_config.NumberColumn(
                            "ratio",
                            format="%.2f"
                        )
                    }
                )
            else:
                st.dataframe(comp_b_display, width='stretch', hide_index=True)
    else:
        st.info("부품 데이터가 부족합니다.")


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

            all_cluster_data.append({
                'cluster': cluster_id,
                'total_count': data['total_count'],
                'deaths': data['harm_summary']['total_deaths'],
                'serious_injuries': data['harm_summary']['total_serious_injuries'],
                'minor_injuries': data['harm_summary']['total_minor_injuries'],
                'no_harm': data['harm_summary']['total_no_injuries']
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
    fig_bar.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

    st.markdown("---")

    # ==================== 2. 클러스터별 환자 피해 분포 (적층 바) ====================
    st.markdown("#### 🎯 클러스터별 환자 피해 분포")

    fig_stacked = go.Figure()

    fig_stacked.add_trace(go.Bar(
        name='Death',
        x=overview_df['cluster_label'],
        y=overview_df['deaths'],
        marker_color=ChartStyles.DANGER_COLOR
    ))

    fig_stacked.add_trace(go.Bar(
        name='Serious Injury',
        x=overview_df['cluster_label'],
        y=overview_df['serious_injuries'],
        marker_color=ChartStyles.WARNING_COLOR
    ))

    fig_stacked.add_trace(go.Bar(
        name='Minor Injury',
        x=overview_df['cluster_label'],
        y=overview_df['minor_injuries'],
        marker_color='#ffd700'
    ))

    fig_stacked.add_trace(go.Bar(
        name='No Harm',
        x=overview_df['cluster_label'],
        y=overview_df['no_harm'],
        marker_color=ChartStyles.SUCCESS_COLOR
    ))

    fig_stacked.update_layout(
        barmode='stack',
        xaxis_title="클러스터",
        yaxis_title="케이스 수",
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

    # 요약 테이블
    with st.expander("📋 전체 클러스터 요약 테이블"):
        display_df = overview_df[[
            'cluster_label', 'total_count', 'deaths',
            'serious_injuries', 'minor_injuries', 'no_harm', 'cfr'
        ]].rename(columns={
            'cluster_label': '클러스터',
            'total_count': '전체 케이스',
            'deaths': '사망',
            'serious_injuries': '중증 부상',
            'minor_injuries': '경증 부상',
            'no_harm': '부상 없음',
            'cfr': '치명률 (%)'
        })

        # 소수점 2자리 표시 포맷 적용
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "치명률 (%)": st.column_config.NumberColumn(
                    "치명률 (%)",
                    format="%.2f"
                )
            }
        )


def render_cluster_insights(lf, available_clusters, selected_dates, year_month_expr, manufacturers, products):
    """자동 인사이트 생성"""
    st.subheader("💡 핵심 인사이트")

    insights = []

    with st.spinner("인사이트 생성 중..."):
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
            "text": f"📊 **Cluster {largest_cluster[0]}**가 가장 많은 케이스를 포함합니다 ({largest_cluster[1]['total_count']:,}건)"
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
                "text": f"⚠️ **Cluster {highest_cfr[0]}**의 치명률이 **{highest_cfr[1]:.2f}%**로 가장 높습니다 (중대 피해 {highest_cfr[2]:,}건)"
            })

        # 3. 가장 안전한 클러스터
        lowest_cfr = min(cfr_rates, key=lambda x: x[1])
        insights.append({
            "type": "success",
            "text": f"✅ **Cluster {lowest_cfr[0]}**의 치명률이 **{lowest_cfr[1]:.2f}%**로 가장 낮습니다"
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
                    "text": f"🔧 **여러 클러스터에서 공통으로 발견된 문제 부품**: {common_parts}"
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

    # 권장 사항
    st.markdown("### 🎯 권장 사항")

    recommendations = []

    # 치명률 높은 클러스터에 대한 권장
    if highest_cfr[1] > 5.0:
        recommendations.append(f"- **Cluster {highest_cfr[0]}**에 대한 집중 조사 및 안전성 개선이 필요합니다")

    # 케이스 수 많은 클러스터
    if largest_cluster[1]['total_count'] > 100:
        recommendations.append(f"- **Cluster {largest_cluster[0]}**의 대량 케이스에 대한 패턴 분석을 수행하세요")

    # 공통 부품
    if all_components:
        recommendations.append(f"- 여러 클러스터에서 반복되는 문제 부품에 대한 근본 원인 분석이 필요합니다")

    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.markdown("- 현재 데이터에서 특별한 조치가 필요한 항목은 없습니다")

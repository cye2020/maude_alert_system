# eda_tab.py (전면 리팩토링 버전)
import streamlit as st
import polars as pl
import pandas as pd

# utils 함수 import
from utils.constants import ColumnNames, Defaults, PatientHarmLevels, DisplayNames, Terms
from utils.data_utils import get_year_month_expr
from utils.filter_helpers import (
    get_available_filters,
    get_available_defect_types
)
from utils.analysis import (
    get_filtered_products,
    get_monthly_counts,
    analyze_manufacturer_defects,
    analyze_defect_components,
    calculate_cfr_by_device
)
from utils.analysis_cluster import (
    get_available_clusters,
    cluster_keyword_unpack,
    get_patient_harm_summary
)
from dashboard.utils.ui_components import (
    render_filter_summary_badge,
    convert_date_range_to_months,
    create_harm_pie_chart,
    # render_bookmark_manager  # 북마크 기능 비활성화
)

# 기존 북마크 함수들은 ui_components.py의 render_bookmark_manager로 통합됨


def show(filters=None, lf: pl.LazyFrame = None):
    """EDA 탭 메인 함수 (전면 리팩토링)

    Args:
        filters: 사이드바 필터 값 (딕셔너리)
        lf: LazyFrame 데이터 (Home.py에서 전달)
    """
    from utils.constants import DisplayNames

    st.title(DisplayNames.FULL_TITLE_EDA)

    # 데이터 확인
    if lf is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    # ==================== 사이드바 필터 추출 ====================
    date_range = filters.get("date_range")  # (start, end) tuple
    manufacturers = filters.get("manufacturers", [])
    products = filters.get("products", [])
    devices = filters.get("devices", [])
    clusters = filters.get("clusters", [])
    defect_types = filters.get("defect_types", [])
    top_n = filters.get("top_n", Defaults.TOP_N)
    min_cases = filters.get("min_cases", Defaults.MIN_CASES)

    # 날짜 범위 → 년-월 리스트 변환 (공통 함수 사용)
    selected_dates = convert_date_range_to_months(date_range)

    # ==================== 북마크 관리 (비활성화) ====================
    # render_bookmark_manager(
    #     tab_name="eda",
    #     current_filters=filters,
    #     filter_keys=["date_range", "manufacturers", "products", "devices", "clusters", "defect_types", "top_n", "min_cases"]
    # )

    # ==================== 필터 요약 배지 (공통 함수 사용) ====================
    render_filter_summary_badge(
        date_range=date_range,
        manufacturers=manufacturers,
        products=products,
        devices=devices,
        clusters=clusters,
        defect_types=defect_types,
        top_n=top_n,
        min_cases=min_cases
    )
    st.markdown("---")

    # ==================== 데이터 유효성 검사 ====================
    if not selected_dates:
        st.warning("⚠️ 분석할 기간을 선택해주세요 (사이드바에서 날짜 범위 설정)")
        st.stop()

    try:
        # 년-월 컬럼 생성 표현식 (재사용)
        date_col = ColumnNames.DATE_RECEIVED
        year_month_expr = get_year_month_expr(lf, date_col)

        # ==================== 스마트 인사이트 (새로 추가) ====================
        render_smart_insights(
            lf,
            date_col,
            selected_dates,
            manufacturers,
            products,
            devices,
            clusters,
            defect_types,
            year_month_expr,
            min_cases
        )

        # ==================== 누적 보고서 수 ====================
        render_total_reports_chart(
            lf,
            date_col,
            selected_dates,
            manufacturers,
            products,
            devices,
            clusters,
            defect_types,
            top_n,
            year_month_expr
        )

        # ==================== 제조사-제품군별 결함 분석 ====================
        st.markdown("---")
        render_defect_analysis(
            lf,
            date_col,
            selected_dates,
            manufacturers,
            products,
            devices,
            clusters,
            defect_types,
            year_month_expr
        )

        # ==================== 기기별 치명률(CFR) 분석 ====================
        st.markdown("---")
        render_cfr_analysis(
            lf,
            date_col,
            selected_dates,
            manufacturers,
            products,
            devices,
            clusters,
            defect_types,
            year_month_expr,
            min_cases,
            top_n
        )

        # ==================== 결함 유형별 상위 문제 부품 및 환자 피해 분포 ====================
        st.markdown("---")
        render_cluster_and_event_analysis(
            lf,
            date_col,
            selected_dates,
            manufacturers,
            products,
            devices,
            clusters,
            defect_types,
            year_month_expr
        )

    except Exception as e:
        st.error(f"데이터 분석 중 오류가 발생했습니다: {str(e)}")
        st.exception(e)


def render_smart_insights(
    lf,
    date_col,
    selected_dates,
    manufacturers,
    products,
    devices,
    clusters,
    defect_types,
    year_month_expr,
    min_cases
):
    """스마트 인사이트: 자동 이상 감지 및 주요 발견사항 (terminology 기반)

    Args:
        lf: LazyFrame
        date_col: 날짜 컬럼명
        selected_dates: 현재 기간 (년-월 리스트)
        manufacturers: 선택된 제조사 리스트
        products: 선택된 제품 리스트
        devices: 선택된 기기 리스트
        clusters: 선택된 클러스터 리스트
        defect_types: 선택된 결함 유형 리스트
        year_month_expr: 년-월 표현식
        min_cases: 최소 케이스 수
    """
    from dashboard.utils.terminology import get_term_manager

    term = get_term_manager()
    st.subheader("💡 핵심 인사이트")

    insights = []

    with st.spinner(term.messages.get('analyzing', '분석 중...')):
        # ==================== 1. 상위 보고 제품 ====================
        # 모든 필터 적용
        top_product_df = get_filtered_products(
            lf,
            date_col=date_col,
            selected_dates=selected_dates,
            selected_manufacturers=manufacturers if manufacturers else None,
            selected_products=products if products else None,
            top_n=1,
            _year_month_expr=year_month_expr
        )

        if len(top_product_df) > 0:
            top_mfr_product = top_product_df["manufacturer_product"][0]
            top_count = top_product_df["total_count"][0]
            insights.append({
                "type": "info",
                "text": term.format_message('eda_top_product',
                                           manufacturer_product=top_mfr_product,
                                           count=top_count)
            })

        # ==================== 2. 고위험 CFR 기기 경고 ====================
        # CFR 메트릭: 모든 필터 적용
        cfr_df = calculate_cfr_by_device(
            lf,
            date_col=date_col,
            selected_dates=selected_dates if selected_dates else None,
            selected_manufacturers=manufacturers if manufacturers else None,
            selected_products=products if products else None,
            top_n=5,
            min_cases=min_cases,
            _year_month_expr=year_month_expr
        )

        if len(cfr_df) > 0:
            high_cfr = cfr_df.filter(pl.col("cfr") > 5.0)
            if len(high_cfr) > 0:
                top_device = high_cfr[0, "manufacturer_product"]
                top_cfr = high_cfr[0, "cfr"]
                severe_harm_count = high_cfr[0, "severe_harm_count"]
                insights.append({
                    "type": "error",
                    "text": term.format_message('eda_high_cfr',
                                               device=top_device,
                                               cfr=top_cfr,
                                               count=severe_harm_count)
                })
            else:
                # CFR이 낮으면 긍정적 메시지
                avg_cfr = cfr_df["cfr"].mean()
                if avg_cfr < 1.0:
                    insights.append({
                        "type": "success",
                        "text": term.format_message('eda_avg_cfr_good', avg_cfr=avg_cfr)
                    })

        # ==================== 3. 가장 빈번한 결함 유형 ====================
        defect_stats = analyze_manufacturer_defects(
            lf,
            date_col=date_col,
            selected_dates=selected_dates,
            selected_manufacturers=manufacturers if manufacturers else None,
            selected_products=products if products else None,
            _year_month_expr=year_month_expr
        )

        if len(defect_stats) > 0:
            top_defect = defect_stats.group_by(ColumnNames.DEFECT_TYPE).agg(
                pl.col("count").sum().alias("total")
            ).sort("total", descending=True).head(1)

            if len(top_defect) > 0:
                defect_type = top_defect[ColumnNames.DEFECT_TYPE][0]
                defect_count = top_defect["total"][0]
                insights.append({
                    "type": "info",
                    "text": term.format_message('eda_top_defect_type',
                                               defect_type=defect_type,
                                               count=defect_count)
                })

    # ==================== 인사이트 표시 ====================
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
        st.info(term.messages.get('eda_no_anomaly', '특이사항이 감지되지 않았습니다'))

    st.markdown("---")




def render_total_reports_chart(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    devices,
    clusters,
    defect_types,
    top_n,
    year_month_expr
):
    """누적 보고서 수 차트 렌더링 (하이브리드 필터: 시계열이므로 모든 필터 적용)"""
    """누적 보고서 수 차트 렌더링 (하이브리드 필터: 시계열이므로 모든 필터 적용)"""
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("📊 누적 보고서 수")
    st.subheader("📊 누적 보고서 수")

    # 설명 추가
    with st.expander("ℹ️ 누적 보고서 수란?", expanded=False):
    with st.expander("ℹ️ 누적 보고서 수란?", expanded=False):
        st.markdown("""
        **누적 보고서 수**는 제조사-제품군별로 시간에 따른 부작용 보고 건수를 추적합니다.

        **해석 방법**:
        - **막대 차트**: 선택한 기간 동안의 누적 보고 건수를 비교
        - **선 그래프**: 시간에 따른 보고 건수 추세 파악 (증가/감소/계절성)
        - **영역 차트**: 각 제품군이 전체에서 차지하는 비중 변화 확인

        **인사이트**:
        - 보고 건수가 급증하는 시기는 품질 문제나 리콜 가능성을 시사합니다
        - 지속적으로 상위권을 유지하는 제품군은 집중 모니터링이 필요합니다
        - 계절성 패턴이 있다면 특정 시기에 예방 조치를 강화할 수 있습니다
        """)

    with st.spinner("데이터 분석 중..."):
        # 모든 필터 적용
        # TODO: devices/clusters/defect_types 지원 추가 필요
        result_df = get_filtered_products(
            lf,
            date_col=date_col,
            selected_dates=selected_dates if selected_dates else None,
            selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
            selected_products=selected_products if selected_products else None,
            top_n=top_n,
            _year_month_expr=year_month_expr
        )

        if len(result_df) > 0:
            # 결과 테이블
            display_df = result_df.to_pandas().copy()
            display_df.insert(0, "순위", range(1, len(display_df) + 1))
            display_df = display_df[["순위", "manufacturer_product", "total_count"]]
            display_df.columns = ["순위", "제조사-제품군", "보고 건수"]

            # 월별 데이터
            total_df = get_monthly_counts(
                lf,
                date_col=date_col,
                selected_dates=selected_dates if selected_dates else None,
                selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                selected_products=selected_products if selected_products else None,
                _year_month_expr=year_month_expr
            )

            if len(total_df) > 0:
                total_pandas = total_df.to_pandas()
                top_combinations = display_df.head(top_n)["제조사-제품군"].tolist()
                chart_data = total_pandas[
                    total_pandas["manufacturer_product"].isin(top_combinations)
                ].copy()

                # 차트 타입 선택
                chart_type = st.radio(
                    "차트 타입",
                    ["막대 차트", "선 그래프", "영역 차트"],
                    horizontal=True,
                    key="total_chart_type"
                )

                if selected_dates and len(selected_dates) == 1:
                    # 단일 월 선택 시 막대 차트만 표시
                    st.info("단일 월 선택 시 막대 차트만 표시됩니다.")
                    top_10_df = display_df.head(10).copy()

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=top_10_df["보고 건수"],
                        y=top_10_df["제조사-제품군"],
                        orientation='h',
                        marker=dict(
                            color=top_10_df["보고 건수"],
                            colorscale='Blues',
                            showscale=False
                        ),
                        text=top_10_df["보고 건수"],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>보고 건수: %{x:,}<extra></extra>'
                    ))

                    fig.update_layout(
                        xaxis_title="보고 건수",
                        yaxis_title="",
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=40),
                        yaxis=dict(autorange="reversed"),
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )

                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

                elif chart_type == "막대 차트":
                    # 선택된 기간의 합계 막대 차트
                    top_10_df = display_df.head(10).copy()

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=top_10_df["보고 건수"],
                        y=top_10_df["제조사-제품군"],
                        orientation='h',
                        marker=dict(
                            color=top_10_df["보고 건수"],
                            colorscale='Blues',
                            showscale=False
                        ),
                        text=top_10_df["보고 건수"],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>보고 건수: %{x:,}<extra></extra>'
                    ))

                    fig.update_layout(
                        xaxis_title="보고 건수",
                        yaxis_title="",
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=40),
                        yaxis=dict(autorange="reversed"),
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )

                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

                elif chart_type == "선 그래프":
                    # 상위 5개만 선택해서 가독성 확보
                    top_5_combinations = display_df.head(5)["제조사-제품군"].tolist()
                    line_chart_data = chart_data[
                        chart_data["manufacturer_product"].isin(top_5_combinations)
                    ].copy()

                    fig = go.Figure()

                    for product in top_5_combinations:
                        product_data = line_chart_data[
                            line_chart_data["manufacturer_product"] == product
                        ].sort_values("year_month")

                        fig.add_trace(go.Scatter(
                            x=product_data["year_month"],
                            y=product_data["total_count"],
                            mode='lines+markers',
                            name=product,
                            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>건수: %{y:,}<extra></extra>'
                        ))

                    fig.update_layout(
                        xaxis_title="년-월",
                        yaxis_title="보고 건수",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=40),
                        hovermode='x unified',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                    st.caption("📌 상위 5개 제조사-제품군만 표시됩니다")

                else:  # 영역 차트
                    # 상위 5개만 선택
                    top_5_combinations = display_df.head(5)["제조사-제품군"].tolist()
                    area_chart_data = chart_data[
                        chart_data["manufacturer_product"].isin(top_5_combinations)
                    ].copy()

                    fig = go.Figure()

                    for product in top_5_combinations:
                        product_data = area_chart_data[
                            area_chart_data["manufacturer_product"] == product
                        ].sort_values("year_month")

                        fig.add_trace(go.Scatter(
                            x=product_data["year_month"],
                            y=product_data["total_count"],
                            mode='lines',
                            name=product,
                            stackgroup='one',
                            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>건수: %{y:,}<extra></extra>'
                        ))

                    fig.update_layout(
                        xaxis_title="년-월",
                        yaxis_title="보고 건수",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=40),
                        hovermode='x unified',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                    st.caption("📌 상위 5개 제조사-제품군만 표시됩니다")

            # 테이블 표시
            st.markdown("### 📋 상세 데이터")

            # 다운로드 버튼
            col_dl1, col_dl2 = st.columns([1, 5])
            with col_dl1:
                csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_data,
                    file_name=f"total_reports_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                    key="download_total_reports"
                )

            st.dataframe(display_df, width='stretch', hide_index=True)
        else:
            st.info("선택한 조건에 해당하는 데이터가 없습니다.")


def render_defect_analysis(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    devices,
    clusters,
    defect_types,
    year_month_expr
):
    """제조사-제품군별 결함 분석 렌더링 (하이브리드 필터: defect_types 제외)"""
    st.subheader("🔧 제조사 - 제품군별 결함")

    # 설명 추가
    with st.expander("ℹ️ 제조사-제품군별 결함 분석이란?", expanded=False):
        st.markdown("""
        **제조사-제품군별 결함 분석**은 각 제품에서 발생하는 결함 유형의 분포를 비교합니다.

        **탭 구성**:
        - **상위 5개 비교**: 보고 건수가 많은 상위 5개 제품군의 결함 패턴을 한눈에 비교
        - **1:1 비교**: 두 제품군의 결함 유형별 비율을 직접 대조하여 차이점 분석
        - **개별 분석**: 특정 제품군의 결함 분포를 상세히 확인

        **인사이트**:
        - 특정 결함 유형이 집중된 제품은 해당 부분의 설계/제조 개선이 필요합니다
        - 1:1 비교에서 큰 차이를 보이는 결함은 제품 간 품질 차이를 나타냅니다
        - 여러 제품에서 공통적으로 나타나는 결함은 산업 전반의 기술적 과제입니다
        """)

    if not selected_dates:
        st.info("결함 분석을 위해 년-월을 선택해주세요.")
        return

    with st.spinner("결함 분석 중..."):
        # 결함 유형 분포 분석 (defect_types는 분석 대상이므로 필터 제외)
        # TODO: devices/clusters 지원 추가 필요
        defect_df = analyze_manufacturer_defects(
            lf,
            date_col=date_col,
            selected_dates=selected_dates,
            selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
            selected_products=selected_products if selected_products else None,
            _year_month_expr=year_month_expr
        )

    if len(defect_df) > 0:
        display_df = defect_df.to_pandas()
        unique_manufacturers = display_df["manufacturer_product"].unique()

        if len(unique_manufacturers) > 0:
            # 탭 방식으로 변경
            tab1, tab2, tab3 = st.tabs(["📊 상위 5개 비교", "⚖️ 1:1 비교", "🔍 개별 분석"])

            with tab1:
                # 상위 5개 제조사-제품군 비교
                st.markdown("#### 상위 5개 제조사-제품군 결함 비교")

                # 전체 건수 기준 상위 5개 추출
                top5_manufacturers = (
                    display_df.groupby("manufacturer_product")["count"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .index.tolist()
                )

                top5_df = display_df[display_df["manufacturer_product"].isin(top5_manufacturers)]

                # Plotly로 개선된 비교 차트
                import plotly.graph_objects as go

                fig = go.Figure()

                for manufacturer in top5_manufacturers:
                    mfr_data = top5_df[top5_df["manufacturer_product"] == manufacturer]

                    fig.add_trace(go.Bar(
                        name=manufacturer,
                        x=mfr_data[ColumnNames.DEFECT_TYPE],
                        y=mfr_data["percentage"],
                        text=mfr_data["percentage"].apply(lambda x: f"{x:.2f}%"),
                        textposition='outside',
                        hovertemplate='<b>%{fullData.name}</b><br>결함 유형: %{x}<br>비율: %{y:.2f}%<extra></extra>'
                    ))

                fig.update_layout(
                    barmode='group',
                    xaxis_title="결함 유형",
                    yaxis_title="비율 (%)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

                # 상위 5개 상세 테이블
                with st.expander("📋 상세 데이터"):
                    top5_display = top5_df.rename(columns={
                        "manufacturer_product": "제조사-제품군",
                        ColumnNames.DEFECT_TYPE: "결함 유형",
                        "count": "건수",
                        "percentage": "비율(%)"
                    }).sort_values(["제조사-제품군", "비율(%)"], ascending=[True, False])

                    col_dl1, col_dl2 = st.columns([1, 5])
                    with col_dl1:
                        csv_data = top5_display.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 CSV 다운로드",
                            data=csv_data,
                            file_name=f"defect_top5_comparison_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                            mime="text/csv",
                            key="download_defect_top5"
                        )

                    # 소수점 2자리 표시 포맷 적용
                    st.dataframe(
                        top5_display.style.format({"비율(%)": "{:.2f}"}),
                        width='stretch',
                        hide_index=True
                    )

            with tab2:
                # 1:1 비교 모드
                st.markdown("#### 제조사-제품군 1:1 비교")
                st.caption("두 제조사-제품군의 결함 패턴을 나란히 비교합니다")

                col1, col2 = st.columns(2)

                with col1:
                    compare_a = st.selectbox(
                        "비교 대상 A",
                        options=unique_manufacturers,
                        index=0,
                        key="compare_a_selectbox"
                    )

                with col2:
                    compare_b = st.selectbox(
                        "비교 대상 B",
                        options=unique_manufacturers,
                        index=min(1, len(unique_manufacturers) - 1),
                        key="compare_b_selectbox"
                    )

                if compare_a == compare_b:
                    st.warning("⚠️ 서로 다른 제조사-제품군을 선택해주세요")
                else:
                    # 두 제조사-제품군 데이터 추출
                    data_a = display_df[display_df["manufacturer_product"] == compare_a].copy()
                    data_b = display_df[display_df["manufacturer_product"] == compare_b].copy()

                    # 나란히 비교 차트
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(compare_a, compare_b),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )

                    # A 데이터
                    fig.add_trace(
                        go.Bar(
                            x=data_a[ColumnNames.DEFECT_TYPE],
                            y=data_a["percentage"],
                            name=compare_a,
                            marker_color='#3B82F6',
                            text=data_a["percentage"].apply(lambda x: f"{x:.2f}%"),
                            textposition='outside',
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                    # B 데이터
                    fig.add_trace(
                        go.Bar(
                            x=data_b[ColumnNames.DEFECT_TYPE],
                            y=data_b["percentage"],
                            name=compare_b,
                            marker_color='#F59E0B',
                            text=data_b["percentage"].apply(lambda x: f"{x:.2f}%"),
                            textposition='outside',
                            showlegend=False
                        ),
                        row=1, col=2
                    )

                    fig.update_xaxes(title_text="결함 유형", row=1, col=1)
                    fig.update_xaxes(title_text="결함 유형", row=1, col=2)
                    fig.update_yaxes(title_text="비율 (%)", row=1, col=1)
                    fig.update_yaxes(title_text="비율 (%)", row=1, col=2)

                    fig.update_layout(height=500)

                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

                    # 차이 분석
                    st.markdown("#### 📊 차이 분석")

                    # 결함 유형별 차이 계산
                    merged = data_a.merge(
                        data_b,
                        on=ColumnNames.DEFECT_TYPE,
                        how='outer',
                        suffixes=('_A', '_B')
                    ).fillna(0)

                    merged['차이 (A-B)'] = merged['percentage_A'] - merged['percentage_B']
                    merged['절대 차이'] = merged['차이 (A-B)'].abs()

                    diff_df = merged[[
                        ColumnNames.DEFECT_TYPE,
                        'percentage_A',
                        'percentage_B',
                        '차이 (A-B)',
                        '절대 차이'
                    ]].sort_values('절대 차이', ascending=False).rename(columns={
                        ColumnNames.DEFECT_TYPE: '결함 유형',
                        'percentage_A': f'{compare_a} (%)',
                        'percentage_B': f'{compare_b} (%)'
                    })

                    # 차이가 큰 결함 유형 강조
                    st.markdown("**가장 큰 차이를 보이는 결함 유형 (Top 3)**")
                    top_diff = diff_df.head(3)

                    for idx, row in top_diff.iterrows():
                        defect = row['결함 유형']
                        diff = row['차이 (A-B)']
                        if diff > 0:
                            st.info(f"🔹 **{defect}**: {compare_a}가 {abs(diff):.2f}%p 더 높음")
                        else:
                            st.info(f"🔸 **{defect}**: {compare_b}가 {abs(diff):.2f}%p 더 높음")

                    # 상세 테이블
                    with st.expander("📋 전체 비교 데이터"):
                        # 소수점 2자리 표시 포맷 적용
                        st.dataframe(
                            diff_df.style.background_gradient(
                                subset=['차이 (A-B)'],
                                cmap='RdYlGn_r',
                                vmin=-50,
                                vmax=50
                            ).format({
                                f"{compare_a} (%)": "{:.2f}",
                                f"{compare_b} (%)": "{:.2f}",
                                "차이 (A-B)": "{:.2f}"
                            }),
                            width='stretch',
                            hide_index=True
                        )

                        col_dl1, col_dl2 = st.columns([1, 5])
                        with col_dl1:
                            csv_data = diff_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 CSV 다운로드",
                                data=csv_data,
                                file_name=f"defect_comparison_{compare_a}_vs_{compare_b}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                                mime="text/csv",
                                key="download_defect_comparison"
                            )

            with tab3:
                # 개별 분석 (기존 방식)
                st.markdown("#### 개별 제조사-제품군 결함 분석")

                selected_manufacturer = st.selectbox(
                    "제조사-제품군 선택",
                    options=unique_manufacturers,
                    index=0,
                    key="defect_individual_selectbox"
                )

                mfr_data = display_df[
                    display_df["manufacturer_product"] == selected_manufacturer
                ].copy()

                if len(mfr_data) > 0:
                    chart_data = pd.DataFrame({
                        "결함 유형": mfr_data[ColumnNames.DEFECT_TYPE].astype(str),
                        "건수": mfr_data["count"],
                        "비율(%)": mfr_data["percentage"]
                    }).sort_values("건수", ascending=False)

                    st.bar_chart(
                        chart_data.set_index("결함 유형")[["비율(%)"]],
                        width='stretch'
                    )

                    # 다운로드 버튼
                    col_dl1, col_dl2 = st.columns([1, 5])
                    with col_dl1:
                        csv_data = chart_data[["결함 유형", "건수", "비율(%)"]].to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 CSV 다운로드",
                            data=csv_data,
                            file_name=f"defect_analysis_{selected_manufacturer}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                            mime="text/csv",
                            key="download_defect_single"
                        )

                    # 소수점 2자리 표시 포맷 적용
                    st.dataframe(
                        chart_data[["결함 유형", "건수", "비율(%)"]].style.format({"비율(%)": "{:.2f}"}),
                        width='stretch',
                        hide_index=True
                    )
                else:
                    st.info(f"{selected_manufacturer}에 대한 결함 데이터가 없습니다.")
        else:
            st.info("결함 데이터가 없습니다.")
    else:
        st.info("선택한 조건에 해당하는 결함 데이터가 없습니다.")


def render_component_analysis(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    year_month_expr,
    top_n
):
    """문제 부품 분석 렌더링

    Args:
        lf: LazyFrame
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품 리스트
        year_month_expr: 년-월 표현식
        top_n: 상위 N개 표시 (사이드바에서 전달)
    """
    st.subheader("🔩 문제 부품 분석")

    # 설명 추가
    with st.expander("ℹ️ 문제 부품 분석이란?", expanded=False):
        st.markdown("""
        **문제 부품 분석**은 특정 결함 유형에서 어떤 부품이 가장 자주 문제를 일으키는지 식별합니다.

        **사용 방법**:
        1. 결함 유형을 선택합니다 (예: 기계적 결함, 전기적 결함 등)
        2. 해당 결함 유형에서 보고된 문제 부품의 순위와 비율을 확인합니다

        **인사이트**:
        - 상위권 문제 부품은 우선적으로 품질 관리 및 개선이 필요합니다
        - 특정 부품이 압도적으로 높은 비율을 차지한다면 해당 부품의 재설계나 공급업체 변경을 고려해야 합니다
        - 시간이 지나도 지속적으로 상위권에 있는 부품은 구조적 문제를 가질 가능성이 있습니다
        """)


    if not selected_dates:
        st.info("문제 부품 분석을 위해 년-월을 선택해주세요.")
        return

    try:
        with st.spinner("결함 유형 목록 로딩 중..."):
            available_defect_types = get_available_defect_types(
                lf,
                date_col=date_col,
                selected_dates=selected_dates,
                selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                selected_products=selected_products if selected_products else None,
                _year_month_expr=year_month_expr
            )

        if len(available_defect_types) > 0:
            # 결함 유형 선택 (세션 상태 유지)
            prev_selected_defect_type = st.session_state.get('prev_selected_defect_type', None)
            default_index = 0
            if prev_selected_defect_type and prev_selected_defect_type in available_defect_types:
                default_index = available_defect_types.index(prev_selected_defect_type)

            selected_defect_type = st.selectbox(
                "결함 유형 선택",
                options=available_defect_types,
                index=default_index,
                help=f"분석할 결함 유형을 선택하세요 (상위 {top_n}개 표시)",
                key='defect_type_selectbox'
            )
            st.session_state.prev_selected_defect_type = selected_defect_type

            if selected_defect_type:
                with st.spinner("문제 부품 분석 중..."):
                    component_df = analyze_defect_components(
                        lf,
                        defect_type=selected_defect_type,
                        date_col=date_col,
                        selected_dates=selected_dates,
                        selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                        selected_products=selected_products if selected_products else None,
                        top_n=top_n,
                        _year_month_expr=year_month_expr
                    )

                if component_df is not None and len(component_df) > 0:
                    display_df = component_df.to_pandas().copy()

                    display_df[ColumnNames.PROBLEM_COMPONENTS] = display_df[ColumnNames.PROBLEM_COMPONENTS].apply(
                        lambda x: str(x) if x is not None else "(NULL)"
                    )

                    display_df.insert(0, "순위", range(1, len(display_df) + 1))
                    display_df = display_df[["순위", ColumnNames.PROBLEM_COMPONENTS, "count", "percentage"]]
                    display_df.columns = ["순위", "문제 부품", "건수", "비율(%)"]

                    # 다운로드 버튼
                    col_dl1, col_dl2 = st.columns([1, 5])
                    with col_dl1:
                        csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 CSV 다운로드",
                            data=csv_data,
                            file_name=f"component_analysis_{selected_defect_type}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                            mime="text/csv",
                            key="download_component_analysis"
                        )

                    # 소수점 2자리 표시 포맷 적용
                    st.dataframe(
                        display_df.style.format({"비율(%)": "{:.2f}"}),
                        width='stretch',
                        hide_index=True
                    )
                else:
                    st.info(f"'{selected_defect_type}' 결함 유형에 대한 문제 부품 데이터가 없습니다.")
        else:
            st.info("선택한 조건에 해당하는 결함 유형이 없습니다.")

    except Exception as e:
        st.error(f"문제 부품 분석 중 오류가 발생했습니다: {str(e)}")
        st.exception(e)


def render_cfr_analysis(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    devices,
    clusters,
    defect_types,
    year_month_expr,
    sidebar_min_cases,
    sidebar_top_n
):
    """기기별 치명률(CFR) 분석 렌더링 (하이브리드 필터: 모든 필터 적용)"""
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("💀 기기별 치명률(CFR) 분석")

    # 설명 추가
    with st.expander("ℹ️ 치명률(CFR) 분석이란?", expanded=False):
        st.markdown("""
        치명률(Case Fatality Rate, CFR)은 전체 부작용 보고 건수 중 중대한 피해(사망, 중증 부상)가 발생한 비율을 나타냅니다.

        **측정 방식**:
        - CFR (%) = (중대 피해 건수 / 총 보고 건수) × 100
        - 중대 피해 = 사망 + 중증 부상

        **시각화 해석**:
        - **막대 차트**: CFR이 높은 상위 10개 제품군을 보여줍니다
        - **산점도**: 보고 건수(x축)와 CFR(y축)의 관계를 표시하며, 버블 크기는 중대 피해 건수를 나타냅니다
        - **통계적 유의성**: Fisher's Exact Test를 통해 평균 CFR과 통계적으로 유의한 차이를 보이는 제품을 식별합니다

        **인사이트**:
        - CFR이 높은 제품은 발생 시 심각한 결과를 초래하므로 즉각적인 안전 조치가 필요합니다
        - 보고 건수는 적지만 CFR이 높은 제품(산점도 왼쪽 위)은 '저빈도 고위험' 제품으로 특별 관리가 필요합니다
        - p-value < 0.05인 제품은 통계적으로 유의하게 평균보다 위험하거나 안전한 제품입니다
        """)

    try:
        # 사이드바에서 설정된 값 사용
        top_n_cfr = sidebar_top_n
        min_cases = sidebar_min_cases

        st.caption(f"💡 사이드바 설정: 상위 {top_n_cfr}개 표시, 최소 {min_cases}건 이상")

        # CFR 분석: 메트릭이므로 모든 필터 적용
        # TODO: devices/clusters/defect_types 지원 추가 필요
        with st.spinner("기기별 치명률 분석 중..."):
            cfr_result = calculate_cfr_by_device(
                lf,
                date_col=date_col,
                selected_dates=selected_dates if selected_dates else None,
                selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                selected_products=selected_products if selected_products else None,
                top_n=top_n_cfr if top_n_cfr else None,
                min_cases=min_cases,
                _year_month_expr=year_month_expr
            )

        if len(cfr_result) > 0:
            # terminology 사용
            from dashboard.utils.terminology import get_term_manager
            term = get_term_manager()

            display_df = cfr_result.to_pandas().copy()

            display_df.insert(0, "순위", range(1, len(display_df) + 1))
            display_df = display_df[[
                "순위", "manufacturer_product", "total_cases",
                "death_count", "serious_injury_count", "minor_injury_count",
                "severe_harm_count", "cfr"
            ]]
            display_df.columns = [
                "순위",
                term.korean.entities.manufacturer_product,
                term.korean.metrics.total_count,
                term.korean.metrics.death_count,
                term.korean.metrics.serious_injury,
                term.korean.metrics.minor_injury,
                term.korean.metrics.severe_harm,
                f"{term.korean.metrics.cfr}(%)"
            ]

            # ==================== 요약 통계 (상단 배치) ====================
            # terminology 기반 컬럼명 재사용
            col_cfr = f"{term.korean.metrics.cfr}(%)"

            st.markdown("### 📊 요약 통계")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:
                st.metric("분석 기기 수", f"{len(display_df):,}개")

            with summary_col2:
                min_cfr = display_df[col_cfr].min()
                st.metric(f"최소 {term.korean.metrics.cfr}", f"{min_cfr:.2f}%")

            with summary_col3:
                max_cfr = display_df[col_cfr].max()
                st.metric(f"최대 {term.korean.metrics.cfr}", f"{max_cfr:.2f}%")

            with summary_col4:
                cfr_range = max_cfr - min_cfr
                st.metric(f"{term.korean.metrics.cfr} 범위", f"{cfr_range:.2f}%p")

            st.markdown("---")

            # ==================== 시각화 섹션 ====================
            # terminology 기반 컬럼명들
            col_manufacturer_product = term.korean.entities.manufacturer_product
            col_total_count = term.korean.metrics.total_count
            col_severe_harm = term.korean.metrics.severe_harm
            col_death = term.korean.metrics.death_count
            col_serious_injury = term.korean.metrics.serious_injury

            st.markdown(f"### 📈 {term.korean.metrics.cfr} 시각화")

            viz_col1, viz_col2 = st.columns(2)

            # 좌측: CFR Top 10 막대 차트
            with viz_col1:
                st.markdown(f"#### 상위 10개 {col_manufacturer_product} {term.korean.metrics.cfr}")
                top_10_df = display_df.head(10).copy()

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=top_10_df[col_cfr],
                    y=top_10_df[col_manufacturer_product],
                    orientation='h',
                    marker=dict(
                        color=top_10_df[col_cfr],
                        colorscale='Reds',
                        showscale=False,
                        line=dict(color='rgba(0,0,0,0.2)', width=1)
                    ),
                    text=top_10_df[col_cfr].apply(lambda x: f"{x:.2f}%"),
                    textposition='outside',
                    hovertemplate=f'<b>%{{y}}</b><br>{term.korean.metrics.cfr}: %{{x:.2f}}%<br>순위: %{{customdata}}<extra></extra>',
                    customdata=top_10_df["순위"]
                ))

                fig_bar.update_layout(
                    xaxis_title=f"{term.korean.metrics.cfr} (%)",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=40),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgray',
                        gridwidth=0.5
                    )
                )

                st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

            # 우측: 치명률 vs 총 건수 산점도
            with viz_col2:
                st.markdown(f"#### {term.korean.metrics.cfr} vs {col_total_count} ({col_severe_harm} 크기)")

                fig_scatter = px.scatter(
                    display_df,
                    x=col_total_count,
                    y=col_cfr,
                    size=col_severe_harm,
                    color=col_cfr,
                    color_continuous_scale='Reds',
                    hover_name=col_manufacturer_product,
                    hover_data={
                        "순위": True,
                        col_total_count: ":,",
                        col_cfr: ":.2f",
                        col_death: True,
                        col_serious_injury: True,
                        col_severe_harm: True
                    },
                    labels={
                        col_total_count: f"총 {term.korean.metrics.report_count}",
                        col_cfr: f"{term.korean.metrics.cfr} (%)",
                        col_severe_harm: f"{col_severe_harm} 건수"
                    }
                )

                fig_scatter.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgray',
                        gridwidth=0.5,
                        type='log' if len(display_df) > 0 and display_df[col_total_count].max() > 1000 else 'linear'
                    ),
                    yaxis=dict(
                        gridcolor='lightgray',
                        gridwidth=0.5
                    )
                )

                st.plotly_chart(fig_scatter, width='stretch', config={'displayModeBar': False})

            st.markdown("---")

            # ==================== 통계적 유의성 검정 ====================
            st.markdown("### 📊 통계적 유의성 분석")
            st.caption(f"평균 {term.korean.metrics.cfr}과의 비교를 통한 통계적 유의성 검정")

            try:
                from utils.statistical_tests import (
                    fisher_exact_test,
                    interpret_significance,
                    calculate_confidence_interval,
                    get_significance_level
                )

                # 위에서 이미 정의한 컬럼명 변수들 재사용
                # col_manufacturer_product, col_severe_harm, col_total_count, col_cfr

                # 전체 평균 CFR 계산 (치명률 = 중대피해/총건수)
                total_severe_harm = display_df[col_severe_harm].sum()
                total_cases = display_df[col_total_count].sum()
                overall_cfr = (total_severe_harm / total_cases * 100) if total_cases > 0 else 0

                st.info(f"📌 전체 평균 {term.korean.metrics.cfr}: **{overall_cfr:.2f}%** ({term.korean.metrics.severe_harm} {total_severe_harm:,}건 / 총 {total_cases:,}건)")

                # 통계 검정 결과
                significance_results = []

                for idx, row in display_df.head(10).iterrows():
                    device = row[col_manufacturer_product]
                    device_severe_harm = int(row[col_severe_harm])
                    device_total = int(row[col_total_count])
                    device_cfr = row[col_cfr]

                    # 나머지 데이터
                    other_severe_harm = total_severe_harm - device_severe_harm
                    other_total = total_cases - device_total

                    if other_total > 0:
                        # Fisher's Exact Test (중대피해 기준)
                        odds_ratio, p_value = fisher_exact_test(
                            device_severe_harm, device_total,
                            other_severe_harm, other_total
                        )

                        # 신뢰구간 계산
                        ci_lower, ci_upper = calculate_confidence_interval(device_severe_harm, device_total)

                        significance_results.append({
                            col_manufacturer_product: device,
                            col_cfr: device_cfr,
                            "95% CI": f"[{ci_lower:.2f}, {ci_upper:.2f}]",
                            "p-value": p_value,
                            "유의성": get_significance_level(p_value),
                            "해석": interpret_significance(p_value)
                        })

                if significance_results:
                    sig_df = pd.DataFrame(significance_results)

                    # 유의한 결과만 강조 표시
                    significant_devices = sig_df[sig_df["p-value"] < 0.05]

                    if len(significant_devices) > 0:
                        st.markdown("**🔴 통계적으로 유의한 기기 (p < 0.05)**")
                        for _, row in significant_devices.iterrows():
                            device = row[col_manufacturer_product]
                            cfr = row[col_cfr]
                            sig = row["유의성"]
                            interpretation = row["해석"]
                            ci = row["95% CI"]

                            if cfr > overall_cfr:
                                st.error(f"**{device}** {sig}: {term.korean.metrics.cfr} {cfr:.2f}% (평균보다 높음) - {interpretation}, 95% CI {ci}")
                            else:
                                st.success(f"**{device}** {sig}: {term.korean.metrics.cfr} {cfr:.2f}% (평균보다 낮음) - {interpretation}, 95% CI {ci}")
                    else:
                        st.info("통계적으로 유의한 차이를 보이는 기기가 없습니다 (α = 0.05)")

                    # 상세 테이블
                    with st.expander("📋 통계 검정 상세 결과"):
                        # 소수점 2자리 표시 포맷 적용
                        st.dataframe(
                            sig_df.style.apply(
                                lambda x: ['background-color: #fee' if v < 0.05 else '' for v in x],
                                subset=['p-value']
                            ).format({
                                col_cfr: "{:.2f}",
                                "Odds Ratio": "{:.2f}",
                                "p-value": "{:.4f}"
                            }),
                            width='stretch',
                            hide_index=True
                        )

                        st.caption("""
                        **범례:**
                        - *** : p < 0.001 (매우 유의함)
                        - ** : p < 0.01 (유의함)
                        - * : p < 0.05 (유의함)
                        - CI: Confidence Interval (신뢰구간)
                        """)

            except Exception as e:
                st.warning(f"통계적 유의성 검정 중 오류 발생: {str(e)}")

            st.markdown("---")

            # ==================== 데이터 테이블 ====================
            st.markdown("### 📋 상세 데이터")

            # 다운로드 버튼
            col_dl1, col_dl2 = st.columns([1, 5])
            with col_dl1:
                csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_data,
                    file_name=f"cfr_analysis_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                    key="download_cfr_analysis"
                )

            # 소수점 2자리 표시 포맷 적용
            st.dataframe(
                display_df.style.format({"치명률(%)": "{:.2f}"}),
                width='stretch',
                hide_index=True
            )

        else:
            st.info(f"선택한 조건에 해당하는 데이터가 없습니다. (최소 {min_cases}건 이상의 보고 건수 필요)")

    except Exception as e:
        st.error(f"기기별 치명률 분석 중 오류가 발생했습니다: {str(e)}")
        st.exception(e)


def render_cluster_and_event_analysis(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    devices,
    clusters,
    defect_types,
    year_month_expr
):
    """결함 유형별 상위 문제 부품 및 환자 피해 분포 렌더링 (하이브리드 필터: defect_types 제외)"""
    import plotly.graph_objects as go
    import streamlit.components.v1 as components
    import html

    title = Terms.section_title(
        'entity_multi_analysis',
        entity=Terms.KOREAN.DEFECT_TYPE,
        item1='상위 문제 부품',
        item2='환자 피해 분포'
    )

    st.subheader(f"📊 {title}")

    # 설명 추가
    with st.expander(f"ℹ️ {Terms.KOREAN.DEFECT_TYPE}별 상위 문제 부품 및 환자 피해 분포란?", expanded=False):
        st.markdown("""
        **이 섹션**은 결함 유형(결함 유형)별로 어떤 문제 부품이 많이 보고되었는지, 그리고 전체적으로 환자 피해가 어떻게 분포되어 있는지 보여줍니다.

        **환자 피해 분포 (파이 차트)**:
        - 선택한 조건에서 발생한 환자 피해를 사망, 중증 부상, 경증 부상, 부상 없음으로 분류합니다
        - 전체 부작용 보고 중 실제로 심각한 피해로 이어진 비율을 파악할 수 있습니다
        - 결함 유형 필터를 선택하면 해당 결함 유형의 환자 피해 분포만 표시됩니다

        **결함 유형별 상위 문제 부품**:
        - 특정 결함 유형(카테고리)을 선택하면 해당 결함에서 가장 빈번하게 보고된 문제 부품 상위 10개를 표시합니다
        - 각 부품의 건수와 비율을 직관적인 막대 차트로 확인할 수 있습니다

        **인사이트**:
        - 사망/중증 부상 비율이 높다면 해당 조건의 제품들은 고위험군으로 분류됩니다
        - 특정 부품이 압도적으로 높은 비율을 차지한다면 해당 부품의 품질 개선이 시급합니다
        - 결함 유형과 문제 부품을 함께 분석하면 근본 원인을 더 명확히 파악할 수 있습니다
        """)

    try:
        # 사용 가능한 결함 유형 가져오기 (defect_types는 분석 대상이므로 필터 제외)
        # TODO: devices/clusters 지원 추가 필요
        with st.spinner("결함 유형 목록 로딩 중..."):
            available_clusters = get_available_clusters(
                lf,
                cluster_col=ColumnNames.DEFECT_TYPE,
                date_col=date_col,
                selected_dates=selected_dates if selected_dates else None,
                selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                selected_products=selected_products if selected_products else None,
                exclude_minus_one=False,  # defect_type은 문자열이므로 -1 제외 안 함
                _year_month_expr=year_month_expr
            )

        if len(available_clusters) > 0:
            # 상단에 결함 유형 선택 필터 배치
            st.markdown("### 결함 유형 선택")

            # 이전에 선택한 결함 유형 가져오기
            prev_selected_cluster = st.session_state.get('prev_selected_cluster', None)
            default_index = 0
            if prev_selected_cluster and prev_selected_cluster in available_clusters:
                default_index = available_clusters.index(prev_selected_cluster)

            selected_cluster = st.selectbox(
                "카테고리 선택",
                options=available_clusters,
                index=default_index,
                help="분석할 결함 유형를 선택하세요",
                key='cluster_selectbox'
            )
            st.session_state.prev_selected_cluster = selected_cluster

            st.markdown("---")

            # 좌우 레이아웃
            event_col, cluster_col = st.columns([1, 1])

            # 우측: 결함 유형별 상위 문제 부품
            with cluster_col:
                st.markdown(f"### {Terms.KOREAN.DEFECT_TYPE}별 상위 문제 부품")

                # 상위 N개 설정 (기본값 10개)
                top_n_cluster = 10

                # 결함 유형별 상위 문제 분석 실행
                if selected_cluster:
                    with st.spinner(f"{Terms.KOREAN.DEFECT_TYPE}별 상위 문제 부품 분석 중..."):
                        cluster_result = cluster_keyword_unpack(
                            lf,
                            col_name=ColumnNames.PROBLEM_COMPONENTS,
                            cluster_col=ColumnNames.DEFECT_TYPE,
                            date_col=date_col,
                            selected_dates=selected_dates if selected_dates else None,
                            selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                            selected_products=selected_products if selected_products else None,
                            top_n=top_n_cluster,
                            _year_month_expr=year_month_expr
                        )

                    # 선택된 결함 유형의 데이터만 필터링
                    cluster_data = cluster_result.filter(
                        pl.col(ColumnNames.DEFECT_TYPE) == selected_cluster
                    )

                    if len(cluster_data) > 0:
                        # 결과를 pandas DataFrame으로 변환
                        display_df = cluster_data.to_pandas().copy()

                        # problem_components를 문자열로 변환
                        display_df[ColumnNames.PROBLEM_COMPONENTS] = display_df[ColumnNames.PROBLEM_COMPONENTS].apply(
                            lambda x: str(x) if x is not None else "(NULL)"
                        )

                        # 정렬 (count 내림차순)
                        display_df = display_df.sort_values('count', ascending=False).reset_index(drop=True)

                        # HTML/CSS를 사용한 부드럽고 둥근 막대 차트
                        max_visible_items = 10  # 화면에 보이는 항목 수
                        item_height = 55  # 각 항목의 높이
                        container_height = max_visible_items * item_height  # 스크롤 컨테이너 높이

                        # 최대 비율 계산 (막대 길이 계산용)
                        max_ratio = display_df['ratio'].max() if len(display_df) > 0 else 100

                        # HTML/CSS 스타일과 컨테이너 (f-string 사용)
                        bar_height = item_height - 10
                        html_content = f"""
                        <style>
                            .cluster-bar-container {{
                                height: {container_height}px;
                                overflow-y: auto;
                                overflow-x: hidden;
                                padding: 10px 5px;
                                scroll-behavior: smooth;
                            }}
                            .cluster-bar-container::-webkit-scrollbar {{
                                width: 8px;
                            }}
                            .cluster-bar-container::-webkit-scrollbar-track {{
                                background: #f1f1f1;
                                border-radius: 10px;
                            }}
                            .cluster-bar-container::-webkit-scrollbar-thumb {{
                                background: #888;
                                border-radius: 10px;
                            }}
                            .cluster-bar-container::-webkit-scrollbar-thumb:hover {{
                                background: #555;
                            }}
                            .cluster-item {{
                                display: flex;
                                align-items: center;
                                gap: 10px;
                                margin-bottom: 12px;
                                padding: 8px 0;
                                transition: transform 0.2s ease;
                            }}
                            .cluster-item:hover {{
                                transform: translateX(3px);
                            }}
                            .component-name {{
                                width: 140px;
                                font-size: 14px;
                                color: #374151;
                                flex-shrink: 0;
                                text-align: left;
                                font-weight: 500;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;
                            }}
                            .bar-wrapper {{
                                flex: 1;
                                position: relative;
                                height: {bar_height}px;
                                background-color: #F3F4F6;
                                border-radius: 20px;
                                overflow: hidden;
                            }}
                            .bar-fill {{
                                position: absolute;
                                left: 0;
                                top: 0;
                                height: 100%;
                                background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
                                border-radius: 20px;
                                transition: width 0.3s ease;
                                box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
                            }}
                            .bar-content {{
                                position: absolute;
                                top: 50%;
                                transform: translateY(-50%);
                                left: 15px;
                                font-size: 15px;
                                font-weight: 600;
                                color: white;
                                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                                z-index: 2;
                            }}
                            .bar-ratio {{
                                position: absolute;
                                top: 50%;
                                transform: translateY(-50%);
                                right: 15px;
                                font-size: 14px;
                                font-weight: 500;
                                color: #6B7280;
                                background-color: rgba(243, 244, 246, 0.95);
                                padding: 5px 10px;
                                border-radius: 12px;
                                z-index: 2;
                                backdrop-filter: blur(4px);
                            }}
                        </style>
                        <div class="cluster-bar-container">
                        """

                        for idx, row in display_df.iterrows():
                            component = row[ColumnNames.PROBLEM_COMPONENTS]
                            count = int(row['count'])
                            ratio = float(row['ratio'])
                            # 막대 길이는 비율에 비례 (최대 비율을 100%로 설정)
                            bar_width = (ratio / max_ratio) * 100 if max_ratio > 0 else 0

                            # 컴포넌트 이름이 너무 길면 자르기
                            display_component = component[:30] + "..." if len(component) > 30 else component

                            # HTML 이스케이프 처리
                            escaped_component = html.escape(str(component))
                            escaped_display = html.escape(str(display_component))

                            html_content += f"""
                            <div class="cluster-item">
                                <div class="component-name" title="{escaped_component}">{escaped_display}</div>
                                <div class="bar-wrapper">
                                    <div class="bar-fill" style="width: {bar_width}%;"></div>
                                    <span class="bar-content">{count:,}</span>
                                    <span class="bar-ratio">{ratio:.2f}%</span>
                                </div>
                            </div>
                            """

                        html_content += "</div>"

                        # HTML 렌더링 (components.html 사용)
                        components.html(html_content, height=container_height + 20, scrolling=True)
                    else:
                        st.info(f"'{selected_cluster}' 결함 유형에 대한 문제 부품 데이터가 없습니다.")

            # 좌측: 환자 피해 분포 파이 차트
            with event_col:
                st.markdown("### 환자 피해 분포")
                st.caption(f"선택된 결함 유형: **{selected_cluster}**")

                with st.spinner("환자 피해 데이터 로딩 중..."):
                    harm_summary = get_patient_harm_summary(
                        lf,
                        event_column=ColumnNames.PATIENT_HARM,
                        date_col=date_col,
                        selected_dates=selected_dates if selected_dates else None,
                        selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                        selected_products=selected_products if selected_products else None,
                        selected_defect_types=[selected_cluster] if selected_cluster else None,
                        _year_month_expr=year_month_expr
                    )

                total_deaths = harm_summary['total_deaths']
                total_serious = harm_summary['total_serious_injuries']
                total_minor = harm_summary['total_minor_injuries']
                total_none = harm_summary['total_no_injuries']
                total_unknown = harm_summary.get('total_unknown', 0)
                total_all = total_deaths + total_serious + total_minor + total_none + total_unknown

                if total_all > 0:
                    # 값이 0보다 큰 항목만 필터링
                    harm_data = [
                        ('사망', total_deaths, '#DC2626'),
                        ('중증 부상', total_serious, '#F59E0B'),
                        ('경증 부상', total_minor, '#ffd700'),
                        ('부상 없음', total_none, '#2ca02c'),
                        ('Unknown', total_unknown, '#9CA3AF')
                    ]

                    # 값이 0보다 큰 항목만 선택
                    filtered_harm_data = [(label, value, color) for label, value, color in harm_data if value > 0]

                    if filtered_harm_data:
                        harm_labels = [item[0] for item in filtered_harm_data]
                        harm_values = [item[1] for item in filtered_harm_data]
                        harm_colors = [item[2] for item in filtered_harm_data]

                        # 파이 차트 데이터 준비
                        pie_data = pd.DataFrame({
                            '유형': harm_labels,
                            '건수': harm_values,
                            '비율': [(v / total_all * 100) for v in harm_values]
                        })

                        # Plotly 파이 차트 생성
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=pie_data['유형'],
                            values=pie_data['건수'],
                            hole=0.4,  # 도넛 차트 스타일
                            marker=dict(
                                colors=harm_colors,
                                line=dict(color='#FFFFFF', width=2)
                            ),
                            textinfo='label+percent+value',
                            texttemplate='%{label}<br>%{value:,}건<br>(%{percent})',
                            hovertemplate='<b>%{label}</b><br>건수: %{value:,}<br>비율: %{percent}<extra></extra>'
                        )])

                        fig_pie.update_layout(
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="middle",
                                y=0.5,
                                xanchor="left",
                                x=1.05
                            ),
                            height=400,
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )

                        # 파이 차트 표시
                        st.plotly_chart(fig_pie, width='stretch', config={'displayModeBar': False})
                    else:
                        st.info("환자 피해 데이터가 없습니다.")

                    # 요약 정보
                    st.markdown("**전체 요약**")
                    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)

                    with summary_col1:
                        st.metric(Terms.KOREAN.DEATH_COUNT, f"{total_deaths:,}건")

                    with summary_col2:
                        st.metric(Terms.KOREAN.SERIOUS_INJURY, f"{total_serious:,}건")

                    with summary_col3:
                        st.metric(Terms.KOREAN.MINOR_INJURY, f"{total_minor:,}건")

                    with summary_col4:
                        st.metric(Terms.KOREAN.NO_HARM, f"{total_none:,}건")

                    with summary_col5:
                        st.metric("Unknown", f"{total_unknown:,}건")
                else:
                    st.info("환자 피해 데이터가 없습니다.")
        else:
            st.info(f"선택한 조건에 해당하는 {Terms.KOREAN.DEFECT_TYPE}가 없습니다.")

    except Exception as e:
        st.error(f"{Terms.KOREAN.DEFECT_TYPE}별 상위 문제 부품 분석 중 오류가 발생했습니다: {str(e)}")
        st.exception(e)

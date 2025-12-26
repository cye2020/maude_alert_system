# eda_tab.py (리팩토링 버전)
import streamlit as st
import polars as pl
import pandas as pd

# utils 함수 import
from utils.constants import ColumnNames, Defaults
from utils.data_utils import get_year_month_expr, get_window_dates
from utils.filter_helpers import (
    get_available_filters,
    get_manufacturers_by_dates,
    get_products_by_manufacturers,
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
from datetime import datetime


def render_bookmark_manager():
    """북마크 관리 UI"""
    with st.expander("🔖 필터 설정 북마크", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            bookmark_name = st.text_input(
                "북마크 이름",
                placeholder="예: 2024년 상반기 분석",
                key="bookmark_name_input"
            )

        with col2:
            if st.button("💾 현재 설정 저장", width='stretch'):
                if bookmark_name:
                    save_bookmark(bookmark_name)
                    st.success(f"✅ '{bookmark_name}' 저장됨")
                    st.rerun()
                else:
                    st.warning("북마크 이름을 입력하세요")

        with col3:
            if st.button("🗑️ 모두 삭제", width='stretch'):
                if 'bookmarks' in st.session_state:
                    del st.session_state.bookmarks
                    st.success("모든 북마크 삭제됨")
                    st.rerun()

        # 저장된 북마크 목록
        if 'bookmarks' in st.session_state and st.session_state.bookmarks:
            st.markdown("**저장된 북마크:**")

            for bookmark_id, bookmark_data in st.session_state.bookmarks.items():
                col_a, col_b, col_c = st.columns([3, 1, 1])

                with col_a:
                    st.caption(f"📌 **{bookmark_data['name']}** - {bookmark_data['timestamp']}")

                with col_b:
                    if st.button("불러오기", key=f"load_{bookmark_id}", width='stretch'):
                        load_bookmark(bookmark_data)
                        st.success(f"'{bookmark_data['name']}' 불러옴")
                        st.rerun()

                with col_c:
                    if st.button("삭제", key=f"delete_{bookmark_id}", width='stretch'):
                        del st.session_state.bookmarks[bookmark_id]
                        st.success("북마크 삭제됨")
                        st.rerun()

                # 북마크 상세 정보
                with st.expander(f"상세 정보: {bookmark_data['name']}", expanded=False):
                    st.json(bookmark_data['filters'])
        else:
            st.info("저장된 북마크가 없습니다")


def save_bookmark(name: str):
    """현재 필터 설정을 북마크로 저장"""
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = {}

    # 현재 필터 상태 수집
    current_filters = {
        'selected_dates': st.session_state.get('prev_selected_dates', []),
        'selected_manufacturers': st.session_state.get('prev_selected_manufacturers', []),
        'selected_products': st.session_state.get('prev_selected_products', []),
        'as_of_month': st.session_state.get('selected_as_of_month'),
        'window': st.session_state.get('selected_window'),
        'top_n': st.session_state.get('top_n'),
        'min_cases': st.session_state.get('min_cases')
    }

    bookmark_id = f"bookmark_{len(st.session_state.bookmarks)}"
    st.session_state.bookmarks[bookmark_id] = {
        'name': name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filters': current_filters
    }


def load_bookmark(bookmark_data: dict):
    """저장된 북마크에서 필터 설정 불러오기"""
    filters = bookmark_data['filters']

    # 세션 상태에 필터 값 복원
    if filters.get('selected_dates'):
        st.session_state.prev_selected_dates = filters['selected_dates']

    if filters.get('selected_manufacturers'):
        st.session_state.prev_selected_manufacturers = filters['selected_manufacturers']

    if filters.get('selected_products'):
        st.session_state.prev_selected_products = filters['selected_products']

    if filters.get('as_of_month'):
        st.session_state.selected_as_of_month = filters['as_of_month']

    if filters.get('window'):
        st.session_state.selected_window = filters['window']

    if filters.get('top_n'):
        st.session_state.top_n = filters['top_n']

    if filters.get('min_cases'):
        st.session_state.min_cases = filters['min_cases']


def show(filters=None, lf: pl.LazyFrame = None):
    """EDA 탭 메인 함수

    Args:
        filters: 사이드바 필터 값
        lf: LazyFrame 데이터 (Home.py에서 전달)
    """
    st.title("📈 Detailed Analytics")

    # 북마크 관리 UI (상단 배치)
    render_bookmark_manager()

    # 사이드바 필터 값 가져오기
    selected_year_month = filters.get("as_of_month")  # 기준 월
    sidebar_window = filters.get("window", 1)  # 윈도우 크기
    sidebar_top_n = filters.get("top_n", Defaults.TOP_N)  # 상위 N개
    sidebar_min_cases = filters.get("min_cases", Defaults.MIN_CASES)  # 최소 건수

    # 데이터 확인
    if lf is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    try:
        # 년-월 컬럼 생성 표현식을 한 번만 계산 (재사용)
        date_col = ColumnNames.DATE_RECEIVED
        year_month_expr = get_year_month_expr(lf, date_col)

        # 사용 가능한 필터 옵션 가져오기
        with st.spinner("필터 옵션 로딩 중..."):
            available_dates, available_manufacturers, available_products = get_available_filters(
                lf,
                date_col=date_col,
                _year_month_expr=year_month_expr
            )

        if len(available_dates) == 0:
            st.warning("사용 가능한 날짜 데이터가 없습니다. 데이터 파일과 날짜 컬럼을 확인해주세요.")
            st.stop()

        # 필터 UI 렌더링 (사이드바 값 전달)
        selected_dates, selected_manufacturers, selected_products = render_filter_ui(
            available_dates,
            available_manufacturers,
            available_products,
            lf,
            date_col,
            year_month_expr,
            selected_year_month,
            sidebar_window
        )

        # 인사이트 요약 (필터 선택 후)
        if selected_dates:
            render_insights_summary(
                lf,
                date_col,
                selected_dates,
                selected_manufacturers,
                selected_products,
                sidebar_top_n,
                year_month_expr
            )

        # 월별 보고서 수 그래프
        render_monthly_reports_chart(
            lf,
            date_col,
            selected_dates,
            selected_manufacturers,
            selected_products,
            sidebar_top_n,
            year_month_expr
        )

        # 제조사 - 모델별 결함 분석
        st.markdown("---")
        render_defect_analysis(
            lf,
            date_col,
            selected_dates,
            selected_manufacturers,
            selected_products,
            year_month_expr
        )

        # 문제 부품 분석
        st.markdown("---")
        render_component_analysis(
            lf,
            date_col,
            selected_dates,
            selected_manufacturers,
            selected_products,
            year_month_expr
        )

        # 기기별 치명률(CFR) 분석
        st.markdown("---")
        render_cfr_analysis(
            lf,
            date_col,
            selected_dates,
            selected_manufacturers,
            selected_products,
            year_month_expr,
            sidebar_min_cases,
            sidebar_top_n
        )

        # defect type별 상위 문제 & 사건 유형별 분포
        st.markdown("---")
        render_cluster_and_event_analysis(
            lf,
            date_col,
            selected_dates,
            selected_manufacturers,
            selected_products,
            year_month_expr
        )

    except Exception as e:
        st.error(f"데이터 분석 중 오류가 발생했습니다: {str(e)}")
        st.exception(e)


def render_insights_summary(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    top_n,
    year_month_expr
):
    """인사이트 요약 섹션 렌더링"""
    st.markdown("### 💡 주요 발견사항")

    with st.spinner("인사이트 생성 중..."):
        # 1. 월별 보고서 수 Top 1
        top_product = get_filtered_products(
            lf,
            date_col=date_col,
            selected_dates=selected_dates,
            selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
            selected_products=selected_products if selected_products else None,
            top_n=1,
            _year_month_expr=year_month_expr
        )

        # 2. 가장 많은 결함 유형
        defect_stats = analyze_manufacturer_defects(
            lf,
            date_col=date_col,
            selected_dates=selected_dates,
            selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
            selected_products=selected_products if selected_products else None,
            _year_month_expr=year_month_expr
        )

        # 3. CFR Top 3
        cfr_top = calculate_cfr_by_device(
            lf,
            date_col=date_col,
            selected_dates=selected_dates if selected_dates else None,
            selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
            selected_products=selected_products if selected_products else None,
            top_n=3,
            min_cases=10,
            _year_month_expr=year_month_expr
        )

        # 인사이트 생성
        insights = []

        if len(top_product) > 0:
            top_mfr_product = top_product["manufacturer_product"][0]
            top_count = top_product["total_count"][0]
            insights.append(f"**{top_mfr_product}**의 보고 건수가 **{top_count:,}건**으로 가장 많습니다")

        if len(defect_stats) > 0:
            # 가장 많은 결함 유형 찾기
            top_defect = defect_stats.group_by(ColumnNames.DEFECT_TYPE).agg(
                pl.col("count").sum().alias("total")
            ).sort("total", descending=True).head(1)

            if len(top_defect) > 0:
                defect_type = top_defect[ColumnNames.DEFECT_TYPE][0]
                defect_count = top_defect["total"][0]
                insights.append(f"가장 빈번한 결함 유형은 **{defect_type}** ({defect_count:,}건)입니다")

        if len(cfr_top) > 0:
            highest_cfr_product = cfr_top["manufacturer_product"][0]
            highest_cfr = cfr_top["cfr"][0]
            cfr_deaths = cfr_top["death_count"][0]

            if highest_cfr > 5.0:  # CFR이 5% 이상이면 경고
                insights.append(f"⚠️ **{highest_cfr_product}**의 치명률이 **{highest_cfr:.2f}%**로 매우 높습니다 (사망 {cfr_deaths:,}건)")
            else:
                insights.append(f"**{highest_cfr_product}**의 치명률이 **{highest_cfr:.2f}%**로 가장 높습니다 (사망 {cfr_deaths:,}건)")

        # 인사이트 표시
        if insights:
            for idx, insight in enumerate(insights, 1):
                st.info(f"{idx}. {insight}")
        else:
            st.info("선택한 조건에서 인사이트를 생성할 수 없습니다.")

    st.markdown("---")


def render_filter_ui(
    available_dates,
    available_manufacturers,
    available_products,
    lf,
    date_col,
    year_month_expr,
    selected_year_month,
    sidebar_window
):
    """필터 UI 렌더링 (간소화: 사이드바 통합)"""
    st.markdown("### 🔍 데이터 필터")
    st.caption("💡 사이드바에서 기준 월, 윈도우 크기, 상위 개수 등을 설정할 수 있습니다")

    # ==================== 기간 선택 (사이드바 기반 자동 계산) ====================
    with st.expander("📅 분석 기간", expanded=True):
        prev_selected_dates = st.session_state.get('prev_selected_dates', [])

        # 기본값 설정: 사이드바 기준 월 + 윈도우로 자동 계산
        default_dates = []
        if selected_year_month and selected_year_month in available_dates:
            recent_months, base_months = get_window_dates(
                available_dates,
                sidebar_window,
                selected_year_month
            )
            default_dates = list(set(recent_months + base_months))

            # 이전 선택값이 있으면 유지
            if prev_selected_dates:
                valid_prev_dates = [d for d in prev_selected_dates if d in available_dates]
                if valid_prev_dates:
                    default_dates = valid_prev_dates
        elif available_dates:
            # 사이드바 값이 없으면 최근 데이터 기준
            default_dates = [available_dates[0]]

        selected_dates = st.multiselect(
            "분석할 년-월 선택",
            options=available_dates,
            default=default_dates,
            key='dates_multiselect',
            help=f"사이드바 설정(기준월: {selected_year_month}, 윈도우: {sidebar_window}개월)을 기반으로 자동 선택되었습니다."
        )

        if selected_dates:
            st.session_state.prev_selected_dates = selected_dates
            ellipsis = '...' if len(selected_dates) > 3 else ''
            st.info(f"✅ 선택된 기간: {len(selected_dates)}개월 ({', '.join(selected_dates[:3])}{ellipsis})")
        elif 'prev_selected_dates' in st.session_state and not selected_dates:
            del st.session_state.prev_selected_dates

    # ==================== 제조사/제품군 선택 ====================
    with st.expander("🏭 제조사 및 제품군 선택", expanded=True):
        col1, col2 = st.columns(2)

        # 제조사 선택
        with col1:
            if selected_dates:
                filtered_manufacturers = get_manufacturers_by_dates(
                    lf,
                    selected_dates,
                    date_col=date_col,
                    _year_month_expr=year_month_expr
                )
                prev_selected = st.session_state.get('prev_selected_manufacturers', [])
                valid_selected_manufacturers = [m for m in prev_selected if m in filtered_manufacturers]
                manufacturer_options = filtered_manufacturers
                default_manufacturers = valid_selected_manufacturers
            else:
                manufacturer_options = available_manufacturers
                default_manufacturers = []
                if 'prev_selected_manufacturers' in st.session_state:
                    del st.session_state.prev_selected_manufacturers

            selected_manufacturers = st.multiselect(
                "제조사 선택 (선택 안 함 = 전체)",
                options=manufacturer_options,
                default=default_manufacturers,
                help=f"선택된 년-월({len(selected_dates) if selected_dates else 0}개)에 존재하는 제조사만 표시됩니다",
                key='manufacturers_multiselect'
            )

            if selected_manufacturers:
                st.session_state.prev_selected_manufacturers = selected_manufacturers
            else:
                if 'prev_selected_manufacturers' in st.session_state:
                    del st.session_state.prev_selected_manufacturers

        # 제품군 선택
        with col2:
            if selected_manufacturers:
                filtered_products = get_products_by_manufacturers(
                    lf,
                    selected_manufacturers,
                    manufacturer_col=ColumnNames.MANUFACTURER,
                    product_col=ColumnNames.PRODUCT_CODE
                )
                prev_selected = st.session_state.get('prev_selected_products', [])
                valid_selected_products = [p for p in prev_selected if p in filtered_products]
                product_options = filtered_products
                default_products = valid_selected_products
            else:
                product_options = available_products
                default_products = []
                if 'prev_selected_products' in st.session_state:
                    del st.session_state.prev_selected_products

            selected_products = st.multiselect(
                "제품군 선택 (선택 안 함 = 전체)",
                options=product_options,
                default=default_products,
                help=f"선택된 제조사({len(selected_manufacturers) if selected_manufacturers else 0}개)의 제품군만 표시됩니다",
                key='products_multiselect'
            )

            if selected_products:
                st.session_state.prev_selected_products = selected_products
            elif 'prev_selected_products' in st.session_state and not selected_products:
                del st.session_state.prev_selected_products

        # 필터 요약 표시
        if selected_manufacturers or selected_products:
            filter_summary = []
            if selected_manufacturers:
                filter_summary.append(f"제조사 {len(selected_manufacturers)}개")
            if selected_products:
                filter_summary.append(f"제품군 {len(selected_products)}개")
            st.info(f"✅ 선택됨: {', '.join(filter_summary)}")

    st.markdown("---")
    return selected_dates, selected_manufacturers, selected_products


def render_monthly_reports_chart(
    lf,
    date_col,
    selected_dates,
    selected_manufacturers,
    selected_products,
    top_n,
    year_month_expr
):
    """월별 보고서 수 차트 렌더링 (Plotly 인터랙티브 차트)"""
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("📊 월별 보고서 수")

    with st.spinner("데이터 분석 중..."):
        # 데이터 집계
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
            monthly_df = get_monthly_counts(
                lf,
                date_col=date_col,
                selected_dates=selected_dates if selected_dates else None,
                selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                selected_products=selected_products if selected_products else None,
                _year_month_expr=year_month_expr
            )

            if len(monthly_df) > 0:
                monthly_pandas = monthly_df.to_pandas()
                top_combinations = display_df.head(top_n)["제조사-제품군"].tolist()
                chart_data = monthly_pandas[
                    monthly_pandas["manufacturer_product"].isin(top_combinations)
                ].copy()

                # 차트 타입 선택
                chart_type = st.radio(
                    "차트 타입",
                    ["막대 차트", "선 그래프", "영역 차트"],
                    horizontal=True,
                    key="monthly_chart_type"
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
                    file_name=f"monthly_reports_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                    key="download_monthly_reports"
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
    year_month_expr
):
    """제조사-모델별 결함 분석 렌더링"""
    st.subheader("🔧 제조사 - 모델별 결함")

    if not selected_dates:
        st.info("결함 분석을 위해 년-월을 선택해주세요.")
        return

    with st.spinner("결함 분석 중..."):
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
                        text=mfr_data["percentage"].apply(lambda x: f"{x:.1f}%"),
                        textposition='outside',
                        hovertemplate='<b>%{fullData.name}</b><br>결함 유형: %{x}<br>비율: %{y:.1f}%<extra></extra>'
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

                    st.dataframe(top5_display, width='stretch', hide_index=True)

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
                            text=data_a["percentage"].apply(lambda x: f"{x:.1f}%"),
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
                            text=data_b["percentage"].apply(lambda x: f"{x:.1f}%"),
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
                            st.info(f"🔹 **{defect}**: {compare_a}가 {abs(diff):.1f}%p 더 높음")
                        else:
                            st.info(f"🔸 **{defect}**: {compare_b}가 {abs(diff):.1f}%p 더 높음")

                    # 상세 테이블
                    with st.expander("📋 전체 비교 데이터"):
                        st.dataframe(
                            diff_df.style.background_gradient(
                                subset=['차이 (A-B)'],
                                cmap='RdYlGn_r',
                                vmin=-50,
                                vmax=50
                            ),
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
                        "결함 유형": mfr_data[ColumnNames.DEFECT_TYPE],
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

                    st.dataframe(
                        chart_data[["결함 유형", "건수", "비율(%)"]],
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
    year_month_expr
):
    """문제 부품 분석 렌더링"""
    st.subheader("🔩 문제 부품 분석")

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
            col1, col2 = st.columns([2, 1])

            with col1:
                prev_selected_defect_type = st.session_state.get('prev_selected_defect_type', None)
                default_index = 0
                if prev_selected_defect_type and prev_selected_defect_type in available_defect_types:
                    default_index = available_defect_types.index(prev_selected_defect_type)

                selected_defect_type = st.selectbox(
                    "결함 유형 선택",
                    options=available_defect_types,
                    index=default_index,
                    help="분석할 결함 유형을 선택하세요",
                    key='defect_type_selectbox'
                )
                st.session_state.prev_selected_defect_type = selected_defect_type

            with col2:
                default_top_n_components = st.session_state.get('top_n_components', Defaults.TOP_N)
                top_n_components = st.number_input(
                    "상위 N개 표시",
                    min_value=1,
                    max_value=50,
                    value=default_top_n_components,
                    step=1,
                    key='top_n_components_input'
                )
                st.session_state.top_n_components = top_n_components

            if selected_defect_type:
                with st.spinner("문제 부품 분석 중..."):
                    component_df = analyze_defect_components(
                        lf,
                        defect_type=selected_defect_type,
                        date_col=date_col,
                        selected_dates=selected_dates,
                        selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                        selected_products=selected_products if selected_products else None,
                        top_n=top_n_components,
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

                    st.dataframe(
                        display_df,
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
    year_month_expr,
    sidebar_min_cases,
    sidebar_top_n
):
    """기기별 치명률(CFR) 분석 렌더링 (시각화 추가)"""
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("💀 기기별 치명률(CFR) 분석")

    try:
        # 사이드바에서 설정된 값 사용
        top_n_cfr = sidebar_top_n
        min_cases = sidebar_min_cases

        st.caption(f"💡 사이드바 설정: 상위 {top_n_cfr}개 표시, 최소 {min_cases}건 이상")

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
            display_df = cfr_result.to_pandas().copy()

            display_df.insert(0, "순위", range(1, len(display_df) + 1))
            display_df = display_df[[
                "순위", "manufacturer_product", "total_cases",
                "death_count", "injury_count", "malfunction_count",
                "cfr", "injury_rate", "malfunction_rate"
            ]]
            display_df.columns = [
                "순위", "제조사-제품군", "총 건수",
                "사망", "부상", "오작동",
                "CFR(%)", "부상률(%)", "오작동률(%)"
            ]

            # ==================== 요약 통계 (상단 배치) ====================
            st.markdown("### 📊 요약 통계")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:
                st.metric("분석 기기 수", f"{len(display_df):,}개")

            with summary_col2:
                min_cfr = display_df["CFR(%)"].min()
                st.metric("최소 CFR", f"{min_cfr:.2f}%")

            with summary_col3:
                max_cfr = display_df["CFR(%)"].max()
                st.metric("최대 CFR", f"{max_cfr:.2f}%")

            with summary_col4:
                cfr_range = max_cfr - min_cfr
                st.metric("CFR 범위", f"{cfr_range:.2f}%p")

            st.markdown("---")

            # ==================== 시각화 섹션 ====================
            st.markdown("### 📈 CFR 시각화")

            viz_col1, viz_col2 = st.columns(2)

            # 좌측: CFR Top 10 막대 차트
            with viz_col1:
                st.markdown("#### 상위 10개 제조사-제품군 CFR")
                top_10_df = display_df.head(10).copy()

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=top_10_df["CFR(%)"],
                    y=top_10_df["제조사-제품군"],
                    orientation='h',
                    marker=dict(
                        color=top_10_df["CFR(%)"],
                        colorscale='Reds',
                        showscale=False,
                        line=dict(color='rgba(0,0,0,0.2)', width=1)
                    ),
                    text=top_10_df["CFR(%)"].apply(lambda x: f"{x:.2f}%"),
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>CFR: %{x:.2f}%<br>순위: %{customdata}<extra></extra>',
                    customdata=top_10_df["순위"]
                ))

                fig_bar.update_layout(
                    xaxis_title="치명률 (%)",
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

            # 우측: CFR vs 총 건수 산점도
            with viz_col2:
                st.markdown("#### CFR vs 총 건수 (사망 건수 크기)")

                fig_scatter = px.scatter(
                    display_df,
                    x="총 건수",
                    y="CFR(%)",
                    size="사망",
                    color="CFR(%)",
                    color_continuous_scale='Reds',
                    hover_name="제조사-제품군",
                    hover_data={
                        "순위": True,
                        "총 건수": ":,",
                        "CFR(%)": ":.2f",
                        "사망": True,
                        "부상": True,
                        "오작동": True
                    },
                    labels={
                        "총 건수": "총 보고 건수",
                        "CFR(%)": "치명률 (%)",
                        "사망": "사망 건수"
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
                        type='log' if len(display_df) > 0 and display_df["총 건수"].max() > 1000 else 'linear'
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
            st.caption("평균 CFR과의 비교를 통한 통계적 유의성 검정")

            try:
                from utils.statistical_tests import (
                    fisher_exact_test,
                    interpret_significance,
                    calculate_confidence_interval,
                    get_significance_level
                )

                # 전체 평균 CFR 계산
                total_deaths = display_df["사망"].sum()
                total_cases = display_df["총 건수"].sum()
                overall_cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0

                st.info(f"📌 전체 평균 CFR: **{overall_cfr:.2f}%** (사망 {total_deaths:,}건 / 총 {total_cases:,}건)")

                # 통계 검정 결과
                significance_results = []

                for idx, row in display_df.head(10).iterrows():
                    device = row["제조사-제품군"]
                    device_deaths = int(row["사망"])
                    device_total = int(row["총 건수"])
                    device_cfr = row["CFR(%)"]

                    # 나머지 데이터
                    other_deaths = total_deaths - device_deaths
                    other_total = total_cases - device_total

                    if other_total > 0:
                        # Fisher's Exact Test
                        odds_ratio, p_value = fisher_exact_test(
                            device_deaths, device_total,
                            other_deaths, other_total
                        )

                        # 신뢰구간 계산
                        ci_lower, ci_upper = calculate_confidence_interval(device_deaths, device_total)

                        significance_results.append({
                            "제조사-제품군": device,
                            "CFR(%)": device_cfr,
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
                            device = row["제조사-제품군"]
                            cfr = row["CFR(%)"]
                            sig = row["유의성"]
                            interpretation = row["해석"]
                            ci = row["95% CI"]

                            if cfr > overall_cfr:
                                st.error(f"**{device}** {sig}: CFR {cfr:.2f}% (평균보다 높음) - {interpretation}, 95% CI {ci}")
                            else:
                                st.success(f"**{device}** {sig}: CFR {cfr:.2f}% (평균보다 낮음) - {interpretation}, 95% CI {ci}")
                    else:
                        st.info("통계적으로 유의한 차이를 보이는 기기가 없습니다 (α = 0.05)")

                    # 상세 테이블
                    with st.expander("📋 통계 검정 상세 결과"):
                        st.dataframe(
                            sig_df.style.apply(
                                lambda x: ['background-color: #fee' if v < 0.05 else '' for v in x],
                                subset=['p-value']
                            ),
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

            st.dataframe(display_df, width='stretch', hide_index=True)

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
    year_month_expr
):
    """defect type별 상위 문제 & 사건 유형별 분포 렌더링"""
    import plotly.graph_objects as go
    import streamlit.components.v1 as components
    import html

    st.subheader("📊 defect type별 상위 문제 & 사건 유형 분포")

    try:
        # 사용 가능한 defect type 가져오기
        with st.spinner("defect type 목록 로딩 중..."):
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
            # 좌우 레이아웃
            event_col, cluster_col = st.columns([1, 1])

            # 우측: defect type별 상위 문제
            with cluster_col:
                st.markdown("### defect type별 상위 문제")

                # 이전에 선택한 defect type 가져오기
                prev_selected_cluster = st.session_state.get('prev_selected_cluster', None)
                default_index = 0
                if prev_selected_cluster and prev_selected_cluster in available_clusters:
                    default_index = available_clusters.index(prev_selected_cluster)

                selected_cluster = st.selectbox(
                    "카테고리 선택",
                    options=available_clusters,
                    index=default_index,
                    help="분석할 defect type를 선택하세요",
                    key='cluster_selectbox',
                    label_visibility="collapsed"
                )
                st.session_state.prev_selected_cluster = selected_cluster

                # 상위 N개 설정 (기본값 10개)
                top_n_cluster = 10

                # defect type별 상위 문제 분석 실행
                if selected_cluster:
                    with st.spinner("defect type별 상위 문제 분석 중..."):
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

                    # 선택된 defect type의 데이터만 필터링
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
                                    <span class="bar-ratio">{ratio:.1f}%</span>
                                </div>
                            </div>
                            """

                        html_content += "</div>"

                        # HTML 렌더링 (components.html 사용)
                        components.html(html_content, height=container_height + 20, scrolling=True)
                    else:
                        st.info(f"'{selected_cluster}' defect type에 대한 문제 부품 데이터가 없습니다.")

            # 좌측: 환자 피해 분포 파이 차트
            with event_col:
                st.markdown("### 환자 피해 분포")

                with st.spinner("환자 피해 데이터 로딩 중..."):
                    harm_summary = get_patient_harm_summary(
                        lf,
                        event_column=ColumnNames.PATIENT_HARM,
                        date_col=date_col,
                        selected_dates=selected_dates if selected_dates else None,
                        selected_manufacturers=selected_manufacturers if selected_manufacturers else None,
                        selected_products=selected_products if selected_products else None,
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
                        st.metric("사망", f"{total_deaths:,}건")

                    with summary_col2:
                        st.metric("중증 부상", f"{total_serious:,}건")

                    with summary_col3:
                        st.metric("경증 부상", f"{total_minor:,}건")

                    with summary_col4:
                        st.metric("부상 없음", f"{total_none:,}건")

                    with summary_col5:
                        st.metric("Unknown", f"{total_unknown:,}건")
                else:
                    st.info("환자 피해 데이터가 없습니다.")
        else:
            st.info("선택한 조건에 해당하는 defect type가 없습니다.")

    except Exception as e:
        st.error(f"defect type별 상위 문제 분석 중 오류가 발생했습니다: {str(e)}")
        st.exception(e)

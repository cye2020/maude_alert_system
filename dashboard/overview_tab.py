# overview_tab.py
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.analysis import calculate_big_numbers, get_risk_matrix_data
from utils.constants import ColumnNames, PatientHarmLevels, Defaults, DisplayNames, Terms
from dashboard.utils.ui_components import render_filter_summary_badge

def plot_sparkline(data_list, key="sparkline"):
    """Sparkline 미니 차트 생성

    Args:
        data_list: 시계열 데이터 리스트 (최근 6개월)
        key: Streamlit 차트 고유 키
    """
    if not data_list or len(data_list) == 0:
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=data_list,
            mode='lines',
            line=dict(color='#1f77b4', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)',
            showlegend=False,
            hovertemplate='%{y:.2f}<extra></extra>'
        )
    )

    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, width='stretch', key=key)


def plot_dual_axis_chart(
        data: pl.LazyFrame,
        start: str = None,
        end: str = None
    ):
    """Dual-Axis 차트: Report Count (막대) + Severe Harm Rate (라인)

    Args:
        data: LazyFrame 데이터 (이미 공통 필터 적용됨)
        start: 시작 날짜 (예: "2024-01-01"), None이면 전체 기간
        end: 종료 날짜 (예: "2024-12-31"), None이면 전체 기간
    """
    # 1. 날짜 필터링
    filtered_data = data

    # 날짜 필터 적용
    if start and end:
        from datetime import datetime
        start_dt = datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
        end_dt = datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end

        filtered_data = filtered_data.filter(
            (pl.col("date_received") >= start_dt) & (pl.col("date_received") <= end_dt)
        )

    # 2. 월별 집계 (총 count + severe harm count)
    agg_data = (
        filtered_data
        .group_by(pl.col("date_received").dt.truncate("1mo").alias("date"))
        .agg([
            pl.len().alias("count"),
            pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
              .then(1).otherwise(0).sum().alias("severe_harm_count")
        ])
        .with_columns(
            (pl.col("severe_harm_count") / pl.col("count") * 100).alias("severe_harm_rate")
        )
        .sort("date")
        .collect()
    )

    # 3. Dual-Axis 차트 생성
    st.subheader(f"📈 {Terms.KOREAN.REPORT_COUNT} 및 {Terms.KOREAN.SEVERE_HARM_RATE} {Terms.KOREAN.TREND}")

    # subplots 사용하여 이중 축 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 왼쪽 축: Report Count (막대)
    fig.add_trace(
        go.Bar(
            x=agg_data["date"],
            y=agg_data["count"],
            name=Terms.KOREAN.REPORT_COUNT,
            marker_color='rgba(31, 119, 180, 0.6)',
            yaxis='y'
        ),
        secondary_y=False
    )

    # 오른쪽 축: Severe Harm Rate (라인)
    fig.add_trace(
        go.Scatter(
            x=agg_data["date"],
            y=agg_data["severe_harm_rate"],
            name=f"{Terms.KOREAN.SEVERE_HARM_RATE} (%)",
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            yaxis='y2'
        ),
        secondary_y=True
    )

    # 레이아웃 설정
    fig.update_layout(
        height=500,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=40, b=80),
        xaxis=dict(
            title="날짜",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick="M1",
            tickformat="%Y-%m"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Y축 제목 설정
    fig.update_yaxes(title_text="보고 건수", secondary_y=False)
    fig.update_yaxes(title_text="중대 피해율 (%)", secondary_y=True)

    st.plotly_chart(fig, width='stretch', key='dual_axis_chart')


def plot_risk_matrix(
        data: pl.LazyFrame,
        start: str = None,
        end: str = None,
        view_mode: str = "defect_type",
        top_n: int = 20,
        manufacturers: list = None,
        products: list = None
    ):
    """Risk Matrix: 발생 빈도 vs 치명률

    Args:
        data: LazyFrame 데이터 (이미 공통 필터 적용됨)
        start: 시작 날짜
        end: 종료 날짜
        view_mode: "defect_type", "manufacturer", "product"
        top_n: 상위 N개
        manufacturers: 제조사 필터 (캐시 키용)
        products: 제품군 필터 (캐시 키용)
    """
    from datetime import datetime

    # 날짜 변환
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start and isinstance(start, str) else start
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end and isinstance(end, str) else end

    # 데이터 가져오기
    risk_data = get_risk_matrix_data(
        _lf=data,
        start_date=start_dt,
        end_date=end_dt,
        view_mode=view_mode,
        top_n=top_n,
        manufacturers=tuple(manufacturers) if manufacturers else (),
        products=tuple(products) if products else ()
    )

    if len(risk_data) == 0:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    df = risk_data.to_pandas()

    # 제목 설정
    view_titles = {
        "defect_type": "결함 유형별 리스크",
        "manufacturer": "제조사별 리스크",
        "product": "제품군별 리스크"
    }
    title = view_titles.get(view_mode, "리스크 매트릭스")

    st.subheader(f"📍 {title}")

    # 사분면 경계선 계산 (중앙값)
    median_count = df["report_count"].median()
    median_rate = df["severe_harm_rate"].median()

    # Scatter Plot 생성
    fig = go.Figure()

    # 사분면 배경 추가
    fig.add_shape(type="rect", x0=0, y0=0, x1=median_count, y1=median_rate,
                  fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=median_count, y0=0, x1=df["report_count"].max() * 1.1, y1=median_rate,
                  fillcolor="yellow", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=median_rate, x1=median_count, y1=df["severe_harm_rate"].max() * 1.1,
                  fillcolor="orange", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=median_count, y0=median_rate,
                  x1=df["report_count"].max() * 1.1, y1=df["severe_harm_rate"].max() * 1.1,
                  fillcolor="salmon", opacity=0.2, layer="below", line_width=0)

    # 데이터 포인트
    fig.add_trace(
        go.Scatter(
            x=df["report_count"],
            y=df["severe_harm_rate"],
            mode='markers+text',
            marker=dict(
                size=df["defect_confirmed_rate"] * 2,  # 크기: 결함 확정률
                color=df["severe_harm_rate"],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="치명률 (%)", thickness=15),
                line=dict(width=1, color='white')
            ),
            text=df["entity"],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                '발생 건수: %{x:,}<br>' +
                '치명률: %{y:.2f}%<br>' +
                '결함 확정률: %{marker.size:.2f}%<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        )
    )

    # 사분면 경계선
    fig.add_hline(y=median_rate, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_count, line_dash="dash", line_color="gray", opacity=0.5)

    # 사분면 레이블
    max_x = df["report_count"].max() * 1.05
    max_y = df["severe_harm_rate"].max() * 1.05

    annotations = [
        dict(x=median_count/2, y=max_y*0.95, text="저빈도<br>고위험", showarrow=False, font=dict(size=12, color="gray")),
        dict(x=max_x*0.9, y=max_y*0.95, text="고빈도<br>고위험", showarrow=False, font=dict(size=12, color="red")),
        dict(x=median_count/2, y=median_rate/2, text="저빈도<br>저위험", showarrow=False, font=dict(size=12, color="gray")),
        dict(x=max_x*0.9, y=median_rate/2, text="고빈도<br>저위험", showarrow=False, font=dict(size=12, color="gray"))
    ]

    fig.update_layout(
        height=600,
        xaxis=dict(title="발생 빈도 (건)", showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
        yaxis=dict(title="치명률 (%)", showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
        hovermode='closest',
        annotations=annotations,
        margin=dict(l=50, r=50, t=40, b=50)
    )

    st.plotly_chart(fig, width='stretch', key='risk_matrix_chart')


# overview_tab.py
def show(filters=None, lf: pl.LazyFrame = None):
    from utils.constants import DisplayNames

    st.title(DisplayNames.FULL_TITLE_OVERVIEW)

    # 날짜 범위 가져오기 (month_range_picker에서)
    date_range = filters.get("date_range", None)
    start_date = None
    end_date = None

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range

    # 공통 필터 가져오기
    selected_manufacturers = filters.get("manufacturers", [])
    selected_products = filters.get("products", [])
    selected_devices = filters.get("devices", [])
    selected_defect_types = filters.get("defect_types", [])
    selected_clusters = filters.get("clusters", [])

    # 세션 스테이트 초기화 (브러시 선택된 날짜 범위 저장)
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = None

    # ==================== 필터 요약 배지 (공통 함수 사용) ====================
    render_filter_summary_badge(
        date_range=date_range,
        manufacturers=selected_manufacturers,
        products=selected_products,
        devices=selected_devices,
        defect_types=selected_defect_types,
        clusters=selected_clusters
    )
    st.markdown("---")

    # 공통 필터 적용
    from dashboard.utils.filter_helpers import apply_common_filters
    filtered_lf = apply_common_filters(
        lf,
        manufacturers=selected_manufacturers,
        products=selected_products,
        devices=selected_devices,
        defect_types=selected_defect_types,
        clusters=selected_clusters
    )

    # Big Number 표시 (4개) - 선택된 기간의 최신 한 달 vs 전월 비교
    big_numbers = calculate_big_numbers(
        _data=filtered_lf,
        start_date=start_date,
        end_date=end_date,
        manufacturers=tuple(selected_manufacturers) if selected_manufacturers else (),
        products=tuple(selected_products) if selected_products else ()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📁 총 보고서 수",
            value=f"{big_numbers['total_reports']:,}건",
            delta=f"{big_numbers['total_reports_delta']:+.2f}%" if big_numbers['total_reports_delta'] is not None else None
        )
        # Sparkline 추가
        plot_sparkline(big_numbers['total_reports_sparkline'], key="sparkline_total")

    with col2:
        # delta에 이전 기간의 가장 치명적인 defect type 표시
        prev_defect_info = f"이전: {big_numbers['prev_most_critical_defect_type']} ({big_numbers['prev_most_critical_defect_rate']:.2f}%)"
        st.metric(
            label="🔥 가장 치명적인 Defect Type",
            value=big_numbers['most_critical_defect_type'],
            delta=prev_defect_info,
            delta_arrow='off',
            delta_color="off"  # delta를 회색으로 표시 (증감이 아니라 정보)
        )

    with col3:
        st.metric(
            label="⚠️ 중대 피해 발생률",
            value=f"{big_numbers['severe_harm_rate']:.2f}%",
            delta=f"{big_numbers['severe_harm_rate_delta']:+.2f}%p" if big_numbers['severe_harm_rate_delta'] is not None else None
        )
        # Sparkline 추가
        plot_sparkline(big_numbers['severe_harm_sparkline'], key="sparkline_harm")

    with col4:
        st.metric(
            label="🔧 제조사 결함 확정률",
            value=f"{big_numbers['defect_confirmed_rate']:.2f}%",
            delta=f"{big_numbers['defect_confirmed_rate_delta']:+.2f}%p" if big_numbers['defect_confirmed_rate_delta'] is not None else None
        )
        # Sparkline 추가
        plot_sparkline(big_numbers['defect_sparkline'], key="sparkline_defect")

    st.markdown("---")

    # 차트 그리기 (날짜 범위 적용)
    start_str = start_date.strftime("%Y-%m-%d") if start_date else None
    end_str = end_date.strftime("%Y-%m-%d") if end_date else None

    # Dual-Axis 차트 추가 (공통 필터 적용된 데이터 사용)
    plot_dual_axis_chart(filtered_lf, start=start_str, end=end_str)

    st.markdown("---")

    # Risk Matrix Analysis
    st.subheader(f"🔍 {Terms.KOREAN.RISK_MATRIX}")

    # 설명 추가
    with st.expander(f"ℹ️ {Terms.KOREAN.RISK_MATRIX}란?", expanded=False):
        st.markdown(f"""
        **{Terms.KOREAN.RISK_MATRIX}**는 발생 빈도와 {Terms.KOREAN.CFR}을 동시에 고려하여 위험도를 평가하는 도구입니다.

        **해석 방법**:
        - **오른쪽 위**: 빈도 높음 + 치명률 높음 = **최고 위험**
        - **오른쪽 아래**: 빈도 높음 + 치명률 낮음 = 모니터링 필요
        - **왼쪽 위**: 빈도 낮음 + 치명률 높음 = 발생 시 심각
        - **왼쪽 아래**: 빈도 낮음 + 치명률 낮음 = 낮은 위험

        **인사이트**:
        - 버블 크기가 클수록 해당 항목의 총 보고 건수가 많습니다
        - 오른쪽 위 사분면의 항목들에 우선적으로 조치가 필요합니다
        """)

    # Risk Matrix View Mode 선택
    risk_col1, risk_col2 = st.columns([3, 1])

    with risk_col1:
        st.markdown("") # 간격

    with risk_col2:
        view_mode = st.selectbox(
            "분석 단위",
            options=["결함 유형", "제조사", "제품군"],
            index=0,
            key="risk_view_mode"
        )

        view_mode_map = {
            "결함 유형": "defect_type",
            "제조사": "manufacturer",
            "제품군": "product"
        }

        selected_view_mode = view_mode_map[view_mode]

    plot_risk_matrix(
        data=filtered_lf,
        start=start_str,
        end=end_str,
        view_mode=selected_view_mode,
        top_n=20,
        manufacturers=selected_manufacturers,
        products=selected_products
    )


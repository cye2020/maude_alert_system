# overview_tab.py
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.analysis import calculate_big_numbers, get_risk_matrix_data
from utils.constants import ColumnNames, PatientHarmLevels, Defaults

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
            hovertemplate='%{y:.1f}<extra></extra>'
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
        end: str = None,
        segment: str = None,
        segment_value: str = None
    ):
    """Dual-Axis 차트: Report Count (막대) + Severe Harm Rate (라인)

    Args:
        data: LazyFrame 데이터
        start: 시작 날짜 (예: "2024-01-01"), None이면 전체 기간
        end: 종료 날짜 (예: "2024-12-31"), None이면 전체 기간
        segment: 세그먼트 컬럼명 (필터링할 컬럼)
        segment_value: 세그먼트 값 (특정 값으로 필터링)
    """
    # 1. 필터링 (날짜 + 세그먼트)
    filtered_data = data

    # Segment 필터 적용
    if segment and segment_value:
        filtered_data = filtered_data.filter(pl.col(segment) == segment_value)

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
    st.subheader("📊 Report Count & Severe Harm Rate (Dual-Axis)")

    # subplots 사용하여 이중 축 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 왼쪽 축: Report Count (막대)
    fig.add_trace(
        go.Bar(
            x=agg_data["date"],
            y=agg_data["count"],
            name="Report Count",
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
            name="Severe Harm Rate (%)",
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
            title="Date",
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
    fig.update_yaxes(title_text="Report Count", secondary_y=False)
    fig.update_yaxes(title_text="Severe Harm Rate (%)", secondary_y=True)

    st.plotly_chart(fig, width='stretch', key='dual_axis_chart')


def plot_risk_matrix(
        data: pl.LazyFrame,
        start: str = None,
        end: str = None,
        segment_col: str = None,
        segment_value: str = None,
        view_mode: str = "defect_type",
        top_n: int = 20
    ):
    """Risk Matrix: 발생 빈도 vs 치명도

    Args:
        data: LazyFrame 데이터
        start: 시작 날짜
        end: 종료 날짜
        segment_col: 세그먼트 컬럼명
        segment_value: 세그먼트 값
        view_mode: "defect_type", "manufacturer", "product"
        top_n: 상위 N개
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
        segment_col=segment_col,
        segment_value=segment_value,
        view_mode=view_mode,
        top_n=top_n
    )

    if len(risk_data) == 0:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    df = risk_data.to_pandas()

    # 제목 설정
    view_titles = {
        "defect_type": "Defect Type별 리스크",
        "manufacturer": "제조사별 리스크",
        "product": "제품군별 리스크"
    }
    title = view_titles.get(view_mode, "리스크 매트릭스")
    if segment_value:
        title = f"{segment_value} - {title}"

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
                '치명률: %{y:.1f}%<br>' +
                '결함 확정률: %{marker.size:.1f}%<br>' +
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
        xaxis=dict(title="Report Count (발생 빈도)", showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
        yaxis=dict(title="Severe Harm Rate (%) (치명도)", showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
        hovermode='closest',
        annotations=annotations,
        margin=dict(l=50, r=50, t=40, b=50)
    )

    st.plotly_chart(fig, width='stretch', key='risk_matrix_chart')


# overview_tab.py
def show(filters=None, lf: pl.LazyFrame = None):
    st.title("📊 Overview")

    # 필터에서 segment 값 가져오기 (None이면 전체)
    segment = filters.get("segment", None)

    # 날짜 범위 가져오기 (month_range_picker에서)
    date_range = filters.get("date_range", None)
    start_date = None
    end_date = None

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range

    # 세션 스테이트 초기화 (브러시 선택된 날짜 범위 저장)
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = None

    # 특정 값으로 드릴다운 필터 (Sidebar에서 선택한 segment 기준)
    segment_col = None
    segment_value = None

    if segment:  # segment가 None이 아닌 경우 (전체가 아닌 경우)
        with st.expander("🎯 특정 값 선택 (선택 사항)", expanded=False):
            st.info(f"필터를 적용하지 않으면 모든 {segment}를 분석합니다.")

            # Sidebar의 segment 값을 column name으로 사용
            segment_col = segment

            # 해당 컬럼의 고유값 가져오기
            unique_values = lf.select(segment_col).unique().sort(segment_col).collect()[segment_col].to_list()

            # None 제거 (있을 경우)
            unique_values = [v for v in unique_values if v is not None]

            # 선택 UI
            filter_options = ["전체"] + unique_values
            selected = st.selectbox(
                f"{segment} 선택",
                options=filter_options,
                index=0,
                key="segment_value_selector"
            )

            # "전체"가 아닌 경우에만 segment_value 설정
            if selected != "전체":
                segment_value = selected

    # Big Number 표시 (4개) - 선택된 기간의 최신 한 달 vs 전월 비교
    big_numbers = calculate_big_numbers(
        _data=lf,
        segment=segment,
        segment_value=segment_value,
        start_date=start_date,
        end_date=end_date
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📁 총 보고서 수",
            value=f"{big_numbers['total_reports']:,}건",
            delta=f"{big_numbers['total_reports_delta']:+.1f}%" if big_numbers['total_reports_delta'] is not None else None
        )
        # Sparkline 추가
        plot_sparkline(big_numbers['total_reports_sparkline'], key="sparkline_total")

    with col2:
        # delta에 이전 기간의 가장 치명적인 defect type 표시
        prev_defect_info = f"이전: {big_numbers['prev_most_critical_defect_type']} ({big_numbers['prev_most_critical_defect_rate']:.1f}%)"
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
            value=f"{big_numbers['severe_harm_rate']:.1f}%",
            delta=f"{big_numbers['severe_harm_rate_delta']:+.1f}%p" if big_numbers['severe_harm_rate_delta'] is not None else None
        )
        # Sparkline 추가
        plot_sparkline(big_numbers['severe_harm_sparkline'], key="sparkline_harm")

    with col4:
        st.metric(
            label="🔧 제조사 결함 확정률",
            value=f"{big_numbers['defect_confirmed_rate']:.1f}%",
            delta=f"{big_numbers['defect_confirmed_rate_delta']:+.1f}%p" if big_numbers['defect_confirmed_rate_delta'] is not None else None
        )
        # Sparkline 추가
        plot_sparkline(big_numbers['defect_sparkline'], key="sparkline_defect")

    st.markdown("---")

    # 차트 그리기 (날짜 범위 적용)
    start_str = start_date.strftime("%Y-%m-%d") if start_date else None
    end_str = end_date.strftime("%Y-%m-%d") if end_date else None

    # Dual-Axis 차트 추가
    plot_dual_axis_chart(lf, start=start_str, end=end_str, segment=segment, segment_value=segment_value)

    st.markdown("---")

    # Risk Matrix Analysis
    st.header("🔍 산업 분석 (Industry Analysis)")

    # Risk Matrix
    st.markdown("---")

    # Risk Matrix View Mode 선택
    risk_col1, risk_col2 = st.columns([3, 1])

    with risk_col1:
        st.markdown("") # 간격

    with risk_col2:
        view_mode = st.selectbox(
            "분석 단위",
            options=["Defect Type", "Manufacturer", "Product"],
            index=0,
            key="risk_view_mode"
        )

        view_mode_map = {
            "Defect Type": "defect_type",
            "Manufacturer": "manufacturer",
            "Product": "product"
        }

        selected_view_mode = view_mode_map[view_mode]

    plot_risk_matrix(
        data=lf,
        start=start_str,
        end=end_str,
        segment_col=segment_col,
        segment_value=segment_value,
        view_mode=selected_view_mode,
        top_n=20
    )

    st.markdown("---")

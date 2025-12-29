"""Plotly 차트 생성 유틸리티"""

import polars as pl
import plotly.graph_objects as go


def draw_donut_chart(count_df: pl.DataFrame, col: str, top_n: int = 5) -> None:
    """데이터프레임을 기반으로 도넛 차트 생성

    상위 N개 항목을 표시하고 나머지는 'Minor Group'으로 묶어서 표시

    Args:
        count_df (pl.DataFrame): 'count'와 'percentage' 컬럼을 포함한 DataFrame
        col (str): 레이블로 사용할 컬럼명
        top_n (int, optional): 개별 표시할 상위 항목 개수. Defaults to 5.

    Returns:
        None: Plotly 차트를 화면에 표시

    Examples:
        >>> count_df = pl.DataFrame({
        ...     'category': ['A', 'B', 'C', 'D', 'E', 'F'],
        ...     'count': [100, 80, 60, 40, 20, 10]
        ... })
        >>> draw_donut_chart(count_df, 'category', top_n=3)
    """
    # top_n이 지정되고 데이터가 그보다 많으면 처리
    if top_n and len(count_df) > top_n:
        # 상위 N개 추출
        top = count_df.head(top_n)

        # 나머지 합계 계산
        rest_sum = count_df[top_n:].select(pl.col('count').sum()).item()

        if rest_sum > 0:
            # 나머지 비율 계산
            rest_percentage = round(rest_sum / count_df.select(pl.col('count').sum()).item() * 100, 2)

            # 'Minor Group' 행 생성
            other_row = pl.DataFrame({
                col: ['Minor Group'],
                'count': [rest_sum],
                'percentage': [rest_percentage]
            }).with_columns(
                pl.col('count').cast(count_df['count'].dtype)  # 타입 맞추기
            )

            # 상위 N개와 Minor Group 합치기
            count_df = pl.concat([top, other_row])
        else:
            count_df = top

    # Plotly 도넛 차트 생성
    fig = go.Figure(data=[go.Pie(
        labels=count_df[col],
        values=count_df['count'],
        hole=.4,  # 가운데 구멍 크기 (도넛 모양)
        hoverinfo="label+percent",  # 호버 시 표시 정보
        textinfo='label+percent',  # 차트에 표시할 정보
        textposition='outside'  # 텍스트 위치
    )])

    # 컬럼명을 제목으로 변환 (중간에 줄바꿈 추가)
    title = col.title().split('_')
    text = ' '.join(title[:len(title)//2]) + '<br>' + ' '.join(title[len(title)//2:])

    # 레이아웃 설정
    fig.update_layout(
        title_text=f"{col.title()} Distribution",
        annotations=[dict(
            text=text,  # 도넛 중앙에 표시할 텍스트
            x=0.5,
            y=0.5,
            font_size=20,
            showarrow=False
        )]
    )

    fig.show()

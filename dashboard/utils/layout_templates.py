# layout_templates.py
"""
대시보드 표준 레이아웃 템플릿
모든 탭에서 일관된 구조를 유지하기 위한 템플릿 함수들
"""

import streamlit as st
from typing import List, Dict, Any, Callable, Optional
from dashboard.utils.ui_components import render_section_header
from dashboard.utils.constants import DisplayNames


# ==================== 표준 레이아웃 패턴 ====================

class StandardLayout:
    """표준 대시보드 레이아웃 클래스

    모든 탭이 동일한 구조를 따르도록 하는 표준 레이아웃:
    1. 제목
    2. 필터 요약
    3. 핵심 메트릭 (4개)
    4. 구분선
    5. 주요 시각화
    6. 구분선
    7. 상세 분석 (탭 또는 섹션)
    8. 데이터 테이블 + 다운로드
    """

    def __init__(self, title: str):
        """
        Args:
            title: 페이지 제목
        """
        self.title = title
        self._sections = []

    def render_title(self):
        """제목 렌더링"""
        st.title(self.title)

    def render_filter_summary(self, render_func: Callable):
        """필터 요약 렌더링

        Args:
            render_func: 필터 요약 렌더링 함수
        """
        render_func()
        st.markdown("---")

    def render_metrics(self, metrics: List[Dict[str, Any]], columns: int = 4):
        """핵심 메트릭 렌더링

        Args:
            metrics: 메트릭 딕셔너리 리스트
                - label: 라벨
                - value: 값
                - delta: 변화량 (선택)
                - delta_color: 델타 색상 (선택)
                - help: 도움말 (선택)
            columns: 컬럼 수
        """
        cols = st.columns(columns)

        for i, metric in enumerate(metrics[:columns]):
            with cols[i]:
                st.metric(
                    label=metric.get("label", ""),
                    value=metric.get("value", "N/A"),
                    delta=metric.get("delta"),
                    delta_color=metric.get("delta_color", "normal"),
                    help=metric.get("help")
                )

        st.markdown("---")

    def add_section(
        self,
        title: str,
        render_func: Callable,
        icon: str = "",
        divider: bool = True
    ):
        """섹션 추가

        Args:
            title: 섹션 제목
            render_func: 섹션 내용을 렌더링하는 함수
            icon: 아이콘 이모지
            divider: 구분선 표시 여부
        """
        self._sections.append({
            'title': title,
            'render_func': render_func,
            'icon': icon,
            'divider': divider
        })

    def render_sections(self):
        """등록된 모든 섹션 렌더링"""
        for section in self._sections:
            # 섹션 제목
            render_section_header(
                title=section['title'],
                icon=section['icon'],
                divider=False
            )

            # 섹션 내용
            section['render_func']()

            # 구분선
            if section['divider']:
                st.markdown("---")

    def render_data_table(
        self,
        data,
        title: str = None,
        download_button: bool = True,
        download_filename: str = "data"
    ):
        """데이터 테이블 + 다운로드 렌더링

        Args:
            data: 데이터프레임
            title: 테이블 제목
            download_button: 다운로드 버튼 표시 여부
            download_filename: 다운로드 파일명 접두사
        """
        if title:
            st.subheader(title)

        if data is not None and len(data) > 0:
            st.dataframe(data, width='stretch', height=600)

            if download_button:
                from dashboard.utils.ui_components import render_download_button
                st.markdown("---")
                render_download_button(
                    data=data,
                    filename_prefix=download_filename,
                    key=f"download_{download_filename}"
                )
        else:
            st.info(DisplayNames.NO_DATA)


# ==================== 특정 패턴 헬퍼 함수 ====================

def render_two_column_layout(
    left_content: Callable,
    right_content: Callable,
    left_title: str = None,
    right_title: str = None,
    ratio: List[int] = [1, 1]
):
    """2컬럼 레이아웃 렌더링

    Args:
        left_content: 왼쪽 컬럼 렌더링 함수
        right_content: 오른쪽 컬럼 렌더링 함수
        left_title: 왼쪽 제목
        right_title: 오른쪽 제목
        ratio: 컬럼 비율 [왼쪽, 오른쪽]

    Example:
        >>> def render_left():
        ...     st.write("왼쪽 내용")
        >>> def render_right():
        ...     st.write("오른쪽 내용")
        >>> render_two_column_layout(render_left, render_right, ratio=[2, 1])
    """
    col_left, col_right = st.columns(ratio)

    with col_left:
        if left_title:
            st.markdown(f"#### {left_title}")
        left_content()

    with col_right:
        if right_title:
            st.markdown(f"#### {right_title}")
        right_content()


def render_tabbed_content(
    tabs: List[Dict[str, Any]]
):
    """탭 기반 컨텐츠 렌더링

    Args:
        tabs: 탭 딕셔너리 리스트
            - label: 탭 라벨
            - render_func: 탭 내용 렌더링 함수

    Example:
        >>> tabs = [
        ...     {"label": "📊 차트", "render_func": render_chart},
        ...     {"label": "📋 테이블", "render_func": render_table}
        ... ]
        >>> render_tabbed_content(tabs)
    """
    if not tabs:
        return

    # 탭 생성
    tab_labels = [tab['label'] for tab in tabs]
    tab_objects = st.tabs(tab_labels)

    # 각 탭 렌더링
    for i, (tab_obj, tab_info) in enumerate(zip(tab_objects, tabs)):
        with tab_obj:
            tab_info['render_func']()


def render_expandable_section(
    title: str,
    render_func: Callable,
    expanded: bool = False,
    icon: str = ""
):
    """확장 가능한 섹션 렌더링

    Args:
        title: 섹션 제목
        render_func: 내용 렌더링 함수
        expanded: 기본 확장 여부
        icon: 아이콘

    Example:
        >>> def render_details():
        ...     st.write("상세 내용")
        >>> render_expandable_section("📋 상세 정보", render_details)
    """
    full_title = f"{icon} {title}" if icon else title

    with st.expander(full_title, expanded=expanded):
        render_func()


# ==================== 공통 섹션 템플릿 ====================

def render_insights_section(insights: List[str], title: str = "💡 인사이트"):
    """인사이트 섹션 렌더링

    Args:
        insights: 인사이트 문자열 리스트
        title: 섹션 제목

    Example:
        >>> insights = [
        ...     "클러스터 3에서 높은 사망률 발견",
        ...     "제조사 A의 제품에서 반복적인 결함 발생"
        ... ]
        >>> render_insights_section(insights)
    """
    st.subheader(title)

    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("분석 결과에서 특별한 인사이트가 발견되지 않았습니다.")


def render_summary_cards(
    summaries: List[Dict[str, Any]],
    columns: int = 3
):
    """요약 카드 렌더링

    Args:
        summaries: 요약 딕셔너리 리스트
            - title: 카드 제목
            - content: 카드 내용 (마크다운)
            - color: 카드 색상 ("info", "success", "warning", "error")
        columns: 컬럼 수

    Example:
        >>> summaries = [
        ...     {"title": "총 보고 건수", "content": "1,234건", "color": "info"},
        ...     {"title": "주요 발견", "content": "클러스터 5 주의 필요", "color": "warning"}
        ... ]
        >>> render_summary_cards(summaries, columns=2)
    """
    cols = st.columns(columns)

    color_func_map = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error
    }

    for i, summary in enumerate(summaries):
        with cols[i % columns]:
            st.markdown(f"**{summary.get('title', '')}**")

            color = summary.get('color', 'info')
            func = color_func_map.get(color, st.info)

            func(summary.get('content', ''))


# ==================== 메트릭 헬퍼 ====================

def create_metric_dict(
    label: str,
    value: Any,
    delta: Optional[Any] = None,
    delta_color: str = "normal",
    help: Optional[str] = None
) -> Dict[str, Any]:
    """메트릭 딕셔너리 생성 헬퍼

    Args:
        label: 메트릭 라벨
        value: 메트릭 값
        delta: 변화량
        delta_color: 델타 색상
        help: 도움말

    Returns:
        메트릭 딕셔너리

    Example:
        >>> metric = create_metric_dict(
        ...     label="총 보고 건수",
        ...     value="1,234건",
        ...     delta="+10%"
        ... )
    """
    return {
        "label": label,
        "value": value,
        "delta": delta,
        "delta_color": delta_color,
        "help": help
    }

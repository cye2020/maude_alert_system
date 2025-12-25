# filter_manager.py
from datetime import datetime
from typing import Any, Dict, Optional
import streamlit as st

import sys
from pathlib import Path
# 프로젝트 루트 경로 설정
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from dashboard.utils.dashboard_config import get_config


class SidebarManager:
    """sidebar.yaml 설정을 기반으로 Streamlit 사이드바를 관리하는 클래스"""

    def __init__(self, dashboard_type: str = "overview"):
        """
        Args:
            dashboard_type: 대시보드 타입 ("overview", "eda", "cluster")
        """
        self.cfg = get_config()
        self.dashboard_type = dashboard_type
        self.TODAY = datetime.now()

        # 공통 설정과 대시보드별 설정 로드
        self.common_config = self.cfg.sidebar.get("common", {})
        self.dashboard_config = self.cfg.sidebar.get("dashboards", {}).get(dashboard_type, {})

    # ==================== 공통 컴포넌트 ====================

    def render_header(self):
        """프로젝트 로고 및 정보 렌더링"""
        header_config = self.common_config.get("header", {})

        # 로고
        logo_path = header_config.get("logo")
        if logo_path:
            st.image(logo_path, width=200)

        # 프로젝트 정보
        project_info = header_config.get("project_info", {})
        if project_info:
            st.markdown("### 📌 프로젝트 정보")
            st.info(f"""
            **버전**: {project_info.get('version', 'N/A')}
            **업데이트**: {project_info.get('update_date', 'N/A')}
            **환경**: {project_info.get('environment', 'N/A')}
            """)
            st.markdown("---")

    def render_date_selector(self) -> Optional[datetime]:
        """날짜 선택기 렌더링 (공통 필터 - 탭 전환 시에도 값 유지)

        Returns:
            선택된 날짜 (datetime 객체) 또는 None
        """
        date_config = self.common_config.get("date_selector", {})

        if not date_config.get("enabled", False):
            return None

        # 년도 범위 계산
        year_range = date_config.get("year_range", [-2, 0])
        year_options = range(
            self.TODAY.year + year_range[0],
            self.TODAY.year + year_range[1] + 1
        )
        default_year_index = date_config.get("default_year_index", 0)
        default_month = date_config.get("default_month", 1)

        # 날짜 선택 UI (공통 key 사용으로 탭 전환 시에도 값 유지)
        with st.container():
            st.markdown("### 📅 기준 날짜")
            col1, col2 = st.columns(2)

            with col1:
                year = st.selectbox(
                    "년도",
                    options=list(year_options),
                    index=min(default_year_index, len(list(year_options)) - 1),
                    format_func=lambda x: f"{x}년",
                    key="common_year"  # 공통 key로 모든 탭에서 값 유지
                )

            with col2:
                month = st.selectbox(
                    "월",
                    options=range(1, 13),
                    index=default_month - 1,
                    format_func=lambda x: f"{x:02d}월",
                    key="common_month"  # 공통 key로 모든 탭에서 값 유지
                )

        selected_date = datetime(year, month, 1)
        st.caption(f"선택: {selected_date.strftime('%Y년 %m월')}")
        st.markdown("---")

        return selected_date

    # ==================== Overview 대시보드 전용 ====================

    def render_window_selector(self) -> Optional[int]:
        """관측 기간 선택기 렌더링 (Overview 전용)

        Returns:
            선택된 관측 기간(개월) 또는 None
        """
        window_config = self.dashboard_config.get("window_selector", {})

        if not window_config.get("enabled", False):
            return None

        options = window_config.get("options", [1, 3])
        default = window_config.get("default", options[0])
        label = window_config.get("label", "관측 기간")

        window = st.selectbox(
            label=f"### {label}",
            options=options,
            index=options.index(default) if default in options else 0,
            format_func=lambda x: f"{x}개월",
            key=f"{self.dashboard_type}_window"
        )
        st.markdown("---")

        return window

    # ==================== EDA 대시보드 전용 ====================

    def render_category_selector(self) -> Optional[list]:
        """분석 카테고리 선택기 렌더링 (EDA 전용)

        Returns:
            선택된 카테고리 리스트 또는 None
        """
        category_config = self.dashboard_config.get("category_selector", {})

        if not category_config.get("enabled", False):
            return None

        label = category_config.get("label", "카테고리 선택")
        options = category_config.get("options", [])
        default = category_config.get("default", [])
        selector_type = category_config.get("type", "multiselect")

        st.markdown(f"### {label}")

        if selector_type == "multiselect":
            selected = st.multiselect(
                label=label,
                options=options,
                default=default,
                key=f"{self.dashboard_type}_category",
                label_visibility="collapsed"
            )
        else:
            selected = st.selectbox(
                label=label,
                options=options,
                index=options.index(default[0]) if default and default[0] in options else 0,
                key=f"{self.dashboard_type}_category",
                label_visibility="collapsed"
            )

        st.markdown("---")
        return selected

    def render_confidence_interval(self) -> Optional[float]:
        """신뢰구간 선택기 렌더링 (EDA 전용)

        Returns:
            선택된 신뢰구간 값 또는 None
        """
        ci_config = self.dashboard_config.get("confidence_interval", {})

        if not ci_config.get("enabled", False):
            return None

        label = ci_config.get("label", "신뢰구간")
        min_val = ci_config.get("min", 0.8)
        max_val = ci_config.get("max", 0.99)
        default = ci_config.get("default", 0.95)
        step = ci_config.get("step", 0.01)

        st.markdown(f"### {label}")
        ci_value = st.slider(
            label=label,
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=step,
            format="%.2f",
            key=f"{self.dashboard_type}_ci",
            label_visibility="collapsed"
        )
        st.caption(f"선택: {ci_value:.0%}")
        st.markdown("---")

        return ci_value

    # ==================== Cluster 대시보드 전용 ====================

    def render_model_selector(self) -> Optional[str]:
        """모델 선택기 렌더링 (Cluster 전용)

        Returns:
            선택된 모델명 또는 None
        """
        model_config = self.dashboard_config.get("model_selector", {})

        if not model_config.get("enabled", False):
            return None

        label = model_config.get("label", "모델 선택")
        options = model_config.get("options", [])
        default = model_config.get("default", options[0] if options else None)

        st.markdown(f"### {label}")
        model = st.selectbox(
            label=label,
            options=options,
            index=options.index(default) if default in options else 0,
            key=f"{self.dashboard_type}_model",
            label_visibility="collapsed"
        )
        st.markdown("---")

        return model

    def render_training_period(self) -> Optional[int]:
        """학습 기간 입력기 렌더링 (Cluster 전용)

        Returns:
            선택된 학습 기간(개월) 또는 None
        """
        period_config = self.dashboard_config.get("training_period", {})

        if not period_config.get("enabled", False):
            return None

        label = period_config.get("label", "학습 기간")
        min_val = period_config.get("min", 6)
        max_val = period_config.get("max", 24)
        default = period_config.get("default", 12)

        st.markdown(f"### {label}")
        period = st.number_input(
            label=label,
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=1,
            key=f"{self.dashboard_type}_period",
            label_visibility="collapsed"
        )
        st.caption(f"{period}개월")
        st.markdown("---")

        return period

    # ==================== 메인 렌더링 메서드 ====================

    def render_sidebar(self) -> Dict[str, Any]:
        """사이드바 전체 렌더링 및 선택된 값들 반환

        Returns:
            선택된 필터 값들을 담은 딕셔너리
        """
        filters = {}

        with st.sidebar:
            # 공통: 헤더 (로고 + 프로젝트 정보)
            self.render_header()

            # 공통: 날짜 선택기
            selected_date = self.render_date_selector()
            if selected_date:
                filters['date'] = selected_date

            # 대시보드별 필터
            if self.dashboard_type == "overview":
                window = self.render_window_selector()
                if window:
                    filters['window'] = window

            elif self.dashboard_type == "eda":
                categories = self.render_category_selector()
                if categories:
                    filters['categories'] = categories

                ci_value = self.render_confidence_interval()
                if ci_value:
                    filters['confidence_interval'] = ci_value

            elif self.dashboard_type == "cluster":
                model = self.render_model_selector()
                if model:
                    filters['model'] = model

                period = self.render_training_period()
                if period:
                    filters['training_period'] = period

        return filters


# ==================== 편의 함수 ====================

def create_sidebar(dashboard_type: str = "overview") -> Dict[str, Any]:
    """사이드바 생성 및 필터 값 반환하는 헬퍼 함수

    Args:
        dashboard_type: "overview", "eda", "cluster" 중 하나

    Returns:
        선택된 필터 값들의 딕셔너리

    Example:
        >>> filters = create_sidebar("overview")
        >>> print(filters['date'])  # datetime 객체
        >>> print(filters['window'])  # 1 또는 3
    """
    manager = SidebarManager(dashboard_type)
    return manager.render_sidebar()

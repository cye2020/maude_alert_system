# filter_manager.py
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple, Callable
from functools import wraps
import streamlit as st
from dateutil.relativedelta import relativedelta

import sys
from pathlib import Path
# 프로젝트 루트 경로 설정
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from dashboard.utils.dashboard_config import get_config


# ==================== 데코레이터 ====================

def check_enabled(config_path: str):
    """enabled 플래그를 체크하는 데코레이터

    Args:
        config_path: 체크할 설정 경로 (예: 'common.header', 'common.date_selector')

    Returns:
        enabled=False면 None을 반환, True면 원래 함수 실행

    Example:
        @check_enabled('common.header')
        def render_header(self):
            # header의 enabled가 False면 실행되지 않음
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # config_path를 따라 설정 탐색
            parts = config_path.split('.')
            config = self.cfg.sidebar

            for part in parts:
                config = config.get(part, {})
                if not config:
                    # 설정이 없으면 기본적으로 활성화
                    return func(self, *args, **kwargs)

            # enabled 체크
            if not config.get('enabled', True):
                return None

            return func(self, *args, **kwargs)

        return wrapper
    return decorator


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

    @check_enabled('common.header')
    def render_header(self):
        """프로젝트 로고 및 정보 렌더링"""
        header_config = self.common_config.get("header", {})

        # 로고
        logo_path = header_config.get("logo")
        if logo_path:
            st.image(logo_path, width='stretch')

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

    @check_enabled('common.date_selector')
    def render_date_selector(self) -> Optional[datetime]:
        """날짜 선택기 렌더링 (공통 필터 - 탭 전환 시에도 값 유지)

        Returns:
            선택된 날짜 (datetime 객체) 또는 None
        """
        date_config = self.common_config.get("date_selector", {})

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
            st.info("📌 분석 기준이 되는 날짜를 선택합니다. 이 날짜를 기준으로 데이터를 필터링하고 집계합니다.")
            col1, col2 = st.columns(2)

            with col1:
                year = st.selectbox(
                    "년도",
                    options=list(year_options),
                    index=min(default_year_index, len(list(year_options)) - 1),
                    format_func=lambda x: f"{x}년",
                    key="common_year",  # 공통 key로 모든 탭에서 값 유지
                    help="분석할 년도를 선택하세요"
                )

            with col2:
                month = st.selectbox(
                    "월",
                    options=range(1, 13),
                    index=default_month - 1,
                    format_func=lambda x: f"{x:02d}월",
                    key="common_month",  # 공통 key로 모든 탭에서 값 유지
                    help="분석할 월을 선택하세요"
                )

        selected_date = datetime(year, month, 1)
        st.markdown("---")

        return selected_date

    # ==================== 범용 위젯 렌더러 ====================

    def _apply_format_func(self, template: str, value: Any) -> str:
        """포맷 템플릿을 실제 값으로 변환

        Args:
            template: 포맷 문자열 (예: "{value}개월")
            value: 실제 값

        Returns:
            포맷팅된 문자열
        """
        return template.format(value=value)

    def render_widget(self, filter_config: Dict[str, Any], is_common: bool = False, dynamic_options: Dict[str, List] = None) -> Any:
        """config 기반으로 Streamlit 위젯을 동적 렌더링

        Args:
            filter_config: 필터 설정 딕셔너리
                - type: 위젯 타입 (selectbox, multiselect, slider, number_input 등)
                - key: 위젯 고유 키
                - label: 위젯 라벨
                - args: 위젯별 인자 (options, min_value, max_value 등)
                - enabled: (선택) 위젯 활성화 여부 (기본값: True)
                - caption: (선택) 값 표시 포맷 (예: "{value}개월")
            is_common: 공통 필터 여부 (True면 common_key, False면 dashboard_type_key)
            dynamic_options: 동적으로 채워질 옵션들 (key: options 리스트)

        Returns:
            위젯에서 선택된 값 (enabled=False면 None)
        """
        # enabled 체크 - False면 렌더링하지 않음
        if not filter_config.get("enabled", True):
            return None

        widget_type = filter_config.get("type")
        key = filter_config.get("key")
        label = filter_config.get("label", "")
        args = filter_config.get("args", {})
        # caption_template = filter_config.get("caption")  # 제거됨 - 상단 필터 배지로 대체

        # 동적 옵션이 제공되면 args의 options를 덮어씀
        if dynamic_options and key in dynamic_options:
            args = args.copy()  # 원본 수정 방지
            args["options"] = dynamic_options[key]

        # 위젯별 고유 key 생성
        if is_common:
            widget_key = f"common_{key}"  # 공통 필터는 모든 탭에서 값 유지
        else:
            widget_key = f"{self.dashboard_type}_{key}"  # 대시보드별 필터

        # help 텍스트 추출 (위젯 파라미터로 전달)
        help_text = filter_config.get("help")

        # 위젯 타입별 렌더링
        selected_value = None

        if widget_type == "selectbox":
            options = args.get("options", [])
            index = args.get("index", 0)
            format_func_template = args.get("format_func")

            # options가 dict 리스트인 경우 (label-value 형식)
            if options and len(options) > 0 and isinstance(options[0], dict) and "label" in options[0]:
                option_labels = [opt["label"] for opt in options]
                option_values = [opt["value"] for opt in options]

                selected_label = st.selectbox(
                    label=label,
                    options=option_labels,
                    index=index,
                    key=widget_key,
                    help=help_text
                )

                # label에 해당하는 value 찾기
                selected_idx = option_labels.index(selected_label)
                selected_value = option_values[selected_idx]
            else:
                # 기존 방식 (단순 리스트)
                selectbox_kwargs = {
                    "label": label,
                    "options": options,
                    "index": index,
                    "key": widget_key,
                    "help": help_text
                }

                if format_func_template:
                    selectbox_kwargs["format_func"] = lambda x, template=format_func_template: self._apply_format_func(template, x)

                selected_value = st.selectbox(**selectbox_kwargs)

        elif widget_type == "multiselect":
            options = args.get("options", [])
            default = args.get("default", [])

            # Cascading filter 지원: 범용 cascade 로직
            if dynamic_options and key in dynamic_options.get("_cascade_config", {}):
                cascade_config = dynamic_options.get("_cascade_config", {}).get(key, {})
                depends_on = cascade_config.get("depends_on")

                # depends_on이 문자열 또는 리스트일 수 있음
                if isinstance(depends_on, str):
                    depends_on = [depends_on]

                if depends_on:
                    # 의존하는 모든 필터의 값 가져오기
                    parent_values = {}
                    all_parents_selected = True

                    for parent_key in depends_on:
                        if is_common:
                            parent_widget_key = f"common_{parent_key}"
                        else:
                            parent_widget_key = f"{self.dashboard_type}_{parent_key}"

                        parent_value = st.session_state.get(parent_widget_key, [])
                        parent_values[parent_key] = parent_value

                        # 하나라도 선택 안 되었으면 cascade 안 함
                        if not parent_value or len(parent_value) == 0:
                            all_parents_selected = False

                    # 모든 parent가 선택되었을 때만 cascade 필터링
                    if all_parents_selected:
                        from utils.constants import ColumnNames
                        data_source = cascade_config.get("data_source")

                        if data_source is not None:
                            # key에 따라 적절한 함수 호출
                            if key == "products":
                                from dashboard.utils.filter_helpers import get_products_by_manufacturers
                                options = get_products_by_manufacturers(
                                    data_source,
                                    parent_values.get("manufacturers", []),
                                    manufacturer_col=ColumnNames.MANUFACTURER,
                                    product_col=ColumnNames.PRODUCT_CODE
                                )
                            elif key == "devices":
                                from dashboard.utils.filter_helpers import get_devices_by_filters
                                options = get_devices_by_filters(
                                    data_source,
                                    selected_manufacturers=parent_values.get("manufacturers"),
                                    selected_products=parent_values.get("products"),
                                    manufacturer_col=ColumnNames.MANUFACTURER,
                                    product_col=ColumnNames.PRODUCT_CODE,
                                    device_col=ColumnNames.UDI_DI
                                )

                            # 기존 선택값 중 유효한 것만 유지
                            prev_selected = st.session_state.get(f"prev_{widget_key}", [])
                            default = [p for p in prev_selected if p in options]

            selected_value = st.multiselect(
                label=label,
                options=options,
                default=default,
                key=widget_key,
                help=help_text
            )

            # 선택값 저장 (다음 렌더링에서 참조)
            st.session_state[f"prev_{widget_key}"] = selected_value

        elif widget_type == "slider":
            min_value = args.get("min_value", 0.0)
            max_value = args.get("max_value", 1.0)
            value = args.get("value", 0.5)
            step = args.get("step", 0.01)
            format_str = args.get("format", "%.2f")

            selected_value = st.slider(
                label=label,
                min_value=min_value,
                max_value=max_value,
                value=value,
                step=step,
                format=format_str,
                key=widget_key,
                help=help_text
            )

        elif widget_type == "number_input":
            min_value = args.get("min_value", 0)
            max_value = args.get("max_value", 100)
            value = args.get("value", 50)
            step = args.get("step", 1)
            format_str = args.get("format", None)

            number_input_kwargs = {
                "label": label,
                "min_value": min_value,
                "max_value": max_value,
                "value": value,
                "step": step,
                "key": widget_key,
                "help": help_text
            }

            if format_str:
                number_input_kwargs["format"] = format_str

            selected_value = st.number_input(**number_input_kwargs)

        elif widget_type == "date_selector":
            # 단일 년-월 선택기 (Spike Detection용)
            default_month = args.get("default_month", "2025-11")

            # 기본값 파싱
            try:
                default_dt = datetime.strptime(default_month, "%Y-%m")
            except:
                default_dt = self.TODAY.replace(day=1)

            # 년월 범위 계산 (최근 3년)
            min_dt = (self.TODAY - relativedelta(years=2)).replace(day=1, month=1)
            max_dt = self.TODAY.replace(day=1)

            st.markdown(f"##### {label}")

            # 년도와 월 선택
            col1, col2 = st.columns(2)

            with col1:
                year_options = range(min_dt.year, max_dt.year + 1)
                default_year_idx = list(year_options).index(default_dt.year) if default_dt.year in year_options else len(year_options) - 1

                selected_year = st.selectbox(
                    "년도",
                    options=list(year_options),
                    index=default_year_idx,
                    format_func=lambda x: f"{x}년",
                    key=f"{widget_key}_year",
                    label_visibility="collapsed"
                )

            with col2:
                month_options = range(1, 13)
                default_month_idx = default_dt.month - 1

                selected_month = st.selectbox(
                    "월",
                    options=list(month_options),
                    index=default_month_idx,
                    format_func=lambda x: f"{x:02d}월",
                    key=f"{widget_key}_month",
                    label_visibility="collapsed"
                )

            # YYYY-MM 형식 문자열로 반환
            selected_value = f"{selected_year:04d}-{selected_month:02d}"

        elif widget_type == "month_range_picker":
            # 슬라이더를 사용한 년월 범위 선택

            # 3년 전 계산 (defaults.yaml에서 설정된 기간 사용)
            analysis_period_years = self.cfg.defaults.get("analysis_period_years", 3)
            min_dt = (self.TODAY - relativedelta(years=analysis_period_years-1)).replace(day=1, month=1)
            max_dt = self.TODAY.replace(day=1)
            default_start_dt = (self.TODAY - relativedelta(years=1)).replace(day=1)

            # 시간 정보 제거 (date만 사용) - slider는 date 객체에서 더 잘 작동
            from datetime import date
            min_date = date(min_dt.year, min_dt.month, 1)
            max_date = date(max_dt.year, max_dt.month, 1) - relativedelta(months=1)
            default_start = date(default_start_dt.year, default_start_dt.month, 1) - relativedelta(months=1)

            # 슬라이더로 범위 선택
            selected_range = st.slider(
                label=label,  # YAML에서 설정한 label 사용
                min_value=min_date,
                max_value=max_date,
                value=(default_start, max_date),
                key=widget_key,
                format="YYYY-MM",
                help=help_text
            )

            # datetime 객체로 변환 (매월 1일, 시간은 00:00:00)
            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                start_date = datetime.combine(selected_range[0], datetime.min.time())
                end_date = datetime.combine(selected_range[1], datetime.min.time())
            else:
                start_date = datetime.combine(default_start, datetime.min.time())
                end_date = datetime.combine(max_date, datetime.min.time())

            # 선택된 기간 표시 (제거됨 - 상단 필터 배지로 대체)
            # st.caption(f"📅 {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")

            selected_value = (start_date, end_date)

            # 계산된 날짜를 세션 스테이트에 명시적으로 저장 (overview_tab에서 사용)
            st.session_state[f"{widget_key}_start_computed"] = start_date
            st.session_state[f"{widget_key}_end_computed"] = end_date

            self.start_date = start_date
            self.end_date = end_date

        # Caption 렌더링 (제거됨 - 상단 필터 배지로 대체)
        # if caption_template and selected_value is not None:
        #     caption_text = self._apply_format_func(caption_template, selected_value)
        #     st.caption(caption_text)

        st.markdown("---")

        return selected_value

    # ==================== 메인 렌더링 메서드 ====================

    def render_sidebar(self, dynamic_options: Dict[str, List] = None) -> Dict[str, Any]:
        """사이드바 전체 렌더링 및 선택된 값들 반환

        Args:
            dynamic_options: 동적으로 채워질 옵션들 (key: options 리스트)

        Returns:
            선택된 필터 값들을 담은 딕셔너리
        """
        filters = {}

        with st.sidebar:
            # 공통: 헤더 (로고 + 프로젝트 정보)
            self.render_header()
            st.markdown('---')
            # 공통: 날짜 선택기
            selected_date = self.render_date_selector()
            if selected_date:
                filters['date'] = selected_date

            # 공통: 공통 필터 (모든 탭에서 공유)
            common_filter_configs = self.common_config.get("filters", [])
            for filter_config in common_filter_configs:
                key = filter_config.get("key")
                value = self.render_widget(filter_config, is_common=True, dynamic_options=dynamic_options)
                filters[key] = value  # None이어도 저장 (전체 선택 의미)

            # 대시보드별 필터 (config에서 동적으로 생성)
            filter_configs = self.dashboard_config.get("filters", [])
            for filter_config in filter_configs:
                key = filter_config.get("key")
                value = self.render_widget(filter_config, dynamic_options=dynamic_options)
                if value is not None:
                    filters[key] = value

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

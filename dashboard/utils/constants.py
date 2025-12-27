# constants.py
"""공통 상수 및 설정 값 (YAML에서 로드)

이 모듈은 모든 대시보드에서 사용하는 공통 상수를 정의합니다.
용어 통일을 위해 terminology.yaml을 함께 사용합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from dashboard.utils.dashboard_config import get_config
from dashboard.utils.terminology import get_term_manager


# Config 로드
_config = get_config()
_defaults_config = _config._defaults if hasattr(_config, '_defaults') else {}
_ui_standards = _config._ui_standards if hasattr(_config, '_ui_standards') else {}

# Terminology Manager (용어 통일)
_term = get_term_manager()


class ColumnNames:
    """데이터베이스 컬럼명 상수"""
    _cols = _defaults_config.get('columns', {})

    MANUFACTURER = _cols.get('manufacturer', 'manufacturer_name')
    PRODUCT_CODE = _cols.get('product_code', 'product_code')
    DATE_RECEIVED = _cols.get('date_received', 'date_received')
    DATE_OCCURRED = _cols.get('date_occurred', 'date_occurred')
    DEFECT_TYPE = _cols.get('defect_type', 'defect_type')
    PROBLEM_COMPONENTS = _cols.get('problem_components', 'problem_components')
    EVENT_TYPE = _cols.get('event_type', 'event_type')
    PATIENT_HARM = _cols.get('patient_harm', 'patient_harm')
    DEFECT_CONFIRMED = _cols.get('defect_confirmed', 'defect_confirmed')
    UDI_DI = _cols.get('udi_di', 'udi_di')
    CLUSTER = _cols.get('cluster', 'cluster')


class EventTypes:
    """이벤트 타입 상수"""
    _event_types = _defaults_config.get('event_types', {})

    DEATH = _event_types.get('death', 'Death')
    INJURY = _event_types.get('injury', 'Injury')
    SERIOUS_INJURY = _event_types.get('serious_injury', 'Serious Injury')
    MALFUNCTION = _event_types.get('malfunction', 'Malfunction')


class PatientHarmLevels:
    """환자 피해 등급"""
    _harm_levels = _defaults_config.get('patient_harm_levels', {})

    SERIOUS = _harm_levels.get('serious', ['Serious Injury', 'Death'])
    MINOR = _harm_levels.get('minor', ['Minor Injury'])
    NONE = _harm_levels.get('none', ['No Apparent Injury'])


class Defaults:
    """기본 설정 값"""
    _defaults = _defaults_config.get('defaults', {})

    # 분석 기본값
    TOP_N = _defaults.get('top_n', 10)
    MIN_CASES = _defaults.get('min_cases', 10)
    WINDOW_SIZE = _defaults.get('window_size', 1)
    DATE_FORMAT = _defaults.get('date_format', "%Y-%m")

    # UI 설정
    CHART_HEIGHT = _defaults.get('chart_height', 600)
    MAX_ITEMS_DISPLAY = _defaults.get('max_items_display', 100)

    # 제외 값
    EXCLUDE_DEFECT_TYPES = _defaults.get('exclude_defect_types', ['Other', 'Unknown'])
    MISSING_VALUE_LABEL = _defaults.get('missing_value_label', '(정보 없음)')

    # CFR 기본값
    _cfr = _defaults.get('cfr_defaults', {})
    CFR_TOP_N = _cfr.get('top_n', 20)
    CFR_MIN_CASES = _cfr.get('min_cases', 10)

    # 부품 분석 기본값
    _component = _defaults.get('component_defaults', {})
    COMPONENT_TOP_N = _component.get('top_n', 10)


class ChartStyles:
    """차트 스타일 설정"""
    _styles = _defaults_config.get('chart_styles', {})

    # 색상
    _colors = _styles.get('colors', {})
    PRIMARY_COLOR = _colors.get('primary', '#1f77b4')
    DANGER_COLOR = _colors.get('danger', '#d62728')
    WARNING_COLOR = _colors.get('warning', '#ff7f0e')
    SUCCESS_COLOR = _colors.get('success', '#2ca02c')

    # Plotly 설정
    PLOTLY_CONFIG = _styles.get('plotly', {
        'margin': {'l': 50, 'r': 20, 't': 40, 'b': 80},
        'hovermode': 'x unified'
    })


class DisplayNames:
    """UI 표시 이름 (한글) - ui_standards.yaml에서 로드"""

    # 페이지/탭 제목
    _page_titles = _ui_standards.get('page_titles', {})
    _icons = _ui_standards.get('icons', {})
    _full_titles = _ui_standards.get('full_titles', {})

    OVERVIEW = _page_titles.get('overview', '개요')
    EDA = _page_titles.get('eda', '상세 분석')
    SPIKE = _page_titles.get('spike', '급증 탐지')
    CLUSTER = _page_titles.get('cluster', '클러스터 분석')

    ICON_OVERVIEW = _icons.get('overview', '📊')
    ICON_EDA = _icons.get('eda', '📈')
    ICON_SPIKE = _icons.get('spike', '🚨')
    ICON_CLUSTER = _icons.get('cluster', '🔍')

    FULL_TITLE_OVERVIEW = _full_titles.get('overview', '📊 개요')
    FULL_TITLE_EDA = _full_titles.get('eda', '📈 상세 분석')
    FULL_TITLE_SPIKE = _full_titles.get('spike', '🚨 급증 탐지')
    FULL_TITLE_CLUSTER = _full_titles.get('cluster', '🔍 클러스터 분석')

    # 메트릭 라벨
    _metric_labels = _ui_standards.get('metric_labels', {})

    TOTAL_REPORTS = _metric_labels.get('total_reports', '총 보고 건수')
    TOTAL_CASES = _metric_labels.get('total_cases', '전체 케이스')
    CFR = _metric_labels.get('cfr', '치명률')
    DEATH_RATE = _metric_labels.get('death_rate', '사망률')
    DEATH_COUNT = _metric_labels.get('death_count', '사망')
    SERIOUS_INJURY = _metric_labels.get('serious_injury', '중증 부상')
    SERIOUS_INJURY_RATE = _metric_labels.get('serious_injury_rate', '중증 부상률')
    MINOR_INJURY = _metric_labels.get('minor_injury', '경증 부상')
    NO_HARM = _metric_labels.get('no_harm', '부상 없음')
    SEVERE_HARM_RATE = _metric_labels.get('severe_harm_rate', '중대 피해 발생률')

    MANUFACTURER = _metric_labels.get('manufacturer', '제조사')
    PRODUCT = _metric_labels.get('product', '제품군')
    DEFECT_TYPE = _metric_labels.get('defect_type', '결함 유형')
    CLUSTER = _metric_labels.get('cluster', '클러스터')
    COMPONENT = _metric_labels.get('component', '부품')
    PROBLEM_COMPONENT = _metric_labels.get('problem_component', '문제 부품')

    DEFECT_CONFIRMED_RATE = _metric_labels.get('defect_confirmed_rate', '제조사 결함 확정률')
    MOST_CRITICAL_DEFECT_TYPE = _metric_labels.get('most_critical_defect_type', '가장 치명적인 결함 유형')
    REPORT_COUNT = _metric_labels.get('report_count', '보고 건수')
    RATIO = _metric_labels.get('ratio', '비율')
    PERCENTAGE = _metric_labels.get('percentage', '백분율')

    # 섹션 제목
    _section_titles = _ui_standards.get('section_titles', {})

    SUMMARY = _section_titles.get('summary', '요약')
    DETAILED_ANALYSIS = _section_titles.get('detailed_analysis', '상세 분석')
    INSIGHTS = _section_titles.get('insights', '인사이트')
    DATA_TABLE = _section_titles.get('data_table', '상세 데이터')
    TIME_SERIES = _section_titles.get('time_series', '시계열 분석')
    MONTHLY_TREND = _section_titles.get('monthly_trend', '월별 추이')
    MONTHLY_REPORTS = _section_titles.get('monthly_reports', '월별 보고서 수')
    HARM_DISTRIBUTION = _section_titles.get('harm_distribution', '환자 피해 분포')
    DEFECT_ANALYSIS = _section_titles.get('defect_analysis', '결함 분석')
    COMPONENT_ANALYSIS = _section_titles.get('component_analysis', '문제 부품 분석')
    CFR_ANALYSIS = _section_titles.get('cfr_analysis', '기기별 치명률(CFR) 분석')

    # 메시지
    _messages = _ui_standards.get('messages', {})

    NO_DATA = _messages.get('no_data', '선택한 조건에 해당하는 데이터가 없습니다.')
    LOADING = _messages.get('loading', '데이터 로딩 중...')
    ANALYZING = _messages.get('analyzing', '분석 중...')


class HarmColors:
    """환자 피해 관련 색상 (ui_standards.yaml에서 로드)"""
    _harm_colors = _ui_standards.get('colors', {}).get('harm', {})

    DEATH = _harm_colors.get('death', '#DC2626')
    SERIOUS_INJURY = _harm_colors.get('serious_injury', '#F59E0B')
    MINOR_INJURY = _harm_colors.get('minor_injury', '#ffd700')
    NO_HARM = _harm_colors.get('no_harm', '#2ca02c')
    UNKNOWN = _harm_colors.get('unknown', '#9CA3AF')


class SeverityColors:
    """위험도/패턴 관련 색상 (ui_standards.yaml에서 로드)"""
    _severity_colors = _ui_standards.get('colors', {}).get('severity', {})

    SEVERE = _severity_colors.get('severe', '#DC2626')
    ALERT = _severity_colors.get('alert', '#F59E0B')
    ATTENTION = _severity_colors.get('attention', '#ffd700')
    GENERAL = _severity_colors.get('general', '#2ca02c')


# ==================== 용어 통일 (Terminology) ====================
class Terms:
    """용어 통일 클래스

    terminology.yaml에서 용어를 동적으로 로드하여 일관된 용어 사용을 보장합니다.

    Usage:
        >>> Terms.KOREAN.CFR  # '치명률'
        >>> Terms.KOREAN.DEATH_RATE  # '사망률'
        >>> Terms.COLUMN.DEATH_COUNT  # 'death_count'
        >>> Terms.format_message('high_cfr_alert', device='ABC', cfr=12.5, count=100)
    """

    # 동적으로 생성된 한국어 용어 (terminology.yaml에서 자동 로드)
    KOREAN = _term.korean if hasattr(_term, 'korean') else type('KOREAN', (), {})()

    # 동적으로 생성된 컬럼명 (terminology.yaml에서 자동 로드)
    COLUMN = _term.columns.calculated_columns if hasattr(_term, 'columns') else type('COLUMN', (), {})()

    @staticmethod
    def get_column_header(col_name: str) -> str:
        """컬럼명 -> 표시용 한글 헤더 변환

        Args:
            col_name: 영문 컬럼명

        Returns:
            한글 컬럼 헤더

        Example:
            >>> Terms.get_column_header('death_count')
            '사망'
        """
        headers = _term.column_headers
        return headers.get(col_name, col_name)

    @staticmethod
    def rename_columns(df, mapping: dict = None):
        """DataFrame 컬럼명을 한글로 변환

        Args:
            df: polars 또는 pandas DataFrame
            mapping: 커스텀 매핑 (없으면 기본 매핑 사용)

        Returns:
            컬럼명이 변환된 DataFrame
        """
        if mapping is None:
            mapping = _term.column_headers

        # polars인지 pandas인지 확인
        if hasattr(df, 'rename'):
            # polars
            return df.rename(mapping)
        else:
            # pandas
            return df.rename(columns=mapping)

    @staticmethod
    def format_message(template_key: str, **kwargs) -> str:
        """메시지 템플릿 포맷

        Args:
            template_key: 메시지 템플릿 키
            **kwargs: 템플릿 변수

        Returns:
            포맷된 메시지

        Example:
            >>> Terms.format_message('high_cfr_alert', device='ABC', cfr=12.5, count=100)
        """
        return _term.format_message(template_key, **kwargs)

    @staticmethod
    def get_description(term_key: str) -> str:
        """용어 설명 가져오기

        Args:
            term_key: 용어 키

        Returns:
            설명 텍스트
        """
        return _term.get_description(term_key)

    @staticmethod
    def section_title(template_key: str, **kwargs) -> str:
        """섹션 제목 생성 (템플릿 사용)

        Args:
            template_key: 템플릿 키 (예: 'entity_analysis', 'metric_by_entity')
            **kwargs: 템플릿 변수

        Returns:
            생성된 제목

        Example:
            >>> Terms.section_title('entity_analysis', entity='결함 유형')
            '결함 유형 분석'
            >>> Terms.section_title('metric_by_entity', entity='제조사', metric='치명률')
            '제조사별 치명률'
        """
        template = _term.get(f'korean_terms.sections.{template_key}', '')
        if not template:
            return ''
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"[Title format error: missing {e}]"

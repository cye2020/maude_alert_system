# terminology.py
"""용어 통일 관리 모듈

중앙 집중식 용어 사전(terminology.yaml)을 로드하고 관리합니다.
모든 대시보드 코드에서 일관된 용어 사용을 보장합니다.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class TerminologyManager:
    """용어 사전 관리 클래스

    Usage:
        term = TerminologyManager()

        # 용어 가져오기 (dot notation)
        cfr_label = term.get('korean_terms.metrics.cfr')  # '치명률'
        death_col = term.get('column_names.calculated_columns.death_count')  # 'death_count'

        # 직접 속성 접근
        cfr_label = term.korean.metrics.cfr  # '치명률'

        # 컬럼 헤더 매핑
        df_display = df.rename(term.column_headers)

        # 메시지 포맷
        msg = term.format_message('high_cfr_alert', device='ABC', cfr=12.5, count=100)
    """

    _instance = None
    _config = None

    def __new__(cls):
        """싱글톤 패턴 - 한 번만 로드"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """terminology.yaml 로드"""
        if self._config is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'dashboard' / 'terminology.yaml'

            if not config_path.exists():
                raise FileNotFoundError(f"Terminology config not found: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                self.__class__._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """dot notation으로 값 가져오기

        Args:
            key_path: 'section.subsection.key' 형식의 경로
            default: 값이 없을 때 기본값

        Returns:
            해당하는 값, 없으면 default

        Example:
            >>> term = TerminologyManager()
            >>> term.get('korean_terms.metrics.cfr')
            '치명률'
            >>> term.get('korean_terms.metrics.cfr_full')
            '치명률(CFR)'
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    # ==================== 편의 속성 (자주 사용하는 것들) ====================

    @property
    def korean(self) -> 'KoreanTerms':
        """한국어 용어 접근

        Example:
            >>> term.korean.metrics.cfr
            '치명률'
        """
        return KoreanTerms(self._config.get('korean_terms', {}))

    @property
    def english(self) -> 'EnglishTerms':
        """영문 용어 접근"""
        return EnglishTerms(self._config.get('english_terms', {}))

    @property
    def columns(self) -> 'ColumnNames':
        """컬럼명 접근

        Example:
            >>> term.columns.death_count
            'death_count'
            >>> term.columns.manufacturer
            'manufacturer_name'
        """
        return ColumnNames(self._config.get('column_names', {}))

    @property
    def column_headers(self) -> Dict[str, str]:
        """DataFrame 표시용 컬럼 헤더 매핑

        Returns:
            영문 컬럼명 -> 한글 컬럼명 딕셔너리

        Example:
            >>> df.rename(columns=term.column_headers)
        """
        return self._config.get('column_headers', {})

    @property
    def formats(self) -> Dict[str, str]:
        """표시 형식 포맷

        Example:
            >>> term.formats['percentage']
            '{:.2f}%'
        """
        return self._config.get('display_formats', {})

    @property
    def messages(self) -> Dict[str, str]:
        """메시지 템플릿"""
        return self._config.get('message_templates', {})

    def format_message(self, message_key: str, **kwargs) -> str:
        """메시지 템플릿 포맷

        Args:
            message_key: 메시지 템플릿 키
            **kwargs: 템플릿에 전달할 변수

        Returns:
            포맷된 메시지

        Example:
            >>> term.format_message('high_cfr_alert',
            ...                     device='ABC Corp', cfr=12.5, count=100)
            '⚠️ **ABC Corp**의 치명률이 **12.50%**로 매우 높습니다 (중대 피해 100건)'
        """
        template = self.messages.get(message_key, '')
        if not template:
            return ''

        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"[Message format error: missing {e}]"

    def get_description(self, term_key: str) -> str:
        """용어 설명 가져오기 (툴팁용)

        Args:
            term_key: 용어 키 (예: 'cfr', 'spike')

        Returns:
            용어 설명
        """
        descriptions = self._config.get('term_descriptions', {})
        return descriptions.get(term_key, '')


class _DotDict:
    """Dict를 dot notation으로 접근 가능하게 하는 헬퍼 클래스

    snake_case 키를 UPPER_CASE로 자동 변환하여 접근 가능
    """

    def __init__(self, data: dict):
        self._data = data
        # snake_case -> UPPER_CASE 매핑 캐시
        self._upper_cache = {}
        self._build_upper_mapping()

    def _build_upper_mapping(self):
        """모든 중첩된 키를 UPPER_CASE로 매핑"""
        def flatten_dict(d: dict, parent_key: str = ''):
            items = []
            for k, v in d.items():
                upper_key = k.upper()
                new_key = f"{parent_key}.{k}" if parent_key else k

                if isinstance(v, dict):
                    # 중첩된 dict는 재귀적으로 처리
                    items.extend(flatten_dict(v, new_key))
                else:
                    # 값 저장 (UPPER_CASE 키로 접근 가능)
                    self._upper_cache[upper_key] = v
                    items.append((upper_key, v))
            return items

        flatten_dict(self._data)

    def __getattr__(self, name: str) -> Any:
        # 1. 원본 키로 시도 (snake_case)
        value = self._data.get(name)
        if value is not None:
            if isinstance(value, dict):
                return _DotDict(value)
            return value

        # 2. UPPER_CASE 키로 시도
        lower_name = name.lower()
        value = self._data.get(lower_name)
        if value is not None:
            if isinstance(value, dict):
                return _DotDict(value)
            return value

        # 3. 캐시에서 직접 검색
        if name in self._upper_cache:
            return self._upper_cache[name]

        return None

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, key: str, default=None) -> Any:
        result = self.__getattr__(key)
        return result if result is not None else default


class KoreanTerms(_DotDict):
    """한국어 용어 접근용 클래스

    Example:
        >>> korean = KoreanTerms(config['korean_terms'])
        >>> korean.metrics.cfr
        '치명률'
        >>> korean.entities.manufacturer
        '제조사'
    """
    pass


class EnglishTerms(_DotDict):
    """영문 용어 접근용 클래스"""
    pass


class ColumnNames(_DotDict):
    """컬럼명 접근용 클래스

    Example:
        >>> cols = ColumnNames(config['column_names'])
        >>> cols.manufacturer
        'manufacturer_name'
        >>> cols.calculated_columns.death_count
        'death_count'
    """
    pass


# ==================== 전역 인스턴스 ====================
# 모듈 임포트 시 자동으로 생성
_term_manager = None

def get_term_manager() -> TerminologyManager:
    """전역 TerminologyManager 인스턴스 가져오기

    Returns:
        TerminologyManager 싱글톤 인스턴스
    """
    global _term_manager
    if _term_manager is None:
        _term_manager = TerminologyManager()
    return _term_manager


# ==================== 편의 함수 ====================

def get_korean_term(key_path: str, default: str = '') -> str:
    """한국어 용어 가져오기

    Args:
        key_path: 'metrics.cfr' 형식의 경로 (korean_terms. 생략)
        default: 기본값

    Returns:
        한국어 용어

    Example:
        >>> get_korean_term('metrics.cfr')
        '치명률'
    """
    term = get_term_manager()
    return term.get(f'korean_terms.{key_path}', default)


def get_column_name(key: str, default: str = '') -> str:
    """컬럼명 가져오기

    Args:
        key: 컬럼 키 (예: 'manufacturer', 'death_count')
        default: 기본값

    Returns:
        실제 컬럼명

    Example:
        >>> get_column_name('manufacturer')
        'manufacturer_name'
        >>> get_column_name('calculated_columns.death_count')
        'death_count'
    """
    term = get_term_manager()

    # calculated_columns인지 확인
    if '.' in key:
        return term.get(f'column_names.{key}', default)
    else:
        # 일반 컬럼
        value = term.get(f'column_names.{key}')
        if value is None:
            # calculated_columns에서 찾기
            value = term.get(f'column_names.calculated_columns.{key}', default)
        return value


def format_percentage(value: float, decimals: int = 2) -> str:
    """퍼센트 포맷

    Args:
        value: 숫자값
        decimals: 소수점 자릿수

    Returns:
        포맷된 문자열

    Example:
        >>> format_percentage(12.3456)
        '12.35%'
    """
    return f"{value:.{decimals}f}%"


def format_integer(value: int) -> str:
    """정수 포맷 (천 단위 구분)

    Args:
        value: 정수값

    Returns:
        포맷된 문자열

    Example:
        >>> format_integer(1234567)
        '1,234,567'
    """
    return f"{value:,}"


# ==================== 사용 예시 ====================
if __name__ == '__main__':
    # 테스트
    term = TerminologyManager()

    print("=== 한국어 용어 ===")
    print(f"CFR: {term.korean.metrics.cfr}")
    print(f"사망률: {term.korean.metrics.death_rate}")
    print(f"제조사: {term.korean.entities.manufacturer}")
    print(f"급증: {term.korean.analysis.spike}")

    print("\n=== 컬럼명 ===")
    print(f"manufacturer: {term.columns.manufacturer}")
    print(f"death_count: {term.columns.calculated_columns.death_count}")

    print("\n=== 메시지 ===")
    msg = term.format_message('high_cfr_alert',
                              device='ABC Corp - XYZ',
                              cfr=12.5,
                              count=100)
    print(msg)

    print("\n=== 설명 ===")
    print(term.get_description('cfr'))

    print("\n=== UPPER_CASE 접근 테스트 ===")
    print(f"term.korean.metrics.CFR: {term.korean.metrics.CFR}")
    print(f"term.korean.entities.MANUFACTURER: {term.korean.entities.MANUFACTURER}")
    print(f"term.korean.analysis.SPIKE: {term.korean.analysis.SPIKE}")

    print("\n✅ snake_case와 UPPER_CASE 모두 동작!")

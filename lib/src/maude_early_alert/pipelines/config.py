"""
파이프라인 설정 관리

Bronze: load/extract.yaml, load/load.yaml
Silver: preprocess/columns.yaml, preprocess/cleaning.yaml, preprocess/filtering.yaml
"""
from typing import List, Optional
from maude_early_alert.utils.config_loader import load_config


class BronzeConfig:
    """Bronze 레이어 적재 설정"""

    def __init__(self):
        self._extract = load_config('extract')
        self.storage = load_config('storage')

    @property
    def extract(self) -> dict:
        return self._extract

    @property
    def storage(self) -> dict:
        return self.storage

    # ==================== extract 설정 ====================

    def get_extract_url(self) -> str:
        """FDA API 다운로드 URL"""
        return self._extract['url']

    def get_extract_period(self) -> int:
        """추출 기간 (연 단위)"""
        return self._extract['period']

    def get_extract_categories(self) -> List[str]:
        """추출 대상 카테고리 목록 (event, udi)"""
        return self._extract['categories']

    # ==================== storage 설정 ====================

    def get_s3_enabled(self) -> bool:
        """S3 사용 여부"""
        return self.storage['s3']['enabled']

    def get_s3_region(self) -> str:
        """S3 리전"""
        return self.storage['s3']['region']

    def get_s3_bucket_name(self) -> str:
        """S3 버킷 이름"""
        return self.storage['s3']['bucket_name']

    def get_snowflake_enabled(self) -> bool:
        """Snowflake 사용 여부"""
        return self.storage['snowflake']['enabled']    

    def get_snowflake_load_database(self) -> str:
        """Snowflake load 데이터베이스"""
        return self.storage['snowflake']['load']['database']

    def get_snowflake_load_schema(self) -> str:
        """Snowflake load 스키마"""
        return self.storage['snowflake']['load']['schema']
    
    def get_snowflake_load_tables(self) -> List[str]:
        """Snowflake load 테이블"""
        return self.storage['snowflake']['load']['table']

    def get_snowflake_load_stage(self) -> str:
        """Snowflake load S3 스테이지명"""
        return self.storage['snowflake']['load']['stage']

class SilverConfig:
    """Silver 레이어 전처리 설정"""

    def __init__(self):
        self._columns = load_config('preprocess/columns')
        self._cleaning = load_config('preprocess/cleaning')
        self._filtering = load_config('preprocess/filtering')

    @property
    def columns(self) -> dict:
        """컬럼 설정 (columns.yaml)"""
        return self._columns

    @property
    def cleaning(self) -> dict:
        """클린징 설정 (cleaning.yaml)"""
        return self._cleaning

    @property
    def filtering(self) -> dict:
        """필터링 설정 (filtering.yaml)"""
        return self._filtering


# ==================== ConfigManager (싱글톤 패턴) ====================

class ConfigManager:
    """파이프라인 Config 매니저"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._bronze_config = None
        self._silver_config = None

    @property
    def bronze(self) -> BronzeConfig:
        """Bronze 적재 설정"""
        if self._bronze_config is None:
            self._bronze_config = BronzeConfig()
        return self._bronze_config

    @property
    def silver(self) -> SilverConfig:
        """Silver 전처리 설정"""
        if self._silver_config is None:
            self._silver_config = SilverConfig()
        return self._silver_config


_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """전역 ConfigManager 인스턴스 반환

    사용 예시:
        from maude_early_alert.pipelines.config import get_config

        # Bronze
        bcfg = get_config().bronze
        url = bcfg.get_extract_url()
        categories = bcfg.get_extract_categories()

        # Silver
        scfg = get_config().silver
        event_cols = scfg.columns['event']['cols']
        build_clean_sql("table", scfg.cleaning['maude'], udf_schema=...)
        build_filter_pipeline(scfg.filtering, source='EVENT_STAGE_10')
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

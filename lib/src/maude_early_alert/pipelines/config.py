"""
파이프라인 설정 관리

Bronze: load/extract.yaml, load/load.yaml
Silver: preprocess/columns.yaml, preprocess/cleaning.yaml, preprocess/filtering.yaml
"""
from typing import Dict, List, Optional, Union
from maude_early_alert.utils.config_loader import load_config


class BronzeConfig:
    """Bronze 레이어 적재 설정"""

    def __init__(self):
        self._extract = load_config('extract')
        self._storage = load_config('storage')

    @property
    def extract(self) -> dict:
        return self._extract

    @property
    def storage(self) -> dict:
        return self._storage

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
        """Snowflake load 테이블명 목록"""
        return list(self.storage['snowflake']['load']['tables'].keys())

    def get_snowflake_load_table_config(self, table_name: str) -> Dict[str, Union[str, List[str]]]:
        """테이블별 적재 설정 (primary_key, business_primary_key)"""
        return self.storage['snowflake']['load']['tables'][table_name]

    def get_snowflake_load_primary_key(self, table_name: str) -> Union[str, List[str]]:
        """테이블의 MERGE primary key"""
        return self.storage['snowflake']['load']['tables'][table_name]['primary_key']

    def get_snowflake_load_business_primary_key(self, table_name: str) -> str:
        """테이블의 비즈니스 primary key"""
        return self.storage['snowflake']['load']['tables'][table_name]['business_primary_key']

    def get_snowflake_load_stage(self) -> str:
        """Snowflake load S3 스테이지명"""
        return self.storage['snowflake']['load']['stage']

class SilverConfig:
    """Silver 레이어 전처리 설정"""

    def __init__(self):
        self._filtering = load_config('preprocess/filtering')
        self._columns = load_config('preprocess/columns')
        self._cleaning = load_config('preprocess/cleaning')
        self._flatten = load_config('preprocess/flatten')
        self._transform = load_config('preprocess/transform')
        self._imputation = load_config('preprocess/imputation')
        self._matching = load_config('preprocess/matching')
        self._llm_extraction = load_config('preprocess/llm_extraction')
        self._storage = load_config('storage')

    @property
    def filtering(self) -> dict:
        """필터링 설정 (filtering.yaml)"""
        return self._filtering

    @property
    def columns(self) -> dict:
        """컬럼 설정 (columns.yaml)"""
        return self._columns

    @property
    def cleaning(self) -> dict:
        """클린징 설정 (cleaning.yaml)"""
        return self._cleaning

    # ==================== snowflake 설정 ====================

    def get_snowflake_enabled(self) -> bool:
        """Snowflake 사용 여부"""
        return self._storage['snowflake']['enabled']
    
    def get_snowflake_load_database(self) -> str:
        """Silver load 데이터베이스"""
        return self._storage['snowflake']['load']['database']

    def get_snowflake_load_schema(self) -> str:
        """Silver load 스키마"""
        return self._storage['snowflake']['load']['schema']

    def get_snowflake_transform_database(self) -> str:
        """Silver transform 데이터베이스"""
        return self._storage['snowflake']['transform']['database']

    def get_snowflake_transform_schema(self) -> str:
        """Silver transform 스키마"""
        return self._storage['snowflake']['transform']['schema']

    def get_snowflake_udf_database(self) -> str:
        """UDF 데이터베이스"""
        return self._storage['snowflake']['udf']['database']

    def get_snowflake_udf_schema(self) -> str:
        """UDF 스키마"""
        return self._storage['snowflake']['udf']['schema']

    # ==================== transform 설정 ====================

    def get_combine_category(self) -> str:
        return self._transform['combine']['category']

    def get_combine_columns(self) -> dict:
        return self._transform['combine']['columns']

    def get_primary_category(self) -> str:
        return self._transform['primary']['category']

    def get_primary_columns(self) -> dict:
        return self._transform['primary']['columns']

    def get_extract_udi_di_category(self) -> str:
        return self._transform['extract_udi_di']['category']

    def get_extract_udi_di_columns(self) -> dict:
        return self._transform['extract_udi_di']['columns']

    def get_fuzzy_match_threshold(self) -> float:
        return self._transform['fuzzy_match']['threshold']

    def get_fuzzy_match_source_category(self) -> str:
        return self._transform['fuzzy_match']['source_category']

    def get_fuzzy_match_target_category(self) -> str:
        return self._transform['fuzzy_match']['target_category']

    def get_fuzzy_match_mfr_col(self) -> str:
        return self._transform['fuzzy_match']['mfr_col']

    def get_ma_company_col(self) -> str:
        return self._transform['M&A']['company_col']

    def get_ma_aliases(self) -> dict:
        return self._transform['M&A']['aliases']

    # ==================== columns 설정 ====================

    def get_column_categories(self) -> List[str]:
        """columns.yaml에 정의된 카테고리 목록 반환 (e.g. ['event', 'udi'])"""
        return list(self._columns.keys())

    def get_final_categories(self) -> List[str]:
        """final 스테이지가 있는 카테고리만 반환 (e.g. ['event'])"""
        return [cat for cat in self._columns if self.get_final_cols(cat) is not None]

    def get_column_cols(self, category: str) -> List[dict]:
        """카테고리별 cols 목록 반환

        Args:
            category: 카테고리명 (e.g. 'event', 'udi')

        Returns:
            [{'name': ..., 'type': ..., 'alias': ..., ...}, ...]
        """
        return self._columns.get(category, {}).get('cols', [])

    def get_final_cols(self, category: str) -> Optional[List[dict]]:
        """카테고리의 final 컬럼 목록 반환 (name → alias로 교체)

        final 필드가 정의되지 않은 카테고리(e.g. udi)는 None 반환.

        Args:
            category: 카테고리명 (e.g. 'event', 'udi')

        Returns:
            final=True인 컬럼 리스트 (name이 alias로 교체됨), 또는 None
        """
        cols = self.get_column_cols(category)
        if not cols or not any('final' in d for d in cols):
            return None
        return [
            {**d, 'name': d['alias']}
            for d in cols
            if d.get('final', False)
        ]

    # ==================== cleaning 설정 ====================

    def get_cleaning_categories(self) -> List[str]:
        """cleaning 처리 대상 카테고리 목록 반환"""
        return list(self._cleaning['categories'].keys())

    def get_cleaning_config(self, category: str) -> dict:
        """카테고리별 cleaning 설정 반환"""
        return self._cleaning['categories'][category]

    # ==================== matching 설정 ====================

    def get_matching_target_category(self) -> str:
        """매칭 결과가 기록되는 카테고리 반환 (e.g. 'event')"""
        return self._matching['categories']['target']

    def get_matching_source_category(self) -> str:
        """UDI 참조 데이터 카테고리 반환 (e.g. 'udi')"""
        return self._matching['categories']['source']

    def get_matching_kwargs(self) -> dict:
        """build_matching_sql에 **spread할 인자 dict 반환"""
        m = self._matching
        return {
            **m['columns'],
            'udi_col_prefix':   m['udi_col_prefix'],
            'status':           m['status'],
            'confidence':       m['confidence'],
            'min_device_match': m['thresholds']['min_device_match'],
        }

    # ==================== imputation 설정 ====================

    def get_imputation_categories(self) -> List[str]:
        """결측치 처리 대상 카테고리 목록 반환"""
        return list(self._imputation.keys())

    def get_imputation_alias(self, category: str) -> str:
        """카테고리의 테이블 alias 반환"""
        return self._imputation[category]['alias']

    def get_imputation_mode(self, category: str) -> Dict[str, str]:
        """카테고리의 {group_col: target_col} 매핑 반환"""
        return {item['group']: item['target'] for item in self._imputation[category]['mode']}

    # ==================== flatten 설정 ====================

    def get_flatten_categories(self) -> List[str]:
        """flatten 대상 카테고리 목록 반환 (e.g. ['event', 'udi'])"""
        return list(self._flatten.keys())

    def get_flatten_config(self, category: str) -> Dict[str, List[str]]:
        """카테고리별 flatten 전략 딕셔너리 반환

        Args:
            category: 카테고리명 (e.g. 'event', 'udi')

        Returns:
            {'flatten': [...], 'transform': [...], 'first_element': [...]}

        Example:
            scfg.get_flatten_config('event')
            # {'flatten': ['device'], 'transform': ['mdr_text'], 'first_element': ['patient']}
        """
        return self._flatten.get(category, {})

    # ==================== filtering 설정 ====================

    def get_filtering_step(self, key: str) -> Dict[str, dict]:
        """특정 filter step key가 enabled된 category와 step 설정 반환

        Args:
            key: 필터 스텝명 (e.g. 'DEDUP', 'SCOPING', 'QUALITY_FILTER')

        Returns:
            standalone: {category: {'type': 'standalone', 'where': [...], 'qualify': [...]}}
            chain:      {category: {'type': 'chain', 'ctes': [...], 'final': str}}
            enabled=False인 카테고리는 제외

        Example:
            scfg.get_filtering_step('DEDUP')
            # {'event': {'type': 'standalone', 'qualify': [...]}, 'udi': {...}}
            scfg.get_filtering_step('QUALITY_FILTER')
            # {'event': {'type': 'chain', 'ctes': [...], 'final': 'logical'}}
        """
        result = {}
        for category, steps in self._filtering.items():
            if not isinstance(steps, dict):
                continue
            step = steps.get(key)
            if not step or not step.get('enabled', False):
                continue
            step_type = step.get('type', 'standalone')
            if step_type == 'chain':
                result[category] = {
                    'type': 'chain',
                    'ctes': step['ctes'],
                    'final': step['final'],
                }
            else:
                result[category] = {
                    'type': 'standalone',
                    **{k: v for k, v in step.items() if k in ('where', 'qualify')},
                }
        return result

    # ==================== silver metadata 설정 ====================

    def get_silver_primary_key(self, category: str) -> List[str]:
        """Silver CURRENT 테이블의 MERGE primary key 컬럼 목록 반환

        storage.yaml transform.tables.{CATEGORY}_CURRENT.primary_key 에서 읽습니다.

        Args:
            category: 카테고리명 (e.g. 'event', 'udi')

        Returns:
            primary key 컬럼명 목록 (e.g. ['MDR_REPORT_KEY', 'UDI_DI', 'SOURCE_BATCH_ID'])
        """
        table = f'{category}_CURRENT'.upper()
        pk = self._storage['snowflake']['transform']['tables'][table]['primary_key']
        return [pk] if isinstance(pk, str) else list(pk)

    def get_silver_business_key(self, category: str) -> Union[str, List[str]]:
        """Silver CURRENT 테이블의 비즈니스 기준 키 컬럼명 반환

        Silver 전처리 후 확정된 키를 사용합니다 (storage.yaml transform.tables).
        Bronze의 business_primary_key와 다를 수 있습니다 (예: UDI_DI 추가 후 복합 키).

        Args:
            category: 카테고리명 (e.g. 'event', 'udi')

        Returns:
            단일 키: str (e.g. 'mdr_report_key')
            복합 키: List[str] (e.g. ['MDR_REPORT_KEY', 'UDI_DI'])
        """
        table = f'{category}_CURRENT'.upper()
        return self._storage['snowflake']['transform']['tables'][table]['business_primary_key']

    # ==================== llm_extraction 설정 ====================

    def get_llm_model_config(self) -> dict:
        """LLM 모델 설정 반환 (path, tensor_parallel_size, ...)"""
        return self._llm_extraction['model']

    def get_llm_sampling_config(self) -> dict:
        """LLM 샘플링 파라미터 반환 (temperature, max_tokens, top_p)"""
        return self._llm_extraction['sampling']

    def get_llm_prompt_mode(self) -> str:
        """프롬프트 모드 반환 (general | sample)"""
        return self._llm_extraction['prompt_mode']

    def get_llm_checkpoint_config(self) -> dict:
        """체크포인트 설정 반환 (interval, prefix)"""
        return self._llm_extraction['checkpoint']

    def get_llm_source_category(self) -> str:
        """LLM 추출 대상 카테고리 반환 (e.g. 'event')"""
        return self._llm_extraction['source']['category']

    def get_llm_source_columns(self) -> List[str]:
        """추출 대상 컬럼 목록 반환 (e.g. ['MDR_TEXT', 'PRODUCT_PROBLEMS'])"""
        return self._llm_extraction['source']['columns']

    def get_llm_source_where(self) -> Optional[str]:
        """추출 WHERE 절 반환 (없으면 None)"""
        return self._llm_extraction['source'].get('where')

    def get_llm_extracted_suffix(self) -> str:
        """추출 결과 테이블 suffix 반환 (e.g. '_EXTRACTED')"""
        return self._llm_extraction['extracted']['suffix']

    def get_llm_extracted_columns(self) -> List[dict]:
        """추출 결과 테이블 컬럼 스키마 반환"""
        return self._llm_extraction['extracted']['columns']

    def get_llm_extracted_pk_column(self) -> str:
        """추출 결과 테이블의 primary key 컬럼명 반환"""
        for col in self._llm_extraction['extracted']['columns']:
            if col.get('primary_key'):
                return col['name']
        raise ValueError("llm_extraction.yaml: extracted.columns에 primary_key가 없습니다")

    def get_llm_extracted_non_pk_columns(self) -> List[str]:
        """추출 결과 테이블의 pk 외 컬럼명 목록 반환"""
        return [
            col['name']
            for col in self._llm_extraction['extracted']['columns']
            if not col.get('primary_key')
        ]


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

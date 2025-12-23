# src/preprocess/config.py (전처리 전용 - 고수준)
from pathlib import Path
from typing import Dict, Any, List, Optional
from config.config_loader import load_config

class PreprocessConfig:
    """전처리 설정 통합 관리 클래스
    
    IDE 자동완성을 지원하며 타입 안전성을 제공합니다.
    """
    
    def __init__(self):
        # 기본 설정들 (캐싱됨)
        self._base = load_config("base")
        self._storage = load_config("storage")
        self._pipeline = load_config("pipeline")
        
        # 전처리 설정들 (캐싱됨)
        self._columns = load_config("preprocess/columns")
        self._filtering = load_config("preprocess/filtering")
        self._cleaning = load_config("preprocess/cleaning")
        self._legacy_cleaning = load_config("preprocess/legacy_cleaning")  # UDI 정제
        self._deduplication = load_config("preprocess/deduplication")
        self._udi_matching = load_config("preprocess/udi_matching")
        self._transformation = load_config("preprocess/transformation")
        self._llm_extraction = load_config("preprocess/llm_extraction")
    
    # ==================== 기본 설정 ====================
    
    @property
    def base(self) -> Dict[Any, Any]:
        """기본 설정"""
        return self._base
    
    @property
    def storage(self) -> Dict[Any, Any]:
        """저장소 설정 (S3, Snowflake)"""
        return self._storage
    
    @property
    def pipeline(self) -> Dict[Any, Any]:
        """파이프라인 설정"""
        return self._pipeline
    
    # ==================== 전처리 설정 ====================
    
    @property
    def cleaning(self) -> Dict[Any, Any]:
        """클린징 설정"""
        return self._cleaning
    
    @property
    def legacy_cleaning(self) -> Dict[Any, Any]:
        """정제 설정 (UDI 매칭용 레거시)"""
        return self._legacy_cleaning
    
    @property
    def deduplication(self) -> Dict[Any, Any]:
        """중복 제거 설정"""
        return self._deduplication
    
    @property
    def udi_matching(self) -> Dict[Any, Any]:
        """UDI 매칭 설정"""
        return self._udi_matching
    
    @property
    def filtering(self) -> Dict[Any, Any]:
        """품질 필터링 설정"""
        return self._filtering
    
    @property
    def transformation(self) -> Dict[Any, Any]:
        """타입 변환 설정"""
        return self._transformation
    
    @property
    def columns(self) -> Dict[Any, Any]:
        """컬럼 선택/제거 설정"""
        return self._columns
    
    @property
    def llm_extraction(self) -> Dict[Any, Any]:
        """LLM 텍스트 추출 설정"""
        return self._llm_extraction
    
    # ==================== 편의 메서드 ====================
    
    def get_na_patterns(self) -> List[str]:
        """NA 패턴 목록 반환"""
        return self._cleaning['na_patterns']['patterns']
    
    def get_dedup_columns(self) -> List[str]:
        """중복 제거 기준 컬럼 반환"""
        return self._deduplication['maude']['subset_columns']
    
    def get_udi_strategies(self) -> List[Dict]:
        """UDI 매칭 전략 반환"""
        return self._udi_matching['strategies']
    
    def get_drop_patterns(self, stage: str = '1st') -> List[str]:
        """컬럼 Drop 패턴 반환
        
        Args:
            stage: '1st' 또는 '2nd'
        """
        key = f'column_drop_{stage}'
        return self._columns[key]['drop_patterns']
    
    def get_final_columns(self) -> List[str]:
        """최종 유지할 컬럼 목록 반환"""
        return self._columns['column_drop_2nd']['keep_columns']
    
    def get_fuzzy_threshold(self) -> int:
        """퍼지 매칭 임계값 반환"""
        return self._udi_matching['fuzzy_matching']['threshold']
    
    def is_enabled(self, feature: str) -> bool:
        """특정 기능 활성화 여부 확인
        
        Args:
            feature: 'deduplication', 'udi_matching' 등
        """
        config_map = {
            'deduplication': self._deduplication['maude']['enabled'],
            'udi_matching': self._udi_matching['strategies'][0]['enabled'],
        }
        return config_map.get(feature, False)
    
    # ==================== legacy_cleaning (UDI 정제) 관련 ====================
    
    def get_legacy_cleaning_fuzzy_threshold(self) -> int:
        """정제용 퍼지 매칭 임계값"""
        return self._legacy_cleaning['fuzzy_matching']['threshold']
    
    def get_low_compliance_threshold(self) -> float:
        """저준수 제조사 기준"""
        return self._legacy_cleaning['manufacturer_compliance']['low_compliance_threshold']
    
    def get_confidence_level(self, match_type: str) -> str:
        """신뢰도 레벨 반환
        
        Args:
            match_type: 'udi_direct', 'udi_secondary', 'meta_match' 등
        """
        return self._legacy_cleaning['confidence']['levels'].get(match_type, 'UNKNOWN')
    
    def get_maude_date_priority(self) -> List[str]:
        """MAUDE 날짜 필드 우선순위"""
        return self._legacy_cleaning['date_fields']['maude_priority']
    
    def get_udi_date_fields(self) -> List[str]:
        """UDI 날짜 필드"""
        return self._legacy_cleaning['date_fields']['udi']
    
    def get_full_group_columns(self) -> List[str]:
        """그룹화 컬럼"""
        return self._legacy_cleaning['grouping']['full_group']
    
    def get_join_key_column(self) -> str:
        """조인 키 컬럼"""
        return self._legacy_cleaning['joining']['key_column']
    
    def should_cleanup_temp(self, on_success: bool = True) -> bool:
        """임시 파일 정리 여부
        
        Args:
            on_success: True면 성공 시, False면 에러 시
        """
        key = 'on_success' if on_success else 'on_error'
        return self._legacy_cleaning['temp_files']['cleanup'][key]
    
    # ==================== LLM Extraction 관련 ====================
    
    def get_llm_model_name(self) -> str:
        """LLM 모델 이름"""
        return self._llm_extraction['model']['name']
    
    def get_llm_vllm_config(self) -> Dict[str, Any]:
        """vLLM 엔진 설정"""
        return self._llm_extraction['model']['vllm']
    
    def get_extraction_columns(self, include_reasoning: bool = False) -> List[str]:
        """추출할 컬럼 목록
        
        Args:
            include_reasoning: True면 추론 필드도 포함
        """
        cols = self._llm_extraction['extraction']['base_fields'].copy()
        
        if include_reasoning:
            suffix = self._llm_extraction['extraction']['reasoning_fields']['suffix']
            reason_cols = [col + suffix for col in cols]
            cols.extend(reason_cols)
        
        return cols
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """체크포인트 설정"""
        return self._llm_extraction['checkpoint']
    
    def get_llm_sampling_config(self) -> Dict[str, Any]:
        """샘플링 설정"""
        return self._llm_extraction['input']['sampling']
    
    # ==================== 경로 관련 ====================
    
    def get_path(self, stage: str, dataset: str = 'maude') -> Path:
        """데이터 경로 반환
        
        Args:
            stage: 'bronze', 'silver', 'gold'
            dataset: 'maude', 'udi'
            
        Returns:
            Path 객체
        """
        use_s3 = self._base['paths']['use_s3']
        
        if use_s3:
            base = self._storage['s3']['paths'][stage]
        else:
            base = self._base['paths']['local'][stage]
        
        filename = self._base['datasets'][dataset][f'{stage}_file']
        return Path(base) / filename
    
    def get_temp_dir(self) -> Path:
        """임시 디렉토리 경로"""
        return Path(self._base['paths']['local']['temp'])
    
    # ==================== 디버그/개발 ====================
    
    def print_config(self, config_name: Optional[str] = None):
        """설정 출력 (디버깅용)
        
        Args:
            config_name: None이면 전체, 'cleaning' 등 특정 설정명
        """
        import json
        
        if config_name is None:
            configs = {
                'base': self._base,
                'columns': self._columns,
                'filtering': self._filtering,
                'cleaning': self._cleaning,
                'legacy_cleaning': self._legacy_cleaning,
                'deduplication': self._deduplication,
                'udi_matching': self._udi_matching,
                'transformation': self._transformation,
                'llm_extraction': self._llm_extraction,
            }
        else:
            configs = {config_name: getattr(self, f'_{config_name}')}
        
        print(json.dumps(configs, indent=2, ensure_ascii=False, default=str))


# 싱글톤 인스턴스 (선택적)
_config = None

def get_config() -> PreprocessConfig:
    """전역 설정 인스턴스 반환 (싱글톤)"""
    global _config
    if _config is None:
        _config = PreprocessConfig()
    return _config
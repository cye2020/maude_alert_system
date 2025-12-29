from pathlib import Path
from typing import Dict, Any, Optional
from config.config_loader import load_config

class DashboardConfig:
    """대시보드 설정 통합 관리 클래스
    """
    
    def __init__(self):
        # 기본 설정들 (캐싱됨)
        self._base = load_config("base")
        self._storage = load_config("storage")
        self._pipeline = load_config("pipeline")

        # 대시보드 설정들 (캐싱됨)
        self._sidebar = load_config("dashboard/sidebar")
        self._defaults = load_config("dashboard/defaults")
        self._ui_standards = load_config("dashboard/ui_standards")
    
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
    def sidebar(self) -> Dict[Any, Any]:
        """사이드바 설정"""
        return self._sidebar
    
    @property
    def defaults(self) -> Dict[Any, Any]:
        """기본 설정"""
        return self._defaults

    @property
    def ui_standards(self) -> Dict[Any, Any]:
        """UI 표준화 설정"""
        return self._ui_standards

    # ==================== 편의 메서드 ====================
    
    # def get_na_patterns(self) -> List[str]:
    #     """NA 패턴 목록 반환"""
    #     return self._cleaning['na_patterns']['patterns']
    
    # ==================== 경로 관련 ====================
    
    def get_path(self, stage: str, dataset: str = 'maude', silver_stage: Optional[str] = None) -> Path:
        """데이터 경로 반환

        Args:
            stage: 'bronze', 'silver', 'gold'
            dataset: 'maude', 'udi'
            silver_stage: Silver 계층의 경우 'stage1_basic_cleaning', 'stage2_text_processing', 'stage3_clustering' 중 선택

        Returns:
            Path 객체
        """
        use_s3 = self._base['paths']['use_s3']

        if use_s3:
            base = self._storage['s3']['paths'][stage]
        else:
            base = self._base['paths']['local'][stage]

        # Silver 계층이고 silver_stage가 지정된 경우
        if stage == 'silver' and silver_stage and isinstance(self._base['datasets'][dataset].get('silver'), dict):
            filename = self._base['datasets'][dataset]['silver'][silver_stage]
        else:
            # 기존 방식 (bronze, gold, 또는 udi의 silver)
            filename = self._base['datasets'][dataset][f'{stage}_file']

        return Path(base) / filename

    def get_silver_stage3_path(self, dataset: str = 'maude') -> Path:
        """Silver Stage3 (클러스터링) 데이터 경로 반환 - 편의 메서드

        Args:
            dataset: 'maude', 'udi'

        Returns:
            Path 객체
        """
        return self.get_path('silver', dataset, silver_stage='stage3_clustering')
    
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
                'storage': self._storage,
                'pipeline': self._pipeline,
                'sidebar': self._sidebar,
                'defaults': self._defaults,
                'ui_standards': self._ui_standards,
            }
        else:
            configs = {config_name: getattr(self, f'_{config_name}')}
        
        print(json.dumps(configs, indent=2, ensure_ascii=False, default=str))


# 싱글톤 인스턴스 (선택적)
_config = None

def get_config() -> DashboardConfig:
    """전역 설정 인스턴스 반환 (싱글톤)"""
    global _config
    if _config is None:
        _config = DashboardConfig()
    return _config
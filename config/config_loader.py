# config/config_loader.py (범용 로더 - 저수준)
import yaml
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any
from copy import deepcopy


class ConfigLoader:
    """범용 YAML 설정 로더 (싱글톤) - base config 상속 지원"""
    
    def __init__(self) -> None:
        self.project_root = self._find_project_root()
        self.config_dir = self.project_root / 'config'
        self._base_cache = {}  # base config 캐시
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 자동 탐색"""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / 'requirements.txt').exists():
                return parent
        return current.parent.parent
    
    @lru_cache(maxsize=32)
    def load(self, config_name: str) -> Dict[Any, Any]:
        """YAML 파일 로드 및 캐싱
        
        Args:
            config_name: 'base', 'preprocess/cleaning' 등
            
        Returns:
            파싱된 설정 딕셔너리
        """
        config_path = self.config_dir / f'{config_name}.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f'Config not found: {config_path}')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # base config 상속 처리
        if '_base' in config:
            config = self._merge_with_base(config, config_path)
        
        return self._replace_env_vars(config)
    
    def _merge_with_base(self, config: Dict[Any, Any], current_path: Path) -> Dict[Any, Any]:
        """base config와 병합
        
        Args:
            config: 현재 config
            current_path: 현재 config 파일 경로
            
        Returns:
            병합된 config
        """
        base_ref = config.pop('_base')
        
        # 상대 경로 해석
        if base_ref.startswith('../'):
            base_path = (current_path.parent / base_ref).resolve()
        else:
            base_path = self.config_dir / base_ref
        
        # base config 로드
        base_name = base_path.relative_to(self.config_dir).with_suffix('').as_posix()
        base_config = self.load(base_name)
        
        # Deep merge: base를 먼저, 현재 config로 오버라이드
        merged = self._deep_merge(deepcopy(base_config), config)
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """딕셔너리 재귀 병합
        
        Args:
            base: 기본 설정
            override: 오버라이드할 설정
            
        Returns:
            병합된 설정
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _replace_env_vars(self, config: Any) -> Any:
        """환경변수 치환 (재귀적)
        
        ${VAR_NAME} 형태를 실제 환경변수 값으로 치환
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(f"Environment variable not found: {env_var}")
            return value
        return config
    
    def get_path(self, config_name: str, *path_keys: str) -> Path:
        """편의 함수: config에서 경로 추출
        
        Args:
            config_name: config 파일명
            *path_keys: 경로 키 체인 (예: 'paths', 'local', 'bronze')
            
        Returns:
            절대 경로
            
        Example:
            >>> loader.get_path('base', 'paths', 'local', 'bronze')
            Path('/project/data/bronze')
        """
        config = self.load(config_name)
        
        value = config
        for key in path_keys:
            value = value[key]
        
        path = Path(value)
        
        # 상대 경로면 project_root 기준으로 변환
        if not path.is_absolute():
            path = self.project_root / path
        
        return path


# 싱글톤 인스턴스
_loader = ConfigLoader()


def load_config(config_name: str) -> Dict[Any, Any]:
    """함수형 인터페이스"""
    return _loader.load(config_name)


def get_config_path(*path_keys: str, config_name: str = 'base') -> Path:
    """경로 추출 편의 함수
    
    Args:
        *path_keys: 경로 키 체인
        config_name: config 파일명 (기본값: 'base')
        
    Returns:
        절대 경로
        
    Example:
        >>> get_config_path('paths', 'local', 'temp')
        Path('/project/data/temp')
    """
    return _loader.get_path(config_name, *path_keys)
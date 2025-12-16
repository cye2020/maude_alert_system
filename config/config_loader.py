import yaml
import os
from pathlib import Path
from functools import lru_cache

class ConfigLoader:
    def __init__(self) -> None:
        self.project_root = self._find_project_root()
        self.config_dir = self.project_root / 'config'
    
    def _find_project_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / 'requirements.txt').exists():
                return parent
        return current.parent.parent.parent

    @lru_cache
    def load(self, config_name: str):
        config_path = self.config_dir / f'{config_name}.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f'Config not found: {config_path}')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return self._replace_env_vars(config)
    
    def _replace_env_vars(self, config):
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2: -1]
            return os.getenv(env_var, config)
        return config
    
_config_loader = ConfigLoader()

def load_config(config_name: str):
    return _config_loader.load(config_name)
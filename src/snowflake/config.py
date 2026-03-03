"""
Snowflake 설정 관리 클래스

config_loader를 사용하여 설정을 로드하고 타입 안전성을 제공합니다.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from config.config_loader import load_config


class SnowflakeConfig:
    """Snowflake 설정 통합 관리 클래스

    IDE 자동완성을 지원하며 타입 안전성을 제공합니다.
    """

    def __init__(self, secrets_path: Optional[Path] = None):
        """
        Args:
            secrets_path: secrets/snowflake.yaml의 커스텀 경로 (선택사항)
        """
        # 기본 설정 로드 (캐싱됨)
        self._base = load_config("base")
        self._storage = load_config("storage")

        # Secrets 로드 (별도 처리)
        self._secrets = self._load_secrets(secrets_path)

    def _load_secrets(self, custom_path: Optional[Path] = None) -> Dict[str, Any]:
        """Secrets 파일을 로드합니다."""

        # 프로젝트 루트 찾기
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / 'requirements.txt').exists():
                project_root = parent
                break
        else:
            project_root = current.parent.parent

        secrets_path = custom_path or project_root / "secrets" / "snowflake.yaml"

        if not secrets_path.exists():
            raise FileNotFoundError(
                f"Secrets 파일을 찾을 수 없습니다: {secrets_path}\n"
                f"secrets/snowflake.yaml 파일에 Snowflake 인증 정보를 입력하세요."
            )

        with open(secrets_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

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
    def secrets(self) -> Dict[Any, Any]:
        """Snowflake Secrets"""
        return self._secrets

    # ==================== 연결 정보 ====================

    def get_account(self) -> str:
        """Snowflake 계정 ID"""
        return self._secrets['snowflake']['account']

    def get_user(self) -> str:
        """사용자명"""
        return self._secrets['snowflake']['user']

    def get_password(self) -> str:
        """비밀번호"""
        return self._secrets['snowflake']['password']

    def get_database(self) -> str:
        """데이터베이스명"""
        return self._secrets['snowflake']['database']

    def get_schema(self) -> str:
        """스키마명"""
        return self._secrets['snowflake']['schema']

    def get_warehouse(self) -> str:
        """웨어하우스명"""
        return self._secrets['snowflake']['warehouse']

    def get_role(self) -> Optional[str]:
        """역할 (선택사항)"""
        return self._secrets['snowflake'].get('role')

    def get_connection_params(self) -> Dict[str, Any]:
        """연결 파라미터 딕셔너리 반환"""
        sf = self._secrets['snowflake']
        conn_config = self._storage['snowflake']['connection']

        params = {
            'account': sf['account'],
            'user': sf['user'],
            'password': sf['password'],
            'database': sf['database'],
            'schema': sf['schema'],
            'warehouse': sf['warehouse'],
            'login_timeout': conn_config.get('login_timeout', 60),
            'network_timeout': conn_config.get('network_timeout', 600),
            'client_session_keep_alive': conn_config.get('client_session_keep_alive', True),
        }

        if sf.get('role'):
            params['role'] = sf['role']

        return params

    # ==================== 스테이지 및 테이블 ====================

    def get_stage_name(self) -> str:
        """스테이지명"""
        return self._secrets['snowflake']['stage']['name']

    def get_table_name(self) -> str:
        """테이블명"""
        return self._secrets['snowflake']['table']['name']

    # ==================== 업로드 설정 ====================

    def get_upload_config(self) -> Dict[str, Any]:
        """업로드 설정"""
        return self._storage['snowflake']['upload']

    def is_auto_compress(self) -> bool:
        """자동 압축 여부"""
        return self._storage['snowflake']['upload'].get('auto_compress', True)

    def get_parallel_threads(self) -> int:
        """병렬 업로드 스레드 수"""
        return self._storage['snowflake']['upload'].get('parallel', 4)

    def is_overwrite(self) -> bool:
        """기존 파일 덮어쓰기 여부"""
        return self._storage['snowflake']['upload'].get('overwrite', True)

    # ==================== COPY INTO 설정 ====================

    def get_copy_into_config(self) -> Dict[str, Any]:
        """COPY INTO 설정"""
        return self._storage['snowflake']['copy_into']

    def get_file_format_type(self) -> str:
        """파일 포맷 타입"""
        return self._storage['snowflake']['copy_into']['file_format']['type']

    def get_file_format_compression(self) -> str:
        """파일 압축 형식"""
        return self._storage['snowflake']['copy_into']['file_format'].get('compression', 'AUTO')

    def get_on_error_behavior(self) -> str:
        """에러 발생 시 동작"""
        return self._storage['snowflake']['copy_into'].get('on_error', 'ABORT_STATEMENT')

    def is_purge_enabled(self) -> bool:
        """로드 후 스테이지 파일 삭제 여부"""
        return self._storage['snowflake']['copy_into'].get('purge', False)

    def is_force_enabled(self) -> bool:
        """이미 로드된 파일 재로드 여부"""
        return self._storage['snowflake']['copy_into'].get('force', False)

    # ==================== 연결 타임아웃 ====================

    def get_timeout(self) -> int:
        """전체 타임아웃 (초)"""
        return self._storage['snowflake']['connection'].get('timeout', 300)

    def get_login_timeout(self) -> int:
        """로그인 타임아웃 (초)"""
        return self._storage['snowflake']['connection'].get('login_timeout', 60)

    def get_network_timeout(self) -> int:
        """네트워크 타임아웃 (초)"""
        return self._storage['snowflake']['connection'].get('network_timeout', 60)

    # ==================== 경로 관련 ====================

    def get_data_file_path(self) -> Path:
        """로컬 데이터 파일 경로"""
        path_str = self._storage['paths']['local']['data_file']
        path = Path(path_str)

        # 상대 경로면 project_root 기준으로 변환
        if not path.is_absolute():
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / 'requirements.txt').exists():
                    project_root = parent
                    break
            else:
                project_root = current.parent.parent

            path = project_root / path

        return path

    # ==================== 테이블 설정 ====================

    def get_table_config(self, table_key: str = 'maude_clustered') -> Dict[str, Any]:
        """테이블 설정 반환

        Args:
            table_key: 테이블 키 (기본값: 'maude_clustered')
        """
        return self._storage['snowflake']['tables'].get(table_key, {})

    def is_auto_create_table(self, table_key: str = 'maude_clustered') -> bool:
        """테이블 자동 생성 여부"""
        table_config = self.get_table_config(table_key)
        return table_config.get('create_if_not_exists', True)

    def is_auto_infer_schema(self, table_key: str = 'maude_clustered') -> bool:
        """스키마 자동 추론 여부"""
        table_config = self.get_table_config(table_key)
        schema_config = table_config.get('schema', {})
        return schema_config.get('auto_infer', True)

    # ==================== 디버그/개발 ====================

    def print_config(self, include_secrets: bool = False):
        """설정 출력 (디버깅용)

        Args:
            include_secrets: True면 민감한 정보도 출력 (주의!)
        """
        import json

        configs = {
            'storage': self._storage,
        }

        if include_secrets:
            configs['secrets'] = self._secrets
        else:
            # 민감한 정보는 마스킹
            masked_secrets = {
                'snowflake': {
                    'account': self._secrets['snowflake']['account'],
                    'user': self._secrets['snowflake']['user'],
                    'password': '***MASKED***',
                    'database': self._secrets['snowflake']['database'],
                    'schema': self._secrets['snowflake']['schema'],
                    'warehouse': self._secrets['snowflake']['warehouse'],
                    'role': self._secrets['snowflake'].get('role', 'N/A'),
                    'stage': self._secrets['snowflake']['stage'],
                    'table': self._secrets['snowflake']['table'],
                }
            }
            configs['secrets'] = masked_secrets

        print(json.dumps(configs, indent=2, ensure_ascii=False, default=str))


# 싱글톤 인스턴스 (선택적)
_config: Optional[SnowflakeConfig] = None


def get_snowflake_config() -> SnowflakeConfig:
    """전역 Snowflake 설정 인스턴스 반환 (싱글톤)"""
    global _config
    if _config is None:
        _config = SnowflakeConfig()
    return _config

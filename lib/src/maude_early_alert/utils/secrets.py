# 표준 라이브러리
import json

# 서드파티 라이브러리
import boto3
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig

_cache_config = SecretCacheConfig(
    max_cache_size=1024,
    secret_refresh_interval=3600,
    default_version_stage='AWSCURRENT'
)

# 리전별 SecretCache 싱글턴 (함수 내부에서 매번 생성하면 캐싱 무효)
_caches: dict = {}


def _get_cache(region_name: str) -> SecretCache:
    if region_name not in _caches:
        _caches[region_name] = SecretCache(
            config=_cache_config,
            client=boto3.client('secretsmanager', region_name=region_name)
        )
    return _caches[region_name]


def get_secret(secret_name, region_name='ap-northeast-2'):
    """AWS Secrets Manager에서 Secret을 조회 (리전별 캐싱)

    Args:
        secret_name (str): Secrets Manager에 저장된 Secret 이름
        region_name (str, optional): AWS 리전. 기본값은 'ap-northeast-2'

    Returns:
        dict: Secret 값 (JSON 파싱된 딕셔너리)

    Raises:
        ClientError: AWS API 호출 실패 시
    """
    cache = _get_cache(region_name)
    secret_json = cache.get_secret_string(secret_name)
    return json.loads(secret_json)

# 사용 예시
if __name__ == '__main__':
    try:
        creds = get_secret('snowflake/prod/credentials')
        print(creds)
        print(f"Conn ID: {creds['conn_id']}")
        # password는 로그에 출력하지 않음!

    except Exception as e:
        print(f"Error: {e}")

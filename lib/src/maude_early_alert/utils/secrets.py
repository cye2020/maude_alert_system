import boto3
import json
from botocore.exceptions import ClientError
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig

# 캐시 설정
config = SecretCacheConfig(
    max_cache_size=1024,  # 최대 캐시 항목 수
    secret_refresh_interval=3600,  # TTL (초)
    default_version_stage='AWSCURRENT'
)

def get_secret(secret_name, region_name='ap-northeast-2'):
    """
    Secrets Manager에서 Secret 조회

    Args:
        secret_name: Secret 이름
        region_name: AWS 리전

    Returns:
        dict: Secret 값 (JSON 파싱됨)

    Raises:
        ClientError: API 호출 실패 시
    """
    cache = SecretCache(config=config, client=boto3.client('secretsmanager', region_name=region_name))

    try:
        # 사용
        secret_json = cache.get_secret_string(secret_name)
    except ClientError as e:
        error_code = e.response['Error']['Code']

        if error_code == 'ResourceNotFoundException':
            raise ValueError(f"Secret '{secret_name}'을 찾을 수 없습니다")
        elif error_code == 'InvalidRequestException':
            raise ValueError("요청이 유효하지 않습니다")
        elif error_code == 'InvalidParameterException':
            raise ValueError("파라미터가 유효하지 않습니다")
        elif error_code == 'DecryptionFailure':
            raise RuntimeError("복호화 실패")
        elif error_code == 'InternalServiceError':
            raise RuntimeError("AWS 내부 오류")
        else:
            raise

    # Secret 값 파싱
    secret = json.loads(secret_json)
    return secret


# 사용 예시
if __name__ == '__main__':
    try:
        creds = get_secret('snowflake/prod/credentials')
        print(creds)
        print(f"Conn ID: {creds['conn_id']}")
        # password는 로그에 출력하지 않음!

    except Exception as e:
        print(f"Error: {e}")

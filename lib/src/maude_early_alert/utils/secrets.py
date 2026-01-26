# 표준 라이브러리
import json

# 서드파티 라이브러리
import boto3
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig
from botocore.exceptions import ClientError


# ============================================================
# Secret Cache 전역 설정
# ============================================================
# 캐시 설정: 성능 향상 및 API 호출 비용 절감을 위한 캐싱 전략
config = SecretCacheConfig(
    max_cache_size=1024,  # 최대 캐시 항목 수 (메모리 사용량 제한)
    secret_refresh_interval=3600,  # TTL (Time To Live): 1시간 (초 단위)
    default_version_stage='AWSCURRENT'  # 기본 버전 스테이지: 현재 활성화된 Secret 사용
)


def get_secret(secret_name, region_name='ap-northeast-2'):
    """
    AWS Secrets Manager에서 Secret을 조회하고 캐싱
    
    이 함수는 캐시를 사용하여 동일한 Secret에 대한 반복 조회 시 
    API 호출을 최소화하고 성능을 향상시킵니다.
    
    Args:
        secret_name (str): Secrets Manager에 저장된 Secret 이름
        region_name (str, optional): AWS 리전. 기본값은 'ap-northeast-2' (서울)
    
    Returns:
        dict: Secret 값 (JSON 형식으로 파싱된 딕셔너리)
    
    Raises:
        ValueError: Secret을 찾을 수 없거나 요청/파라미터가 유효하지 않은 경우
        RuntimeError: 복호화 실패 또는 AWS 내부 오류 발생 시
        ClientError: 그 외 AWS API 호출 실패 시
    
    Examples:
        >>> secret = get_secret('my-database-credentials')
        >>> print(secret['username'])
        'admin'
    """
    # Secret Cache 초기화 (지정된 리전의 Secrets Manager 클라이언트 사용)
    cache = SecretCache(
        config=config, 
        client=boto3.client('secretsmanager', region_name=region_name)
    )

    try:
        # 캐시에서 Secret 조회 (캐시에 없으면 API 호출 후 캐싱)
        secret_json = cache.get_secret_string(secret_name)
        
    except ClientError as e:
        # AWS API 에러 코드 추출
        error_code = e.response['Error']['Code']

        # ================================================
        # 에러 타입별 처리 및 사용자 친화적 메시지 제공
        # ================================================
        
        if error_code == 'ResourceNotFoundException':
            # Secret이 존재하지 않는 경우
            raise ValueError(f"Secret '{secret_name}'을 찾을 수 없습니다")
        
        elif error_code == 'InvalidRequestException':
            # 잘못된 요청 형식
            raise ValueError("요청이 유효하지 않습니다")
        
        elif error_code == 'InvalidParameterException':
            # 유효하지 않은 파라미터 값
            raise ValueError("파라미터가 유효하지 않습니다")
        
        elif error_code == 'DecryptionFailure':
            # KMS 복호화 실패 (권한 또는 키 문제)
            raise RuntimeError("복호화 실패")
        
        elif error_code == 'InternalServiceError':
            # AWS 서비스 내부 오류 (일시적 장애)
            raise RuntimeError("AWS 내부 오류")
        
        else:
            # 그 외 예상치 못한 에러는 원본 예외 발생
            raise

    # JSON 문자열을 Python 딕셔너리로 파싱
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

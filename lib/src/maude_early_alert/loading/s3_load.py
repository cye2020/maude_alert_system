# 표준 라이브러리
import re
import gzip
import json
import math
import zipfile
import io
from typing import Dict, List

# 서드파티 라이브러리
import requests


class S3Loader:
    """
    FDA Open API에서 device 데이터를 다운로드하여 S3에 업로드하는 클래스
    
    주요 기능:
    - FDA API에서 메타데이터 조회
    - 파일 정보 추출 및 필터링
    - S3 키 생성
    - S3 버킷에 파일 업로드
    """

    # FDA 다운로드 베이스 URL
    BASE_URL = 'https://download.open.fda.gov/'

    # Snowflake VARIANT 최대 크기 128MB, 안전 마진 적용
    MAX_VARIANT_BYTES = 100 * 1024 * 1024  # 100MB

    def __init__(self, bucket_name: str, client=None):
        """
        S3Loader 초기화
        
        Args:
            bucket_name (str): 업로드할 S3 버킷 이름
            client: boto3 S3 클라이언트 객체 (선택사항)
        """
        self.url = 'https://api.fda.gov/download.json'
        self.bucket_name = bucket_name
        self.s3_client = client
        
        # 메타데이터 캐싱용 변수
        self.metadata = None

    def fetch_metadata(self) -> Dict:
        """
        FDA API에서 메타데이터 가져오기
        
        Returns:
            Dict: FDA API 응답 메타데이터 (캐싱됨)
        
        Raises:
            requests.HTTPError: API 요청 실패 시
        """
        # 이미 조회한 메타데이터가 있으면 재사용
        if self.metadata is None:
            response = requests.get(self.url)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
            self.metadata = response.json()
        return self.metadata

    def extract(
        self, category: str, 
        start: int = None, end: int = None,
        logical_date: str = None
    ) -> List[Dict[str, str]]:
        """
        특정 카테고리의 파일 정보 추출
        
        Args:
            category (str): 데이터 카테고리 ('udi' 또는 'event')
            start (int, optional): 시작 연도 (event 카테고리에만 적용)
            end (int, optional): 종료 연도 (event 카테고리에만 적용)
        
        Returns:
            List[Dict[str, str]]: 파일 정보 리스트
                - url: 다운로드 URL
                - display_name: 표시 이름
                - s3_key: S3 저장 경로
        
        Raises:
            ValueError: 잘못된 카테고리인 경우
        
        Examples:
            event URL 형식: 
                https://download.open.fda.gov/device/event/{YEAR}q{QUARTER}/device-event-{PART}-of-{TOTAL}.json.zip
            udi URL 형식:
                https://download.open.fda.gov/device/device/udi/device-udi-{PART}-of-{TOTAL}.json.zip
        """
        # 카테고리 유효성 검증
        if category not in ['udi', 'event']:
            raise ValueError(f"Invalid category: {category}")

        # API에서 메타데이터 조회
        metadata = self.fetch_metadata()

        # device 데이터 추출
        device_data = metadata['results']['device']

        # 해당 카테고리에 partitions 데이터가 있는지 확인
        if category not in device_data or 'partitions' not in device_data[category]:
            return []

        files = []

        # 각 파티션(파일) 정보 처리
        for partition in device_data[category]['partitions']:
            # 파일 다운로드 URL 가져오기
            url = partition.get('file', '')

            # URL에 올바른 카테고리 경로가 포함되어 있는지 검증
            if f'device/{category}/' not in url:
                continue
        
            # ================================================
            # event 카테고리: 연도 기반 필터링
            # ================================================
            if category == 'event':
                # URL에서 연도 추출 (예: /2020q1/ -> 2020)
                match = re.search(r'/(\d{4})q\d+/', url)
                year = int(match.group(1)) if match else None
                
                # 연도를 추출하지 못한 경우 건너뛰기
                if year is None:
                    continue
                
                # 시작 연도 필터링
                if start and year < start:
                    continue
                
                # 종료 연도 필터링
                if end and year > end:
                    continue
            
            # S3 저장 경로 생성
            s3_key = self.s3_key_generate(url, logical_date)

            # 파일 정보 추가
            files.append({
                'url': url, 
                'display_name': partition.get('display_name'),
                's3_key': s3_key
            })
            
        return files

    def s3_key_generate(self, url: str, logical_date: str = None) -> str:
        """
        S3 저장 키(경로) 생성

        Args:
            url (str): 파일 다운로드 URL
            logical_date (str, optional): 논리적 날짜 (prefix로 사용)

        Returns:
            str: S3 키 (예: device/event/2020q1/device-event-0001-of-0010.json.gz)
        """
        # 날짜 prefix 설정
        prefix = ''
        if logical_date:
            prefix = logical_date + '/'

        # BASE_URL을 제거하여 상대 경로만 추출 후 prefix 추가
        s3_key = prefix + url.replace(self.BASE_URL, '')

        # .json.zip -> .json.gz 변환 (Snowflake 호환)
        s3_key = s3_key.replace('.json.zip', '.json.gz')
        return s3_key

    def _upload_gzip(self, s3_key: str, content: bytes) -> None:
        """바이트 데이터를 GZIP 압축 후 S3에 업로드

        Args:
            s3_key (str): S3 저장 경로
            content (bytes): 업로드할 바이트 데이터
        """
        gzip_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gzip_buffer, mode='wb') as gz:
            gz.write(content)
        gzip_buffer.seek(0)

        self.s3_client.upload_fileobj(
            gzip_buffer,
            self.bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'application/gzip'}
        )

    def load(self, s3_key: str, file_url: str) -> bool:
        """
        파일을 다운로드하여 S3에 업로드 (ZIP -> GZIP 변환)

        JSON 크기가 Snowflake VARIANT 한도(128MB)를 초과하면
        results 배열을 분할하여 여러 파일로 업로드

        Args:
            s3_key (str): S3 저장 경로
            file_url (str): 다운로드할 파일 URL

        Returns:
            bool: 업로드 성공 여부

        Note:
            ZIP 파일은 중앙 디렉토리가 파일 끝에 있어 스트리밍 해제 불가.
            다운로드는 청크 단위, 업로드는 스트리밍 방식 사용.
        """
        try:
            # 청크 단위로 ZIP 다운로드 (메모리 효율적)
            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            zip_buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                zip_buffer.write(chunk)
            zip_buffer.seek(0)

            # ZIP 압축 해제
            with zipfile.ZipFile(zip_buffer, 'r') as zf:
                json_filename = zf.namelist()[0]
                json_content = zf.read(json_filename)

            # JSON 크기가 VARIANT 한도 이하이면 그대로 업로드
            if len(json_content) <= self.MAX_VARIANT_BYTES:
                self._upload_gzip(s3_key, json_content)
                return True

            # 한도 초과 시 results 배열 분할 업로드
            json_data = json.loads(json_content)
            results = json_data.get('results', [])

            if not results:
                self._upload_gzip(s3_key, json_content)
                return True

            num_chunks = math.ceil(len(json_content) / self.MAX_VARIANT_BYTES)
            chunk_size = math.ceil(len(results) / num_chunks)

            for i in range(num_chunks):
                chunk_results = results[i * chunk_size : (i + 1) * chunk_size]
                chunk_json = json.dumps({"results": chunk_results}).encode('utf-8')
                chunk_s3_key = s3_key.replace('.json.gz', f'_part{i+1}.json.gz')
                self._upload_gzip(chunk_s3_key, chunk_json)

            return True

        except requests.RequestException as e:
            print(f"다운로드 실패: [{file_url}] - {e}")
            return False

        except zipfile.BadZipFile as e:
            print(f"ZIP 파일 오류: [{file_url}] - {e}")
            return False

        except Exception as e:
            print(f"업로드 오류 [{s3_key}] - {e}")
            return False


if __name__=='__main__':
    import boto3
    bucket_name= 'amazon-s3-fda' 
    client = boto3.client('s3')

    s3_loader = S3Loader(bucket_name, client)
    
    import pendulum
    category = 'event'
    end = pendulum.now().year
    start = 2020
    
    files = s3_loader.extract(category, start, end, pendulum.now().strftime('%Y%m'))
    print(files)
    
    import structlog
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)
    for file in files:
        s3_key = file['s3_key']
        file_url = file['url']
        is_successed = s3_loader.load(s3_key, file_url)
        logger.info(f'Load Result', s3_key=s3_key, result=is_successed)
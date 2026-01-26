# 표준 라이브러리
import gzip
import json
import math
import zipfile
import io

# 서드파티 라이브러리
import requests
import structlog


class S3Loader:
    """
    FDA 데이터를 다운로드하여 S3에 업로드하는 클래스

    주요 기능:
    - S3 키 생성
    - 파일 다운로드 (ZIP → GZIP 변환)
    - S3 버킷에 파일 업로드
    """

    # FDA 다운로드 베이스 URL
    BASE_URL = 'https://download.open.fda.gov/'

    # Snowflake VARIANT 최대 크기 128MB, 안전 마진 적용
    MAX_VARIANT_BYTES = 100 * 1024 * 1024  # 100MB

    def __init__(
        self, bucket_name: str,
        client, session: requests.Session = None
    ):
        """
        Args:
            bucket_name: 업로드할 S3 버킷 이름
            client: boto3 S3 클라이언트 (필수)
            session: requests.Session (테스트 시 모킹 가능, 미지정 시 기본 Session 사용)
        """
        self.bucket_name = bucket_name
        self.s3_client = client
        self.session = session or requests.Session()
        self.logger = structlog.get_logger(__name__)

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
            response = self.session.get(file_url, stream=True)
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
            self.logger.error('다운로드 실패', file_url=file_url, error=str(e))
            return False

        except zipfile.BadZipFile as e:
            self.logger.error('ZIP 파일 오류', file_url=file_url, error=str(e))
            return False

        except Exception as e:
            self.logger.error('업로드 오류', s3_key=s3_key, error=str(e))
            return False


if __name__ == '__main__':
    import boto3
    import pendulum
    import structlog

    from maude_early_alert.loading.fda_extract import FDAExtractor

    logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

    # FDA 파일 목록 추출
    extractor = FDAExtractor()
    files = extractor.extract('event', start=2020, end=pendulum.now().year)
    print(files)

    # S3 업로드
    bucket_name = 'amazon-s3-fda'
    client = boto3.client('s3')
    s3_loader = S3Loader(bucket_name, client)
    logical_date = pendulum.now().strftime('%Y%m')

    for file in files:
        s3_key = s3_loader.s3_key_generate(file['url'], logical_date)
        result = s3_loader.load(s3_key, file['url'])
        logger.info('Load Result', s3_key=s3_key, result=result)

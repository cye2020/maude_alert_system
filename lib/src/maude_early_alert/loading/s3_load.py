from typing import Dict, List, Optional
import requests 
import re

class S3Loader:
    """
    FDA open api에서 device 데이터를 s3에 로드하는 함수들

    __init__ : s3 클라이언트 초기화
    extract : 파일정보 추출
    s3_key_generator : s3 키 생성
    load : s3에 로드
    """

    BASE_URL = 'https://download.open.fda.gov/'

    def __init__(self, bucket_name: str, client = None, aws_conn_id: str = 'aws_default'):
        self.url = 'https://api.fda.gov/download.json'
        self.bucket_name = bucket_name
        self.s3_client = client
    
        self.metadata = None

    def fetch_metadata(self) -> Dict:
        """FDA API에서 메타데이터 가져오기"""
        if self.metadata is None:
            response = requests.get(self.url)
            response.raise_for_status()
            self.metadata = response.json()
        return self.metadata
    

    def extract(self, category: str, start: int = None, end: int = None) -> List[Dict[str, str]]:
        """
        category : udi, event
        start : 시작년도
        end : 종료년도

        파일정보 
        event :https://download.open.fda.gov/device/event/{YEAR}q{QUARTER}/device-event-{PART}-of-{TOTAL}.json.zip

        udi : https://download.open.fda.gov/device/device/udi/device-udi-{PART}-of-{TOTAL}.json.zip

        url 앞에 부분만 제거하고 가져오기. 뒷부분은 파일명으로 추출 
        """
        # 카테고리 필터링
        if category not in ['udi', 'event']:
            raise ValueError(f"Invalid category: {category}")

        # api에서 추출한 메타데이터를 가져옴
        metadata = self.fetch_metadata()

        #device 데이터 추출
        device_data = metadata['results']['device']

        # 해당 카테고리 데이터 확인
        if category not in device_data or 'partitions' not in device_data[category]:
            return []

        files = []

        # 각 파일 정보 처리
        for partition in device_data[category]['partitions']:
            #파일 url 가져오기
            url = partition.get('file', '')

            #검증
            if f'device/{category}/' not in url:
                continue
        
            # ================================================
            # event 카테고리 처리
            # ================================================
            if category == 'event':
                match = re.search(r'/(\d{4})q\d+/', url)
                year = int(match.group(1)) if match else None
                # year = self.extract_year(url)
								
                if year is None:
                    continue
                if start and year < start:
                    continue
                if end and year > end:
                    continue
            
            #s3_key 생성
            s3_key = self.s3_key_generate(url)

            files.append({
                'url': url, 
                'display_name': partition.get('display_name'),
                's3_key': s3_key
            })
            
        return files

    def s3_key_generate(self, url: str, logical_date: str = None) -> str:
        """
        s3_key_generator 이름 정하기
        """
        prifix = ''
        if logical_date:
		        prefix = logical_date + '/'
        s3_key = prefix + url.replace(self.BASE_URL, '')
        return s3_key

    def load(self, client, s3_key: str, file_url: str) -> bool:
        # [TODO] 에러 핸들링
        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            #스트리밍 방식으로 업로드
            self.s3_client.upload_fileobj(
                response.raw,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'application/zip'}
                # ExtraArgs: 업로드 시 추가 정보
                # ContentType: 파일 타입
                # application/zip : zip 파일
            )
            return True
        
        except requests.RequestException as e:
            print(f"다운로드 실패: [{file_url}] - {e}")
            return False
        except Exception as e:
            print(f"업로드 오류 [{s3_key}] - {e}")
            return False

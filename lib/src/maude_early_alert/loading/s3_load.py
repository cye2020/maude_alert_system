from typing import Dict
import boto3
import requests

class S3Loader:
    def __init__(self, bucket_name: str):
        self.url = 'https://api.fda.gov/download.json'
        self.bucket_name = bucket_name
    
    def extract(self, category: str, start: int = None, end: int = None) -> Dict[str, str]:
        if start:
            # 시작 년도 필터링
            pass
        
        if end:
            # 끝 년도 필터링
            pass
        
    def load(self, client, s3_key: str, file_url: str):
        # [TODO] 에러 핸들링
        
        response = requests.get(file_url, stream=True)
        
        client.upload_fileobj(
            response.raw,
            self.bucket_name,
            s3_key
        )
    
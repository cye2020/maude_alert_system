# ==================================================
# 데이터 원본 zip 파일 스트리밍
# ==================================================

# -----------------------------
# 표준 라이브러리
# -----------------------------
from typing import Dict, Iterator
import zipfile
import io

# -----------------------------
# 서드파티 라이브러리
# -----------------------------
import requests
import ijson


class ZipStreamer:
    """ZIP 파일에서 JSON 레코드를 스트리밍"""
    
    def __init__(self, url: str, chunk_size: int = 8 * 1024 * 1024):
        self.url = url
        self.chunk_size = chunk_size
        self.filename = url.split('/')[-1]
    
    def stream_records(self) -> Iterator[Dict]:
        """URL에서 레코드를 스트리밍으로 yield"""
        # ZIP 다운로드
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            zip_buffer = io.BytesIO()
            for chunk in r.iter_content(chunk_size=self.chunk_size):
                zip_buffer.write(chunk)
            zip_buffer.seek(0)
        
        # ZIP 압축 해제 및 JSON 스트리밍
        with zipfile.ZipFile(zip_buffer, 'r') as z:
            json_file = [n for n in z.namelist() if n.endswith(".json")][0]
            with z.open(json_file) as f:
                parser = ijson.items(f, 'results.item')
                for record in parser:
                    yield record
        
        del zip_buffer
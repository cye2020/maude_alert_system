from typing import Set, Tuple

from code.loading.zip_streamer import ZipStreamer
from code.loading.flattener import Flattener

class SchemaCollector:
    """URL에서 스키마 수집"""
    
    def __init__(self):
        self.streamer = None
        self.flattener = Flattener()
    
    def collect_from_url(self, url: str) -> Tuple[Set[str], int]:
        """단일 URL에서 스키마 수집 (모든 레코드 순회)"""
        try:
            self.streamer = ZipStreamer(url)
            file_columns = set()
            record_count = 0
            
            # 모든 레코드를 순회하여 전체 스키마 수집
            for record in self.streamer.stream_records():
                columns = self.flattener.extract_columns(record)
                file_columns.update(columns)
                record_count += 1
            
            return file_columns, record_count
        
        except Exception as e:
            return set(), 0
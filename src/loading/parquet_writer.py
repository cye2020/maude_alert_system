# ==================================================
# parquet 파일로 변환
# ==================================================

# -----------------------------
# 표준 라이브러리
# -----------------------------
from typing import List, Dict

# -----------------------------
# 서드파티 라이브러리
# -----------------------------
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetWriter:
    """Parquet 파일 쓰기 (버퍼링 지원)"""
    
    def __init__(self, 
                 schema_columns: List[str],
                 output_file: str,
                 chunk_size: int = 5000):
        self.schema_columns = schema_columns
        self.schema = pa.schema([(col, pa.string()) for col in schema_columns])
        self.writer = pq.ParquetWriter(output_file, self.schema, compression='zstd')
        self.buffer = []
        self.chunk_size = chunk_size
    
    def write(self, record: Dict) -> None:
        """단일 레코드를 버퍼에 추가"""
        self.buffer.append(record)
        if len(self.buffer) >= self.chunk_size:
            self._flush()
    
    def write_table(self, table: pa.Table) -> None:
        """PyArrow Table을 직접 쓰기 (병합용)"""
        self.writer.write_table(table)
    
    def _flush(self) -> None:
        """버퍼를 파일에 쓰기"""
        if self.buffer:
            table = pa.Table.from_pylist(self.buffer, schema=self.schema)
            self.writer.write_table(table)
            self.buffer = []
    
    def close(self) -> None:
        """남은 버퍼 처리 후 파일 닫기"""
        self._flush()
        self.writer.close()
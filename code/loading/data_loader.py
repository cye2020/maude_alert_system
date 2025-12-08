# ==================================================
# 데이터 적재 함수
# ==================================================

# -----------------------------
# 표준 라이브러리
# -----------------------------
import tempfile
from typing import List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import json
import hashlib
from enum import Enum
import shutil

# -----------------------------
# 서드파티 라이브러리
# -----------------------------
import requests
from tqdm import tqdm
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
import polars as pl
import pandas as pd

# -----------------------------
# 로컬 모듈
# -----------------------------
from .zip_streamer import ZipStreamer
from .flattener import Flattener
from .parquet_writer import ParquetWriter
from .schema_collector import SchemaCollector


# -----------------------------
# Dataset 어댑터
# -----------------------------
class DatasetAdapter(Enum):
    SPARK = "spark"
    PANDAS = "pandas"
    POLARS = "polars"

class PolarsFrameType(Enum):
    LAZY_FRAME = 1
    DATA_FRAME = 2

class DataLoader:
    """FDA 데이터 전체 적재 파이프라인"""
    
    SEARCH_URL = 'https://api.fda.gov/download.json'
    
    def __init__(self, 
        start: int,
        end: int,
        output_file: str = 'output.parquet',
        schema_file: str = '.schema_cache.json',
        max_workers: int = 4,
        adapter: DatasetAdapter = DatasetAdapter.PANDAS
    ) -> None:
        self.start = start
        self.end = end
        self.output_file = output_file
        self.schema_file = schema_file
        self.max_workers = max_workers
        self.adapter = adapter
        self.urls = []
        self.schema_columns = []
    
    def search_download_url(self) -> List[str]:
        """다운로드 URL 목록 조회"""
        response = requests.get(self.SEARCH_URL).json()
        partitions = response['results']['device']['event']['partitions']
        
        urls = []
        for item in partitions:
            first = item['display_name'].split()[0]
            if first.isdigit() and self.start <= int(first) <= self.end:
                urls.append(item["file"])
        return urls
    
    def _collect_schema_worker(self, url: str) -> Tuple[str, set, int]:
        """워커 함수: 단일 URL에서 스키마 수집"""
        collector = SchemaCollector()
        return collector.collect_from_url(url)
    
    def _collect_schema(self, skip: bool = False) -> List[str]:
        """Phase 1: 병렬로 전체 스키마 수집"""
        if skip and os.path.exists(self.schema_file):
            print(f"♻️  기존 스키마 로드: {self.schema_file}")
            with open(self.schema_file, 'r') as f:
                schema_columns = json.load(f)
            print(f"✅ {len(schema_columns):,}개 컬럼 로드 완료\n")
            return schema_columns
        
        print(f"\n=== Phase 1: 전체 스키마 수집 (병렬 {self.max_workers}개) ===\n")
        
        all_columns = set()
        total_records = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._collect_schema_worker, url): url 
                      for url in self.urls}
            
            for i, future in enumerate(tqdm(as_completed(futures), 
                                            total=len(self.urls), 
                                            desc="스키마 수집"), 1):
                file_columns, record_count = future.result()
                all_columns.update(file_columns)
                total_records += record_count
        
        schema_columns = sorted(all_columns)
        
        # 스키마 저장
        with open(self.schema_file, 'w') as f:
            json.dump(schema_columns, f)
        
        print(f"\n✅ 총 {total_records:,}개 레코드, {len(schema_columns):,}개 고유 컬럼 발견")
        print(f"📝 스키마 저장: {self.schema_file}\n")
        
        return schema_columns
    
    def _convert_url_to_temp_parquet(self, 
            url: str, 
            temp_dir: str
        ) -> Tuple[str, str, int]:
        """워커 함수: 단일 URL을 임시 Parquet 파일로 변환"""
        try:
            # 고유한 파일명 생성
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{url_hash}_{url.split('/')[-1].replace('.zip', '.parquet')}"
            temp_file = os.path.join(temp_dir, filename)
            
            # 스트리밍 변환
            streamer = ZipStreamer(url)
            flattener = Flattener()
            writer = ParquetWriter(self.schema_columns, temp_file)
            
            record_count = 0
            for record in streamer.stream_records():
                normalized = flattener.normalize(record, self.schema_columns)
                writer.write(normalized)
                record_count += 1
            
            writer.close()
            return temp_file, record_count
        
        except Exception as e:
            return None, 0
    
    def _merge_parquet_files(self, temp_files: List[str]) -> None:
        """임시 Parquet 파일들을 하나로 병합"""
        print("\n📦 Parquet 파일 병합 중...")
        
        existing_files = [f for f in temp_files if os.path.exists(f)]
        
        if not existing_files:
            print("❌ 병합할 파일이 없습니다.")
            return
        
        # ParquetWriter로 병합
        writer = ParquetWriter(self.schema_columns, self.output_file)
        
        for temp_file in tqdm(existing_files, desc="병합"):
            if os.path.exists(temp_file):
                table = pq.read_table(temp_file)
                writer.write_table(table)
                
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        writer.close()
    
    def _convert_to_parquet(self) -> None:
        """Phase 2: 병렬로 Parquet 변환"""
        print(f"=== Phase 2: Parquet 변환 (병렬 {self.max_workers}개) ===\n")
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp(prefix='fda_parquet_')
        print(f"📁 임시 디렉토리: {temp_dir}\n")
        
        total_records = 0
        temp_files = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._convert_url_to_temp_parquet, url, temp_dir): url 
                for url in self.urls
            }
            
            for future in tqdm(as_completed(futures), total=len(self.urls), desc="변환"):
                temp_file, record_count = future.result()
                
                if temp_file:
                    temp_files.append(temp_file)
                    total_records += record_count
                
        # 임시 Parquet 파일들 병합
        if temp_files:
            self._merge_parquet_files(temp_files)
            
            # 임시 디렉토리 삭제
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            file_size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
            print(f"\n✅ 완료! {total_records:,}개 레코드를 {self.output_file}에 저장")
            print(f"📦 파일 크기: {file_size_mb:.2f} MB")
        else:
            print("\n❌ 변환된 파일이 없습니다.")
    
    def process(self, skip: bool = False):
        """전체 파이프라인 실행 및 데이터 로드"""
        start_time = time.time()
        
        # URL 수집
        print("🔍 다운로드 URL 검색 중...")
        self.urls = self.search_download_url()
        print(f"찾은 URL: {len(self.urls)}개\n")
        
        if not self.urls:
            print("❌ 다운로드할 파일이 없습니다.")
            return None
        
        # Phase 1: 스키마 수집
        self.schema_columns = self._collect_schema(skip)
        
        # Phase 2: Parquet 변환
        self._convert_to_parquet()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  전체 실행 시간: {total_time:.2f}초")

    def load(self, adapter: Union[DatasetAdapter, str, None] = None, **kwargs):
        """어댑터에 따라 Parquet 파일 로드"""
        if not os.path.exists(self.output_file):
            print(f"❌ 파일이 존재하지 않습니다: {self.output_file}")
            return None

        target_adapter = adapter or self.adapter or DatasetAdapter.PANDAS
        if isinstance(target_adapter, str):
            try:
                target_adapter = DatasetAdapter(target_adapter.lower())
            except ValueError as exc:
                raise ValueError(f"지원하지 않는 어댑터입니다: {adapter}") from exc

        print(f"\n📖 {self.output_file} 로딩 중... (adapter={target_adapter.value})")

        if target_adapter == DatasetAdapter.PANDAS:
            return pd.read_parquet(self.output_file, **kwargs)
        if target_adapter == DatasetAdapter.POLARS:
            return pl.scan_parquet(self.output_file, **kwargs)
        if target_adapter == DatasetAdapter.SPARK:
            spark = SparkSession.builder.appName("DataLoader").getOrCreate()
            reader = spark.read
            if kwargs:
                reader = reader.options(**kwargs)
            return reader.parquet(self.output_file)

        raise ValueError(f"지원하지 않는 어댑터입니다: {target_adapter}")


# ============ 사용 예시 ============
if __name__ == '__main__':
    loader = DataLoader(
        start=2020,
        end=2025,
        output_file='output.parquet',
        max_workers=4
    )

    loader.process(skip=False)

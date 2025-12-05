from typing import List
import pyarrow as pa
import pyarrow.parquet as pq
from download import search_and_collect_json
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os

def flatten_dict(nested_dict, parent_key='', sep='_'):
    """중첩된 딕셔너리를 평탄화"""
    items = []
    
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for i, item in enumerate(v):
                items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)

def clean_empty_arrays(obj):
    """빈 문자열만 있는 배열을 None으로 변환"""
    if isinstance(obj, dict):
        return {k: clean_empty_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if obj == [""]:
            return None
        return [clean_empty_arrays(item) for item in obj]
    elif obj == "":
        return None
    return obj

def process_record_for_columns(record):
    """단일 레코드에서 컬럼 추출 (멀티프로세싱용)"""
    cleaned = clean_empty_arrays(record)
    flattened = flatten_dict(cleaned)
    return set(flattened.keys())

def process_record_for_conversion(record, all_columns):
    """단일 레코드를 변환 (멀티프로세싱용)"""
    cleaned = clean_empty_arrays(record)
    flattened = flatten_dict(cleaned)
    
    # 정규화
    normalized = {}
    for col in all_columns:
        normalized[col] = flattened.get(col, None)
    
    # 문자열 변환
    normalized = {k: (str(v) if v is not None else None) for k, v in normalized.items()}
    return normalized

def results_to_parquet_streaming_multiprocess(
    records_generator, 
    parquet_file, 
    chunk_size=5000,
    n_workers=None
):
    """
    멀티프로세싱과 progress bar를 사용한 Parquet 변환
    
    Args:
        records_generator: 레코드 제너레이터
        parquet_file: 출력 파일명
        chunk_size: 청크 크기 (기본 5000)
        n_workers: 워커 수 (기본: CPU 코어 수 - 1)
    """
    if n_workers is None:
        max_workers = max(1, cpu_count() - 1)
        n_workers = min(16, max_workers)
    
    print(f"🚀 멀티프로세싱 사용: {n_workers} workers")
    
    # 모든 레코드를 메모리에 로드 (제너레이터이므로 한 번만 순회 가능)
    print("📥 레코드 로딩 중...")
    temp_records = list(tqdm(records_generator, desc="레코드 로딩"))
    total_records = len(temp_records)
    print(f"총 {total_records:,}개 레코드 로드 완료\n")
    
    # Pass 1: 모든 컬럼 수집 (병렬)
    print("=== Pass 1: 모든 컬럼 수집 (병렬 처리) ===")
    all_columns = set()
    
    with Pool(n_workers) as pool:
        # chunksize를 조정하여 작업 분배 효율성 향상
        chunksize = max(1, total_records // (n_workers * 10))
        
        for columns_set in tqdm(
            pool.imap_unordered(process_record_for_columns, temp_records, chunksize=chunksize),
            total=total_records,
            desc="컬럼 스캔"
        ):
            all_columns.update(columns_set)
    
    all_columns = sorted(all_columns)
    print(f"✅ 총 {len(all_columns):,}개 고유 컬럼 발견\n")
    
    # 스키마 생성
    schema = pa.schema([(col, pa.string()) for col in all_columns])
    print(f"📋 스키마 생성 완료: {len(schema)} 컬럼\n")
    
    # Pass 2: Parquet 변환 (병렬 변환 + 순차 쓰기)
    print("=== Pass 2: Parquet 변환 (병렬 처리) ===")
    writer = pq.ParquetWriter(parquet_file, schema, compression='zstd')
    
    # 부분 함수 생성 (all_columns를 고정)
    process_func = partial(process_record_for_conversion, all_columns=all_columns)
    
    records_buffer = []
    total_processed = 0
    
    with Pool(n_workers) as pool:
        chunksize = max(1, total_records // (n_workers * 10))
        
        # imap_unordered를 사용하여 순서 무관하게 빠르게 처리
        for normalized_record in tqdm(
            pool.imap_unordered(process_func, temp_records, chunksize=chunksize),
            total=total_records,
            desc="변환 및 저장"
        ):
            records_buffer.append(normalized_record)
            
            # 청크가 찼으면 파일에 쓰기
            if len(records_buffer) >= chunk_size:
                table = pa.Table.from_pylist(records_buffer, schema=schema)
                writer.write_table(table)
                total_processed += len(records_buffer)
                records_buffer = []
    
    # 남은 레코드 처리
    if records_buffer:
        table = pa.Table.from_pylist(records_buffer, schema=schema)
        writer.write_table(table)
        total_processed += len(records_buffer)
    
    writer.close()
    print(f"\n✅ 완료! {total_processed:,}개 레코드를 {parquet_file}에 저장")
    
    # 파일 크기 출력
    file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
    print(f"📦 파일 크기: {file_size_mb:.2f} MB")

# 사용 예시
def record_generator(results: List[dict]):
    """dict에서 레코드를 하나씩 yield"""
    for record in results:
        yield record

if __name__=='__main__':
    start, end = 2024, 2024
    results = search_and_collect_json(start, end)
    
    # 멀티프로세스 버전 (추천)
    results_to_parquet_streaming_multiprocess(
        record_generator(results), 
        'output.parquet',
        chunk_size=5000,
        n_workers=None  # None이면 자동으로 CPU 코어 수 - 1
    )
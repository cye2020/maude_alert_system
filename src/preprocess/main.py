"""
메인 실행 스크립트 (LazyFrame)
"""
import polars as pl
from pathlib import Path
from code.preprocess.preprocessor import UDIProcessor
from code.preprocess.config import Config

def main(maude_path: str, udi_path: str, output_path: str, chunk_size: int = 1_000_000):
    """
    UDI 처리 실행
    
    Args:
        maude_path: MAUDE 데이터 경로
        udi_path: UDI DB 경로
        output_path: 결과 저장 경로
        chunk_size: chunk 크기
    
    Returns:
        최종 출력 파일 경로
    """
    # LazyFrame으로 로드 (메모리에 안 올림!)
    print("📂 데이터 로드 중 (LazyFrame)...")
    maude_lf = pl.scan_parquet(maude_path)
    udi_lf = pl.scan_parquet(udi_path)
    
    print(f"✓ MAUDE: {maude_path}")
    print(f"✓ UDI DB: {udi_path}\n")
    
    rename_udi_lf = udi_lf.rename({
        'company_name': 'manufacturer',
        'brand_name': 'brand',
        'version_or_model_number': 'model_number',
        'primary_udi_di': 'udi_di',
    })

    rename_maude_lf = maude_lf.rename({
        'device_0_manufacturer_d_name': 'manufacturer',
        'device_0_brand_name': 'brand',
        'device_0_model_number': 'model_number',
        'device_0_catalog_number': 'catalog_number',
        'device_0_lot_number': 'lot_number',
        'device_0_udi_di': 'udi_di',
    })
    
    # 처리
    processor = UDIProcessor(Config())
    result_path = processor.process(
        maude_lf=rename_maude_lf,
        udi_lf=rename_udi_lf,
        output_path=Path(output_path),
        chunk_size=chunk_size
    )
    
    return result_path


if __name__ == "__main__":
    result = main(
        maude_path="data/maude_sample.parquet",
        udi_path="data/udi.parquet",
        output_path="output/maude_with_udi.parquet",
        chunk_size=1_000_000  # 100만 건씩
    )
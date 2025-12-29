import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

from src.preprocess2.config import get_config
from src.preprocess2.steps import ColumnDropStage, FilterStage
logger = logging.getLogger(__name__)


class SilverPipeline:
    def __init__(self, use_s3: bool = False, datasets: List[str] = None) -> None:
        """
        Args:
            use_s3: S3 사용 여부
            datasets: 처리할 데이터셋 리스트 ['maude', 'udi'] (기본값: ['maude'])
        """
        self.config = get_config()
        self.drop_columns_1st_config: dict = self.config.columns.get('column_drop_1st')
        self.scope_config: dict = self.config.filtering.get('scoping')
        
        self.use_s3 = use_s3 if use_s3 is not None else self.config.base['paths'].get('use_s3', False)
        self.datasets = datasets or ['maude', 'udi']
        
        # 데이터셋별 경로 저장
        self.paths = {
            'bronze': {},
            'silver': {}
        }
        
        for dataset in self.datasets:
            self.paths['bronze'][dataset] = self._get_path('bronze', dataset)
            self.paths['silver'][dataset] = self._get_path('silver', dataset)
        
        self.stats = {}
        
        logger.info(f"SilverPipeline initialized for datasets: {self.datasets}")
        for dataset in self.datasets:
            logger.info(f"  {dataset.upper()}")
            logger.info(f"    Bronze: {self.paths['bronze'][dataset]}")
            logger.info(f"    Silver: {self.paths['silver'][dataset]}")
    
    def _get_path(self, stage: str, dataset: str) -> Path:
        """경로 가져오기 (통합)"""
        if self.use_s3:
            base = self.config.storage['s3']['paths'][stage]
        else:
            base = self.config.base['paths']['local'][stage]
        
        filename = self.config.base['datasets'][dataset][f'{stage}_file']
        return Path(base) / filename
    
    def load_bronze_data(self, dataset: str = 'maude') -> pl.LazyFrame:
        """Bronze 데이터 로드"""
        if dataset not in self.datasets:
            raise ValueError(f"Dataset '{dataset}' not configured in pipeline")
        
        data_path = self.paths['bronze'][dataset]
        
        logger.info(f"Loading {dataset} bronze data from {data_path}...")
        
        # 파일 존재 확인 (로컬인 경우만)
        if not self.use_s3 and not data_path.exists():
            raise FileNotFoundError(
                f"Bronze data not found: {data_path}\n"
                f"Please run the bronze pipeline first."
            )
        
        try:
            lf = pl.scan_parquet(str(data_path))
            
            schema = lf.collect_schema()
            col_count = schema.len()
            
            logger.info(f"  ✓ Loaded {dataset} data ({col_count} columns)")
            
            return lf
            
        except Exception as e:
            logger.error(f"Failed to load {dataset} bronze data: {e}")
            raise

    def validate_bronze_data(self, lf: pl.LazyFrame) -> bool:
        """
        Bronze 데이터 기본 검증

        Status:
            Planned (Not yet used in production)

        Notes:
            - 현재 파이프라인에서는 호출되지 않는다.
            - 스키마 안정화 이후 활성화 예정.
        """
        if not self.config.pipeline['validation']['bronze']['enabled']:
            raise RuntimeError("Bronze validation is not enabled yet")
        
        logger.info("Validating bronze data...")
        
        schema = lf.collect_schema()
        
        # 필수 컬럼 확인
        required_columns = self.config.quality['critical_fields']['fields']
        missing_columns = [col for col in required_columns if col not in schema.names()]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # 최소 행 개수 확인 (실제 collect 필요)
        min_rows = self.config.pipeline['validation']['silver']['min_rows']
        row_count = lf.select(pl.len()).collect().item()
        
        if row_count < min_rows:
            logger.error(f"Insufficient rows: {row_count} < {min_rows}")
            return False
        
        self.stats['bronze_rows'] = row_count
        logger.info(f"  ✓ Bronze data validated")
        logger.info(f"  ✓ Rows: {row_count:,}")
        
        return True
    
    def _run_each(self, dataset: str):
        """
        각 데이터셋이 따로 진행
        """
        try:
            logger.info("=" * 60)
            logger.info(f"Run {dataset} Dataset")
            logger.info("=" * 60)
            
            # Bronze 데이터 로드
            logger.info(f'Load Bronze {dataset} Dataset...')
            lf = self.load_bronze_data(dataset)
            
            # 1차 컬럼 드랍
            logger.info(f'Drop {dataset} Columns...')
            lf = self.drop_columns_1st(lf, dataset)

            # 스코핑
            logger.info(f'Scoping {dataset} Data...')
            lf = self.scope_data(lf, dataset)
            
            # 클렌징

        except Exception as e:
            logger.error(f"Run {dataset} Dataset failed: {e}", exc_info=True)
            return False
            
    
    def run(self) -> bool:
        """Silver 파이프라인 전체 실행
        
        Returns:
            성공 여부
        """
        try:
            self.stats['start_time'] = datetime.now()
            logger.info("=" * 60)
            logger.info("Starting Silver Pipeline")
            logger.info("=" * 60)
            
            # maude 데이터 진행
            for dataset in self.datasets:
                self._run_each(dataset)
            
            # 3-2. 클렌징
            # maude_clean_lf = self.clean_columns(maude_lf, 'maude')
            # udi_clean_lf = self.clean_columns(udi_lf, 'udi')
            
            # lf = self.clean_data(lf)
            # lf = self.remove_duplicates(lf)
            
            # 4. Silver 데이터 저장
            # self.save_silver_data(lf)
            
            self.stats['end_time'] = datetime.now()
            self._log_summary()
            
            logger.info("Silver pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Silver pipeline failed: {e}", exc_info=True)
            return False
    
    def drop_columns_1st(self, lf: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
        verbose = self.drop_columns_1st_config.get('verbose')
        drop_config: dict = self.drop_columns_1st_config.get(dataset)
        
        mode = drop_config.get('mode', 'blacklist')
        patterns = drop_config.get('patterns', [])
        cols = drop_config.get('cols', [])
        
        drop_stage = ColumnDropStage(verbose)
        dropped_lf = drop_stage.drop_columns(lf, patterns, cols, mode)
        
        return dropped_lf

    def scope_data(self, lf: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
        verbose = self.scope_config.get('verbose')
        scope_config: dict = self.scope_config.get(dataset)
        groups: Dict[list] = scope_config.get('groups', [])
        combine_groups = scope_config.get('combine_groups', 'AND')
        
        filter_stage = FilterStage(verbose)
        scope_lf = filter_stage.filter_groups(lf, groups, combine_groups)
        
        return scope_lf

    def _load_cleaning_config(self):
        """Config에서 클린징 설정 로드"""
        # 패턴 로드
        self.cleaning_patterns = {
            'generic': self.config.cleaning['patterns']['generic'],
            'udi': self.config.cleaning['patterns']['udi'],
            'company_name': self.config.cleaning['patterns']['company_name'],
            'generic_text': self.config.cleaning['patterns']['generic_text'],
        }
        
        # 실행 옵션
        exec_config = self.config.cleaning['execution']
        self.clean_chunk_size = exec_config['chunk_size']
        self.clean_keep_temp = exec_config['keep_temp']
        
        # 컬럼 전략
        strategies = self.config.cleaning['column_strategies']
        self.udi_columns = strategies['udi_columns']
        self.company_columns = strategies['company_columns']
        self.text_columns = strategies['text_columns']
        
        # 전처리 옵션
        self.preprocess_options = self.config.cleaning['preprocessing_options']
    
    def _merge_patterns(self, pattern_types: List[str]) -> Dict[str, List[Dict]]:
        """여러 패턴 타입을 병합
        
        Args:
            pattern_types: ['generic', 'udi'] 등
            
        Returns:
            {'delete': [...], 'remove': [...]}
        """
        merged = {'delete': [], 'remove': []}
        
        for ptype in pattern_types:
            patterns = self.cleaning_patterns[ptype]
            merged['delete'].extend(patterns.get('delete', []))
            merged['remove'].extend(patterns.get('remove', []))
        
        return merged
    
    def clean_data(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """데이터 클린징 단계"""
        logger.info("Step 2: Data cleaning...")
        
        # 1. UDI 컬럼 클린징
        udi_options = self.preprocess_options['udi']
        udi_patterns = self._merge_patterns(udi_options['patterns'])
        
        output_path = self.silver_path.parent / "temp_cleaned_udi.parquet"
        lf = self.cleaning_stage.clean_columns(
            lf=lf,
            columns=self.udi_columns,
            patterns=udi_patterns,
            output_path=output_path,
            uppercase=udi_options['uppercase'],
            remove_countries=udi_options['remove_countries'],
            chunk_size=self.clean_chunk_size,
            keep_temp=self.clean_keep_temp
        )
        
        # 2. 회사명 컬럼 클린징
        company_options = self.preprocess_options['company_name']
        company_patterns = self._merge_patterns(company_options['patterns'])
        
        output_path = self.silver_path.parent / "temp_cleaned_company.parquet"
        lf = self.cleaning_stage.clean_columns(
            lf=lf,
            columns=self.company_columns,
            patterns=company_patterns,
            output_path=output_path,
            uppercase=company_options['uppercase'],
            remove_countries=company_options['remove_countries'],
            chunk_size=self.chunk_size,
            keep_temp=self.keep_temp
        )
        
        # 3. 일반 텍스트 컬럼 클린징
        text_options = self.preprocess_options['generic_text']
        text_patterns = self._merge_patterns(text_options['patterns'])
        
        output_path = self.silver_path.parent / "temp_cleaned_text.parquet"
        lf = self.cleaning_stage.clean_columns(
            lf=lf,
            columns=self.text_columns,
            patterns=text_patterns,
            output_path=output_path,
            uppercase=text_options['uppercase'],
            remove_countries=text_options['remove_countries'],
            chunk_size=self.chunk_size,
            keep_temp=self.keep_temp
        )
        
        # 통계 저장
        self.stats['cleaning'] = self.cleaning_stage.get_stats()
        
        logger.info("  ✓ Data cleaning completed")
        
        return lf

    def _log_summary(self):
        """처리 결과 요약 로그"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info("=" * 60)
        logger.info("Silver Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        # logger.info(f"Bronze rows: {self.stats['bronze_rows']:,}")
        # logger.info(f"Silver rows: {self.stats['silver_rows']:,}")
        # logger.info(f"Columns dropped: {len(self.stats['columns_dropped'])}")
        # logger.info(f"Duplicates removed: {self.stats['duplicates_removed']:,}")
        logger.info("=" * 60)


# 사용 예시
if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )
    
    # 파이프라인 실행
    pipeline = SilverPipeline(use_s3=False, datasets=['maude', 'udi'])
    success = pipeline.run()
    
    # if success:
    #     print("✓ Pipeline completed successfully")
    # else:
    #     print("✗ Pipeline failed")
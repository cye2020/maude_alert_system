from typing import List
import polars as pl
import polars.selectors as cs
import logging

class ColumnDropStage:
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 로깅 활성화 여부
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # 통계 추적
        self.stats = {
            'original_columns': [],
            'dropped_columns': [],
            'kept_columns': [],
            'drop_count': 0,
        }
    
    def drop_columns(
        self, lf: pl.LazyFrame, 
        patterns: List[str] = None, cols: List[str] = None,
        mode: str = 'blacklist'
    ) -> pl.LazyFrame:
      
        # 원본 컬럼 저장
        original_cols = lf.collect_schema().names()
        self.stats['original_columns'] = original_cols
                
        if mode not in ['blacklist', 'whitelist']:
          msg = f'Invalid mode: "{mode}". Expected one of ["blacklist", "whitelist"].'
          raise ValueError(msg)
        
        regex = "|".join(patterns) if patterns else None

        selectors = []
        if cols:
            selectors.extend(cols)
        if regex:
            selectors.append(cs.matches(regex))

        if selectors:
          result_lf = lf.drop(*selectors) if mode == 'blacklist' else lf.select(*selectors)
        else:
          result_lf = lf

        # 결과 컬럼 확인
        result_cols = result_lf.collect_schema().names()
        
        # 통계 업데이트
        if mode == 'blacklist':
            dropped = set(original_cols) - set(result_cols)
            self.stats['dropped_columns'] = sorted(dropped)
            self.stats['kept_columns'] = sorted(result_cols)
        else:
            kept = set(result_cols)
            self.stats['kept_columns'] = sorted(kept)
            self.stats['dropped_columns'] = sorted(set(original_cols) - kept)
        
        self.stats['drop_count'] = len(self.stats['dropped_columns'])
        
        # 로깅
        if self.verbose:
            self.logger.info(f"Column {mode} applied:")
            self.logger.info(f"  Original: {len(original_cols)} columns")
            self.logger.info(f"  Result: {len(result_cols)} columns")
            self.logger.info(f"  Dropped: {self.stats['drop_count']} columns")
        
        return result_lf

    
    def drop_high_null_columns(self, lf: pl.LazyFrame, threshold: float = 0.95):
        pass
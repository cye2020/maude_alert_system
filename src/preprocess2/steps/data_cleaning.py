"""
데이터 클린징 단계
Pipeline에서 전달받은 패턴으로 텍스트 클린징 실행
"""

import re
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional
import logging
from country_named_entity_recognition import find_countries

from src.utils import process_lazyframe_in_chunks, apply_mapping_to_columns

logger = logging.getLogger(__name__)


class CleaningStage:
    """데이터 클린징 단계
    
    Pipeline에서 전달받은 패턴과 옵션으로 텍스트 클린징 수행
    ColumnDropper와 동일한 수준의 간단한 인터페이스
    """
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 로깅 활성화 여부
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # 통계 추적
        self.stats = {
            'columns_processed': [],
            'total_unique_values': 0,
            'values_kept': 0,
            'values_deleted': 0,
            'patterns_applied': {},  # {column: pattern_count}
        }
        
        # 전처리기 캐시 (동일 패턴 재사용)
        self._preprocessors: Dict[str, 'TextPreprocessor'] = {}
    
    def clean_columns(
        self,
        lf: pl.LazyFrame,
        columns: str | List[str],
        patterns: Dict[str, List[Dict]],  # {'delete': [...], 'remove': [...]}
        output_path: Path,
        uppercase: bool = True,
        remove_countries: bool = False,
        chunk_size: int = 1_000_000,
        keep_temp: bool = False
    ) -> pl.LazyFrame:
        """컬럼 클린징 실행
        
        Args:
            lf: 입력 LazyFrame
            columns: 처리할 컬럼 (단일 또는 리스트)
            patterns: 적용할 패턴
                {'delete': [{'pattern': '...', 'description': '...'}, ...],
                 'remove': [{'pattern': '...', 'description': '...'}, ...]}
            output_path: 출력 경로
            uppercase: 대문자 변환 여부
            remove_countries: 국가명 제거 여부
            chunk_size: 청크 크기
            keep_temp: 임시 파일 유지 여부
            
        Returns:
            처리된 LazyFrame (scan_parquet로 로드됨)
        """
        # 단일 컬럼이면 리스트로
        if isinstance(columns, str):
            columns = [columns]
        
        self.stats['columns_processed'] = columns
        
        if self.verbose:
            self.logger.info(f"Cleaning {len(columns)} column(s): {columns}")
        
        # 1. 전처리기 생성 (패턴 기반)
        pattern_key = self._get_pattern_key(patterns, uppercase, remove_countries)
        
        if pattern_key not in self._preprocessors:
            preprocessor = TextPreprocessor(
                name=f"cleaner_{pattern_key[:8]}",
                uppercase=uppercase,
                remove_countries=remove_countries
            )
            
            # 패턴 추가
            for p in patterns.get('delete', []):
                preprocessor.add_pattern(
                    p['pattern'],
                    'DELETE',
                    p.get('description', '')
                )
            
            for p in patterns.get('remove', []):
                preprocessor.add_pattern(
                    p['pattern'],
                    'REMOVE',
                    p.get('description', '')
                )
            
            self._preprocessors[pattern_key] = preprocessor
        else:
            preprocessor = self._preprocessors[pattern_key]
        
        # 패턴 개수 기록
        total_patterns = len(patterns.get('delete', [])) + len(patterns.get('remove', []))
        for col in columns:
            self.stats['patterns_applied'][col] = total_patterns
        
        # 2. Unique 값 추출
        all_unique_values = set()
        for col in columns:
            unique_vals = lf.select(col).unique().collect()[col]
            all_unique_values.update(unique_vals)
        
        self.stats['total_unique_values'] = len(all_unique_values)
        
        if self.verbose:
            self.logger.info(f"  Processing {len(all_unique_values):,} unique values...")
            self.logger.info(f"  Applying {total_patterns} patterns")
        
        # 3. 매핑 딕셔너리 생성
        mapping_dict = preprocessor.create_mapping_dict(
            pl.Series(list(all_unique_values))
        )
        
        # 통계
        values_kept = len([v for v in mapping_dict.values() if v is not None])
        values_deleted = len(mapping_dict) - values_kept
        
        self.stats['values_kept'] = values_kept
        self.stats['values_deleted'] = values_deleted
        
        if self.verbose:
            total = len(mapping_dict)
            self.logger.info(f"  Kept: {values_kept:,} ({values_kept/total*100:.1f}%)")
            self.logger.info(f"  Deleted: {values_deleted:,} ({values_deleted/total*100:.1f}%)")
        
        # 4. Chunk 단위 처리
        def transform(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            return apply_mapping_to_columns(chunk_lf, columns, mapping_dict)
        
        process_lazyframe_in_chunks(
            lf=lf,
            transform_func=transform,
            output_path=output_path,
            chunk_size=chunk_size,
            temp_dir_name=f'temp_cleaning_{"_".join(columns)}',
            keep_temp=keep_temp,
            desc=f"Cleaning {len(columns)} column(s)"
        )
        
        # 5. 결과 반환
        return pl.scan_parquet(output_path)
    
    def _get_pattern_key(
        self, 
        patterns: Dict, 
        uppercase: bool, 
        remove_countries: bool
    ) -> str:
        """패턴과 옵션의 해시키 생성 (캐싱용)"""
        import hashlib
        import json
        
        key_data = {
            'patterns': patterns,
            'uppercase': uppercase,
            'remove_countries': remove_countries
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_stats(self) -> Dict:
        """처리 통계 반환"""
        return self.stats.copy()
    
    def print_stats(self):
        """통계 출력 (디버깅용)"""
        print("\n" + "="*60)
        print("Cleaning Statistics")
        print("="*60)
        print(f"Columns processed: {len(self.stats['columns_processed'])}")
        for col in self.stats['columns_processed']:
            pattern_count = self.stats['patterns_applied'].get(col, 0)
            print(f"  - {col} ({pattern_count} patterns)")
        print(f"Total unique values: {self.stats['total_unique_values']:,}")
        print(f"Values kept: {self.stats['values_kept']:,}")
        print(f"Values deleted: {self.stats['values_deleted']:,}")
        print("="*60 + "\n")


# ========== 내부 클래스: TextPreprocessor ==========

class TextPreprocessor:
    """텍스트 전처리 엔진 (내부 사용)"""
    
    def __init__(
        self, 
        name: str = "default",
        uppercase: bool = True,
        remove_countries: bool = False
    ):
        """
        Args:
            name: 전처리기 이름
            uppercase: 대문자 변환 여부
            remove_countries: 국가명 제거 여부
        """
        self.name = name
        self.uppercase = uppercase
        self.remove_countries = remove_countries
        self._patterns = {'delete': [], 'remove': []}
        self._compiled = False
    
    def add_pattern(self, pattern: str, action: str, description: str = ""):
        """패턴 추가
        
        Args:
            pattern: 정규표현식 패턴
            action: 'DELETE' (null 변환) 또는 'REMOVE' (문자열 제거)
            description: 패턴 설명
        """
        if action not in ['DELETE', 'REMOVE']:
            raise ValueError("action must be 'DELETE' or 'REMOVE'")
        
        key = action.lower()
        self._patterns[key].append({
            'pattern': pattern,
            'regex': None,  # lazy compile
            'description': description
        })
        self._compiled = False
        
        return self
    
    def _compile(self):
        """패턴 컴파일 (lazy)"""
        if self._compiled:
            return
        
        for key in ['delete', 'remove']:
            for p in self._patterns[key]:
                if p['regex'] is None:
                    p['regex'] = re.compile(p['pattern'], re.IGNORECASE)
        
        self._compiled = True
    
    def clean(self, text: Optional[str]) -> Optional[str]:
        """단일 텍스트 클린징
        
        Args:
            text: 입력 텍스트
            
        Returns:
            클린징된 텍스트 (삭제 시 None)
        """
        if text is None or text == '':
            return None
        
        self._compile()
        
        # 양옆 공백 제거
        text = str(text).strip()
        
        # 대문자 변환 (옵션)
        if self.uppercase:
            text = text.upper()
        
        # REMOVE 패턴 적용 (문자열에서 제거)
        for p in self._patterns['remove']:
            text = p['regex'].sub('', text).strip()
            if not text:
                return None
        
        # DELETE 패턴 체크 (매칭되면 null)
        for p in self._patterns['delete']:
            if p['regex'].match(text):
                return None
        
        # 국가명 제거 (옵션)
        if self.remove_countries:
            text = self._remove_country_names(text)
            if not text:
                return None
        
        return text if text else None
    
    def _remove_country_names(self, text: str) -> Optional[str]:
        """NER을 사용해 텍스트에서 국가명 제거
        
        Args:
            text: 입력 텍스트
            
        Returns:
            국가명이 제거된 텍스트
        """
        if not text:
            return text
        
        try:
            countries_found = find_countries(text)
            
            if not countries_found:
                return text
            
            # 첫 번째 국가명 제거
            match_country = countries_found[0][1][0]
            regex = re.compile(match_country, re.IGNORECASE)
            text = regex.sub('', text).strip()
            
            return text if text else None
            
        except Exception as e:
            # NER 실패 시 원본 반환
            logger.debug(f"Failed to remove country names from '{text}': {e}")
            return text
    
    def create_mapping_dict(self, unique_values: pl.Series) -> Dict[str, Optional[str]]:
        """Unique 값 목록에서 매핑 딕셔너리 생성
        
        Args:
            unique_values: Unique 값 Series
            
        Returns:
            {원본: 클린징된값} 딕셔너리
        """
        return {val: self.clean(val) for val in unique_values}
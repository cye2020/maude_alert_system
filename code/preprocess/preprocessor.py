"""
UDI 처리 메인 클래스 (Score 기반 매칭) - 수정 버전
"""
import polars as pl
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from code.preprocess.config import Config
from code.preprocess.preprocess import (
    extract_di_from_public, 
    fuzzy_match_dict, 
    collect_unique_safe
)
from code.utils.chunk import process_lazyframe_in_chunks


class UDIProcessor:
    """UDI-DI 결측 처리 클래스 (Score 기반 매칭)"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.udi_di_lookup = None  # Primary 직접 매칭용
        self.udi_full_lookup_lf = None  # Score 매칭용 (LazyFrame)
        self.mfr_mapping = None
        self.udi_mapping = None
    
    # ==================== 1단계: 전처리 ====================
    
    def preprocess_maude(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """MAUDE 전처리"""
        print("🔧 MAUDE 전처리...")
        
        total_cols = lf.collect_schema().names()
        
        result_lf = lf.with_columns([
            # UDI-Public → DI 추출
            pl.col('udi_public')
              .map_elements(extract_di_from_public, return_dtype=pl.Utf8)
              .alias('extracted_di'),
            
            # 날짜 통합
            pl.coalesce([pl.col(c) for c in self.config.MAUDE_DATES if c in total_cols])
              .alias('report_date'),
        ])
        
        # UDI 통합
        result_lf = result_lf.with_columns([
            pl.coalesce(['udi_di', 'extracted_di']).alias('udi_combined'),
            
            pl.when(pl.col('udi_di').is_not_null())
              .then(pl.lit('original'))
              .when(pl.col('extracted_di').is_not_null())
              .then(pl.lit('extracted'))
              .otherwise(pl.lit('missing'))
              .alias('udi_source')
        ])
        
        print(f"   ✓ 전처리 완료")
        return result_lf
    
    def preprocess_udi_db(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """UDI DB 전처리"""
        print("🔧 UDI DB 전처리...")
        
        total_cols = lf.collect_schema().names()
        return lf.with_columns([
            pl.coalesce([pl.col(c) for c in self.config.UDI_DATES if c in total_cols])
              .alias('publish_date')
        ])
    
    def normalize_manufacturers(self, maude_lf: pl.LazyFrame, udi_lf: pl.LazyFrame):
        """제조사명 퍼지 매칭"""
        print("🔧 제조사명 퍼지 매칭...")
        
        maude_mfrs = collect_unique_safe(maude_lf, 'manufacturer')
        udi_mfrs = collect_unique_safe(udi_lf, 'manufacturer')
        
        self.mfr_mapping = fuzzy_match_dict(
            maude_mfrs, 
            udi_mfrs, 
            self.config.FUZZY_THRESHOLD
        )
        
        print(f"   매칭: {sum(k!=v for k,v in self.mfr_mapping.items())}/{len(maude_mfrs)} 건")
    
    def apply_normalization(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """제조사명 정규화 적용"""
        return lf.with_columns([
            pl.col('manufacturer').replace(self.mfr_mapping).alias('mfr_std')
        ])
    
    # ==================== 2단계: Lookup 생성 ====================
    
    def build_lookup(self, udi_lf: pl.LazyFrame):
        """
        Lookup 테이블 생성
        1. Primary 직접 매칭용 (collect)
        2. Full info + Secondary list (LazyFrame)
        """
        print("🔧 Lookup 테이블 생성...")
        
        # ========== Lookup 1: Primary 직접 매칭 ==========
        self.udi_di_lookup = udi_lf.select([
            'udi_di',
            'manufacturer',
            'brand',
            'model_number',
            'catalog_number',
            'publish_date'
        ]).unique(subset=['udi_di']).collect()
        
        print(f"   Primary UDI Lookup: {len(self.udi_di_lookup):,} 건")
        
        # ========== Lookup 2: Full info + Secondary list ==========
        schema = udi_lf.collect_schema()
        secondary_cols = [c for c in schema.names() 
                         if c.startswith('identifiers_') and c.endswith('_id')]
        
        if secondary_cols:
            print(f"   Secondary 컬럼: {len(secondary_cols)}개")
            
            # Secondary를 리스트로 묶기 (explode 안 함!)
            self.udi_full_lookup_lf = udi_lf.select([
                'udi_di',
                'manufacturer',
                'brand',
                'model_number',
                'catalog_number',
                'publish_date',
                pl.concat_list(secondary_cols).alias('secondary_list')
            ])
        else:
            print("   ⚠️  Secondary 컬럼 없음")
            self.udi_full_lookup_lf = udi_lf.select([
                'udi_di',
                'manufacturer',
                'brand',
                'model_number',
                'catalog_number',
                'publish_date',
                pl.lit(None).cast(pl.List(pl.Utf8)).alias('secondary_list')
            ])
        
        print(f"   Full UDI Lookup: LazyFrame (collect 안 함)")
    
    # ==================== 3단계: UDI 매핑 (Score 기반) ====================
    
    def build_udi_mapping(self, maude_lf: pl.LazyFrame, chunk_size: int = 10_000):
        """
        UDI 매핑 테이블 생성 (Score 기반)
        
        1. Primary 직접 매칭
        2. Secondary 매칭 (Score 4→3→2)
        3. No UDI 매칭 (Score 4→3→2)
        """
        print("🔧 UDI 매핑 테이블 생성 (Score 기반)...")
        
        # ========== Unique UDI + 메타데이터 추출 ==========
        unique_udi = maude_lf.select([
            'udi_combined',
            'mfr_std',
            'brand',
            'model_number',
            'catalog_number',
            'report_date'
        ]).unique(subset=['udi_combined'])
        
        print(f"   Unique UDI: {unique_udi.select(pl.len()).collect().item():,} 건")
        
        # ========== Case A: Primary 직접 매칭 ==========
        udi_with_primary = unique_udi.join(
            self.udi_di_lookup.lazy(),
            left_on='udi_combined',
            right_on='udi_di',
            how='left',
            suffix='_matched'
        ).with_columns([
            pl.col('manufacturer').is_not_null().alias('primary_matched')
        ])
        
        primary_success = udi_with_primary.filter(
            pl.col('primary_matched')
        ).select([
            'udi_combined',
            pl.col('udi_combined').alias('mapped_primary_udi'),
            pl.col('manufacturer').alias('mapped_manufacturer'),
            pl.col('brand_matched').alias('mapped_brand'),
            pl.col('model_number_matched').alias('mapped_model_number'),
            pl.col('catalog_number_matched').alias('mapped_catalog_number'),
            pl.lit('udi_direct').alias('udi_match_type'),
            pl.lit(3).alias('match_score')  # Perfect match (제조사는 이미 일치, 나머지 3개 필드)
        ])
        
        primary_failed = udi_with_primary.filter(
            ~pl.col('primary_matched')
        ).select([
            'udi_combined',
            'mfr_std',
            'brand',
            'model_number',
            'catalog_number',
            'report_date',
        ])
        
        print(f"   - Primary 직접 매칭: {primary_success.select(pl.len()).collect().item():,} 건")
        
        # ========== Case B: Secondary UDI 매칭 (Score 기반) ==========
        secondary_candidates = primary_failed.filter(
            pl.col('udi_combined').is_not_null()
        )
        
        len_secondary_candidates = secondary_candidates.select(pl.len()).collect().item()
        print(f"   - Secondary 매칭 시도: {len_secondary_candidates:,} 건")
        
        if len_secondary_candidates > 0:
            secondary_matched = self._match_secondary_with_score(secondary_candidates, chunk_size=chunk_size)
        else:
            secondary_matched = pl.LazyFrame()
        
        # ========== Case C: No UDI 매칭 (Score 기반) ==========
        no_udi_candidates = maude_lf.select([
            'udi_combined',
            'mfr_std',
            'brand',
            'model_number',
            'catalog_number',
            'report_date'
        ]).filter(
            pl.col('udi_combined').is_null()
        ).unique()
        
        len_no_udi_candidates = no_udi_candidates.select(pl.len()).collect().item()
        print(f"   - No UDI 매칭 시도: {len_no_udi_candidates:,} 건")
        
        if len_no_udi_candidates > 0:
            no_udi_matched = self._match_no_udi_with_score(no_udi_candidates, chunk_size=chunk_size)
        else:
            no_udi_matched = pl.LazyFrame()
        
        # ========== 매칭 실패 처리 ==========
        # Secondary 매칭 실패
        len_secondary_matched = secondary_matched.select(pl.len()).collect().item()
        if len_secondary_matched > 0:
            matched_udi = secondary_matched.select(pl.col('udi_combined')).collect().to_series().to_list()
            secondary_failed = secondary_candidates.filter(
                ~pl.col('udi_combined').is_in(matched_udi)
            ).select([
                'udi_combined',
                pl.col('udi_combined').alias('mapped_primary_udi'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_manufacturer'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_brand'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_model_number'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_catalog_number'),
                pl.lit('udi_no_match').alias('udi_match_type'),
                pl.lit(0).alias('match_score')
            ])
        else:
            secondary_failed = secondary_candidates.select([
                'udi_combined',
                pl.col('udi_combined').alias('mapped_primary_udi'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_manufacturer'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_brand'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_model_number'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_catalog_number'),
                pl.lit('udi_no_match').alias('udi_match_type'),
                pl.lit(0).alias('match_score')
            ])
        
        # No UDI 매칭 실패
        len_no_udi_matched = no_udi_matched.select(pl.len()).collect().item()
        if len_no_udi_matched > 0:
            print(no_udi_matched.collect_schema().names())
            matched_keys = no_udi_matched.select([
                'mfr_std', 'brand', 'model_number', 'catalog_number'
            ]).unique()
            
            no_udi_failed = no_udi_candidates.join(
                matched_keys,
                on=['mfr_std', 'brand', 'model_number', 'catalog_number'],
                how='anti'
            ).select([
                'udi_combined',  # null
                pl.lit(None).cast(pl.Utf8).alias('mapped_primary_udi'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_manufacturer'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_brand'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_model_number'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_catalog_number'),
                pl.lit('no_match').alias('udi_match_type'),
                pl.lit(0).alias('match_score')
            ])
        else:
            no_udi_failed = no_udi_candidates.select([
                'udi_combined',
                pl.lit(None).cast(pl.Utf8).alias('mapped_primary_udi'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_manufacturer'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_brand'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_model_number'),
                pl.lit(None).cast(pl.Utf8).alias('mapped_catalog_number'),
                pl.lit('no_match').alias('udi_match_type'),
                pl.lit(0).alias('match_score')
            ])
        
        len_secondary_failed = secondary_failed.select(pl.len()).collect().item()
        len_no_udi_failed = no_udi_failed.select(pl.len()).collect().item()
        
        # ========== 통합 ==========
        all_parts = [primary_success]
        if len_secondary_matched > 0:
            all_parts.append(secondary_matched)
        if len_secondary_failed > 0:
            all_parts.append(secondary_failed)
        if len_no_udi_matched > 0:
            all_parts.append(no_udi_matched)
        if len_no_udi_failed > 0:
            all_parts.append(no_udi_failed)
        
        self.udi_mapping = pl.concat(all_parts)
        len_udi_mapping = self.udi_mapping.select(pl.len()).collect().item()
        
        print(f"   ✅ 최종 UDI 매핑: {len_udi_mapping:,} 건")
        
        # 통계
        stats = self.udi_mapping.group_by('udi_match_type').agg([
            pl.len().alias('count')
        ]).sort('count', descending=True)
        
        print(stats.collect())

    
    def _match_secondary_with_score(self, candidates: pl.LazyFrame, chunk_size: int = 10_000) -> pl.LazyFrame:
        """
        Secondary UDI 매칭 (Score 기반)
        candidates를 chunk로 나눠서 처리 (메모리 절약)
        """
        print(f"      Secondary 매칭 (Score 기반)...")
        
        # Secondary list 추출
        candidate_udi_list = candidates.select(pl.col('udi_combined')).collect().to_series().to_list()
        
        temp_filtered_path = Path("data/temp_secondary_filtered.parquet")
        
        # UDI DB에서 secondary_list에 candidate가 있는 행만 필터링
        def filter_secondary(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            exploded = chunk_lf.select([
                'udi_di',
                'manufacturer',
                'brand',
                'model_number',
                'catalog_number',
                'secondary_list',
                'publish_date'
            ]).explode('secondary_list')
            
            matched = exploded.filter(
                pl.col('secondary_list').is_in(candidate_udi_list)
            )
            
            return matched.group_by('secondary_list').agg([
                pl.col('udi_di').alias('udi_list'),
                pl.col('manufacturer').alias('mfr_list'),
                pl.col('brand').alias('brand_list'),
                pl.col('model_number').alias('model_list'),
                pl.col('catalog_number').alias('catalog_list'),
                pl.col('publish_date').alias('publish_list'),
            ])
        
        process_lazyframe_in_chunks(
            lf=self.udi_full_lookup_lf,
            transform_func=filter_secondary,
            output_path=temp_filtered_path,
            chunk_size=chunk_size,
            desc="Secondary 필터링"
        )
        
        # 필터링된 결과 로드
        filtered_df = pl.read_parquet(temp_filtered_path)

        # 이후에는 Lazy로 써도 됨
        filtered_lf = filtered_df.lazy()

        
        # candidates를 chunk로 나눠서 매칭 (메모리 절약)
        all_results = []
        
        len_candidates = candidates.select(pl.len()).collect().item()
        print(f'No UDI 후보 개수: {len_candidates}')
        for offset in tqdm(
            range(0, len_candidates, chunk_size),
            desc="Secondary candidates",
            total=(len_candidates - 1) // chunk_size + 1
        ):
            candidates_chunk = candidates.slice(offset, chunk_size)
            
            result = self._match_with_score(
                candidates=candidates_chunk,
                udi_lookup_lf=filtered_lf,
                join_key='secondary_list',
                match_type='udi_secondary'
            )
            
            if not result.head(1).select(pl.len()).collect().is_empty():
                all_results.append(result)
        
        # 임시 파일 삭제
        temp_filtered_path.unlink(missing_ok=True)
        
        return pl.concat(all_results) if all_results else pl.LazyFrame()
    
    def _match_no_udi_with_score(self, candidates: pl.LazyFrame, chunk_size: int = 10_000) -> pl.LazyFrame:
        """
        No UDI 매칭 (Score 기반)
        candidates를 chunk로 나눠서 처리 (메모리 절약)
        """
        print(f"      No UDI 매칭 (Score 기반)...")
        
        # UDI DB를 단순화
        simplified_lf = self.udi_full_lookup_lf.select([
            'udi_di',
            'manufacturer',
            'brand',
            'model_number',
            'catalog_number',
            'publish_date'
        ])
        
        # candidates를 chunk로 나눠서 처리 (메모리 절약)
        all_results = []
        
        len_candidates = candidates.select(pl.len()).collect().item()
        
        for offset in tqdm(
            range(0, len_candidates, chunk_size),
            desc="No UDI Candidates",
            total=(len_candidates - 1) // chunk_size + 1
        ):
            candidates_chunk = candidates.slice(offset, chunk_size)
            
            results = []
            remaining = candidates_chunk.clone()
            
            len_remaining = remaining.select(pl.len()).collect().item()
            # Score 3 → 2 → 1 순으로 시도
            for min_score in [3, 2, 1]:
                if len_remaining == 0:
                    break
                
                print(f"            Score >= {min_score}: {len_remaining:,} 건 시도 중...")
                
                # Join (제조사로 매칭)
                matched = remaining.join(
                    simplified_lf,
                    left_on='mfr_std',
                    right_on='manufacturer',
                    how='inner'
                ).filter(
                    pl.col('publish_date') < pl.col('report_date')
                ).with_columns([
                    # Score 계산
                    (
                        (pl.col('brand') == pl.col('brand_right')).cast(pl.Int32) +
                        (
                            pl.when(pl.col('model_number').is_not_null() & pl.col('model_number_right').is_not_null())
                            .then(pl.col('model_number') == pl.col('model_number_right'))
                            .otherwise(False)
                        ).cast(pl.Int32) +
                        (
                            pl.when(pl.col('catalog_number').is_not_null() & pl.col('catalog_number_right').is_not_null())
                            .then(pl.col('catalog_number') == pl.col('catalog_number_right'))
                            .otherwise(False)
                        ).cast(pl.Int32)
                    ).alias('match_score')
                ]).filter(
                    pl.col('match_score') >= min_score
                ).group_by([
                    'udi_combined', 'mfr_std', 'brand', 'model_number', 'catalog_number'
                ]).agg([
                    pl.col('udi_di').n_unique().alias('n_primary'),
                    pl.col('udi_di').first().alias('mapped_primary_udi'),
                    pl.col('brand_right').first().alias('mapped_brand'),
                    pl.col('model_number_right').first().alias('mapped_model_number'),
                    pl.col('catalog_number_right').first().alias('mapped_catalog_number'),
                    pl.col('match_score').max().alias('match_score')
                ]).filter(
                    pl.col('n_primary') == 1
                ).select([
                    'udi_combined',
                    'mapped_primary_udi',
                    pl.col('mfr_std').alias('mapped_manufacturer'),
                    'mapped_brand',
                    'mapped_model_number',
                    'mapped_catalog_number',
                    pl.lit('manufacturer_match').alias('udi_match_type'),
                    'match_score'
                ])
                
                len_matched = matched.select(pl.len()).collect().item()
                if len_matched > 0:
                    print(f"               → {len_matched:,} 건 성공")
                    results.append(matched)
                    
                    # 성공한 키 제외
                    matched_candidates = candidates_chunk.join(
                        matched.select(pl.col('udi_combined')),
                        on='udi_combined',
                        how='semi'
                    )
                    
                    remaining = remaining.join(
                        matched_candidates.select(['mfr_std', 'brand', 'model_number', 'catalog_number']),
                        on=['mfr_std', 'brand', 'model_number', 'catalog_number'],
                        how='anti'
                    )
            
            if results:
                all_results.extend(results)
        
        print('='*50, '끝', '='*50)
        return pl.concat(all_results) if all_results else pl.LazyFrame()
    
    def _match_with_score(
        self,
        candidates: pl.LazyFrame,
        udi_lookup_lf: pl.LazyFrame,
        join_key: str,
        match_type: str
    ) -> pl.LazyFrame:
        """
        Score 기반 매칭 (join_key 사용)
        제조사는 이미 정규화되었으므로 join 조건으로 사용, Score는 3개 필드만 계산
        
        Args:
            candidates: 매칭할 MAUDE 행들
            udi_lookup_lf: UDI DB (LazyFrame)
            join_key: join할 컬럼명 ('secondary_list' 등)
            match_type: 매칭 타입 ('udi_secondary' 등)
        """
        # explode된 데이터와 join
        expanded_lf = udi_lookup_lf.explode([
            'udi_list', 'mfr_list', 'brand_list', 'model_list', 'catalog_list', 'publish_list'
        ]).rename({
            'udi_list': 'udi_di',
            'mfr_list': 'manufacturer',
            'brand_list': 'brand',
            'model_list': 'model_number',
            'catalog_list': 'catalog_number',
            'publish_list': 'publish_date'
        })
        
        results = []
        remaining = candidates.clone()
        len_remaining = remaining.select(pl.len()).collect().item()
        
        # Score 3 → 2 → 1 순으로 시도 (제조사 제외, 최대 3점)
        for min_score in [3, 2, 1]:
            if len_remaining == 0:
                break
            
            print(f"         Score >= {min_score}: {len_remaining:,} 건 시도 중...")
            
            # Join (제조사 + join_key 동시 매칭)
            matched = remaining.join(
                expanded_lf,
                left_on=['udi_combined', 'mfr_std'],
                right_on=[join_key, 'manufacturer'],
                how='inner'
            ).filter(
                pl.col('publish_date') < pl.col('report_date')
            ).with_columns([
                # Score 계산 (brand + model + catalog만, 최대 3점)
                (
                    (pl.col('brand') == pl.col('brand_right')).cast(pl.Int32) +
                    (
                        pl.when(pl.col('model_number').is_not_null() & pl.col('model_number_right').is_not_null())
                        .then(pl.col('model_number') == pl.col('model_number_right'))
                        .otherwise(False)
                    ).cast(pl.Int32) +
                    (
                        pl.when(pl.col('catalog_number').is_not_null() & pl.col('catalog_number_right').is_not_null())
                        .then(pl.col('catalog_number') == pl.col('catalog_number_right'))
                        .otherwise(False)
                    ).cast(pl.Int32)
                ).alias('match_score')
            ]).filter(
                pl.col('match_score') >= min_score
            ).group_by([
                'udi_combined', 'mfr_std', 'brand', 'model_number', 'catalog_number'
            ]).agg([
                pl.col('udi_di').n_unique().alias('n_primary'),
                pl.col('udi_di').first().alias('mapped_primary_udi'),
                pl.col('brand_right').first().alias('mapped_brand'),
                pl.col('model_number_right').first().alias('mapped_model_number'),
                pl.col('catalog_number_right').first().alias('mapped_catalog_number'),
                pl.col('match_score').max().alias('match_score')
            ]).filter(
                pl.col('n_primary') == 1  # 단일 Primary만
            ).select([
                'udi_combined',
                'mapped_primary_udi',
                pl.col('mfr_std').alias('mapped_manufacturer'),  # group_by에 있는 컬럼 사용
                'mapped_brand',
                'mapped_model_number',
                'mapped_catalog_number',
                pl.lit(match_type).alias('udi_match_type'),
                'match_score'
            ])
            
            len_matched = matched.select(pl.len()).collect().item()
            if len_matched > 0:
                print(f"            → {len_matched:,} 건 성공")
                results.append(matched)
                
                # 성공한 것 제외
                matched_udi = matched.select(pl.col('udi_combined')).collect().to_series().to_list()
                remaining = remaining.filter(
                    ~pl.col('udi_combined').is_in(matched_udi)
                )
        
        return pl.concat(results) if results else pl.LazyFrame()
    
    # ==================== 4단계: 매칭 적용 ====================
    
    def process_all(
        self,
        maude_lf: pl.LazyFrame,
        output_path: Path,
        chunk_size: int = 1_000_000
    ):
        """전체 파이프라인 (UDI 매핑 활용)"""
        print("\n🔧 매칭 적용 중...")
        
        def transform_chunk(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            # UDI 매핑 join
            matched = chunk_lf.join(
                self.udi_mapping.lazy(),
                on='udi_combined',
                how='left',
                coalesce=True
            )
            
            # 최종 컬럼 생성
            matched = matched.with_columns([
                # device_version_id
                pl.coalesce([
                    'mapped_primary_udi',
                    'udi_combined'
                ]).alias('device_version_id'),
                
                # manufacturer
                pl.coalesce([
                    'mapped_manufacturer',
                    'manufacturer'
                ]).alias('manufacturer_final'),
                
                # brand
                pl.coalesce([
                    'mapped_brand',
                    'brand'
                ]).alias('brand_final'),
                
                # model_number
                pl.coalesce([
                    'mapped_model_number',
                    'model_number'
                ]).alias('model_number_final'),
                
                # catalog_number
                pl.coalesce([
                    'mapped_catalog_number',
                    'catalog_number'
                ]).alias('catalog_number_final'),
                
                # match_source (udi_match_type이 null이면 매핑 자체 실패)
                pl.coalesce([
                    'udi_match_type',
                    pl.lit('not_in_mapping')
                ]).alias('match_source')
            ])
            
            # 원본 컬럼 + 최종 컬럼
            original_cols = chunk_lf.collect_schema().names()
            final_cols = [
                *original_cols,
                'device_version_id',
                'manufacturer_final',
                'brand_final',
                'model_number_final',
                'catalog_number_final',
                'match_source',
                'match_score'
            ]
            
            return matched.select([c for c in final_cols if c in matched.collect_schema().names()])
        
        process_lazyframe_in_chunks(
            lf=maude_lf,
            transform_func=transform_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
            desc="UDI 매핑 적용"
        )
    
    # ==================== 5단계: 후처리 ====================
    
    def _post_process_complex_cases(self, input_path: Path, chunk_size: int):
        """후처리 - Tier 3 생성"""
        print("\n🔧 후처리 (Tier 3)...")
        
        lf = pl.scan_parquet(input_path)
        
        # 제조사별 준수율
        compliance = lf.group_by('mfr_std').agg([
            (pl.col('udi_combined').is_null().sum() / pl.len()).alias('missing_rate')
        ]).collect()
        
        low_compliance_mfrs = compliance.filter(
            pl.col('missing_rate') > self.config.LOW_COMPLIANCE_THRESHOLD
        )['mfr_std'].to_list()
        
        def resolve_chunk(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            # no_match → Tier 3 ID 생성
            chunk_lf = chunk_lf.with_columns([
                pl.when(pl.col('match_source') == 'no_match')
                  .then(
                      pl.when(pl.col('mfr_std').is_in(low_compliance_mfrs))
                        .then(pl.concat_str([
                            pl.lit('LOW_'), pl.col('mfr_std'), pl.lit('_'), pl.col('brand_final')
                        ]))
                        .otherwise(pl.concat_str([
                            pl.lit('UNK_'), pl.col('mfr_std'), pl.lit('_'), 
                            pl.col('brand_final'), pl.lit('_'), pl.col('catalog_number_final')
                        ]))
                  )
                  .otherwise(pl.col('device_version_id'))
                  .alias('device_version_id'),
                
                # 신뢰도
                pl.when(pl.col('match_source') == 'udi_direct')
                  .then(pl.lit('HIGH'))
                  .when(pl.col('match_source') == 'udi_secondary')
                  .then(pl.lit('HIGH'))
                  .when(pl.col('match_source') == 'manufacturer_match')
                  .then(pl.lit('MEDIUM'))
                  .when(pl.col('match_source') == 'udi_no_match')
                  .then(pl.lit('MEDIUM'))
                  .otherwise(pl.lit('VERY_LOW'))
                  .alias('udi_confidence'),
                
                pl.col('match_source').alias('final_source')
            ])
            
            return chunk_lf
        
        output_path = input_path.parent / f"{input_path.stem}_resolved.parquet"
        
        process_lazyframe_in_chunks(
            lf=lf,
            transform_func=resolve_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
            desc="Tier 3 처리"
        )
        
        print(f"✅ 최종 결과: {output_path}")
        return output_path
    
    # ==================== 전체 실행 ====================
    
    def process(
        self,
        maude_lf: pl.LazyFrame,
        udi_lf: pl.LazyFrame,
        output_path: Path,
        chunk_size: int = 1_000_000
    ) -> Path:
        """전체 파이프라인 실행"""
        print("="*60)
        print("UDI 처리 파이프라인 시작 (Score 기반 매칭)")
        print("="*60)
        
        # 1. 전처리
        maude_lf = self.preprocess_maude(maude_lf)
        udi_lf = self.preprocess_udi_db(udi_lf)
        
        # 2. 제조사명 정규화
        self.normalize_manufacturers(maude_lf, udi_lf)
        maude_lf = self.apply_normalization(maude_lf)
        
        # 3. Lookup 생성
        self.build_lookup(udi_lf)
        
        # 4. UDI 매핑 생성 (Score 기반)
        self.build_udi_mapping(maude_lf, chunk_size=chunk_size)
        
        # 5. 매칭 적용
        temp_path = output_path.parent / f"{output_path.stem}_temp.parquet"
        self.process_all(maude_lf, temp_path, chunk_size)
        
        # 6. 후처리
        final_path = self._post_process_complex_cases(temp_path, chunk_size)
        
        # 7. 최종 파일 이동
        final_path.rename(output_path)
        temp_path.unlink(missing_ok=True)
        
        # 통계
        print("\n" + "="*60)
        print("📊 최종 결과")
        print("="*60)
        
        result_lf = pl.scan_parquet(output_path)
        total = result_lf.select(pl.len()).collect().item()
        
        # match_source 분포
        match_stats = result_lf.group_by('match_source').agg([
            pl.len().alias('count'),
            (pl.len() / total * 100).round(2).alias('percent')
        ]).collect().sort('count', descending=True)
        
        print("\n매칭 출처 분포:")
        print(match_stats)
        
        # udi_confidence 분포
        conf_stats = result_lf.group_by('udi_confidence').agg([
            pl.len().alias('count'),
            (pl.len() / total * 100).round(2).alias('percent')
        ]).collect().sort('count', descending=True)
        
        print("\n신뢰도 분포:")
        print(conf_stats)
        
        # Score 분포
        score_stats = result_lf.group_by('match_score').agg([
            pl.len().alias('count'),
            (pl.len() / total * 100).round(2).alias('percent')
        ]).collect().sort('match_score', descending=True)
        
        print("\nScore 분포:")
        print(score_stats)
        
        print(f"\n✅ 총 {total:,} 건 처리 완료!")
        print(f"📁 결과: {output_path}")
        
        return output_path
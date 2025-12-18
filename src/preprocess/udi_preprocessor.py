"""
UDI 처리 메인 클래스 (Score 기반 매칭, Path 기반 설계)
"""
from uuid import uuid4
import polars as pl
from pathlib import Path
from tqdm import tqdm

from src.preprocess.config import Config
from src.preprocess.preprocess import (
    extract_di_from_public,
    fuzzy_match_dict,
    collect_unique_safe
)
from src.utils.chunk import process_lazyframe_in_chunks
from src.utils import uuid5_from_str


class UDIProcessor:
    """
    UDI-DI 결측 처리 클래스
    
    핵심 원칙: 함수 경계 = 실행 경계
    - 모든 내부 함수는 Path를 반환 (LazyFrame ❌)
    - 상위 레벨에서만 scan_parquet
    - temp 삭제는 최상위 finally에서만
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.udi_di_lookup = None  # Primary 직접 매칭용 (collect됨)
        self.udi_full_lookup_lf = None  # Score 매칭용 (LazyFrame, 큰 데이터)
        self.mfr_mapping = None

        self._temp_paths: list[Path] = []
        self.config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== Temp 관리 ====================
    
    def _new_temp_path(self, name: str) -> Path:
        """temp 파일 경로 생성 및 추적"""
        path = self.config.TEMP_DIR / name
        self._temp_paths.append(path)
        return path

    def _cleanup_temps(self):
        """모든 temp 파일 삭제"""
        for p in self._temp_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # ==================== 1단계: 전처리 ====================
    
    def preprocess_maude(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """MAUDE 전처리 (LazyFrame 유지)"""
        print("🔧 MAUDE 전처리...")
        
        cols = lf.collect_schema().names()
        
        lf = lf.with_columns([
            pl.col("udi_public")
              .map_elements(extract_di_from_public, return_dtype=pl.Utf8)
              .alias("extracted_di"),
            
            pl.coalesce([pl.col(c) for c in self.config.MAUDE_DATES if c in cols])
              .alias("report_date"),
        ])
        
        lf = lf.with_columns([
            pl.coalesce(["udi_di", "extracted_di"]).alias("udi_combined"),
            
            pl.when(pl.col("udi_di").is_not_null())
              .then(pl.lit("original"))
              .when(pl.col("extracted_di").is_not_null())
              .then(pl.lit("extracted"))
              .otherwise(pl.lit("missing"))
              .alias("udi_source"),
        ])
        
        print("   ✓ 전처리 완료")
        return lf

    def preprocess_udi_db(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """UDI DB 전처리 (LazyFrame 유지)"""
        print("🔧 UDI DB 전처리...")
        
        cols = lf.collect_schema().names()
        return lf.with_columns([
            pl.coalesce([pl.col(c) for c in self.config.UDI_DATES if c in cols])
              .alias("publish_date")
        ])

    # ==================== 2단계: 제조사 정규화 ====================
    
    def normalize_manufacturers(self, maude_lf: pl.LazyFrame, udi_lf: pl.LazyFrame):
        """제조사명 퍼지 매칭"""
        print("🔧 제조사명 퍼지 매칭...")
        
        maude_mfrs = collect_unique_safe(maude_lf, "manufacturer")
        udi_mfrs = collect_unique_safe(udi_lf, "manufacturer")
        
        self.mfr_mapping = fuzzy_match_dict(
            maude_mfrs, udi_mfrs, self.config.FUZZY_THRESHOLD
        )
        
        print(f"   매칭: {sum(k!=v for k,v in self.mfr_mapping.items())}/{len(maude_mfrs)} 건")

    def apply_normalization(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """제조사명 정규화 적용"""
        return lf.with_columns(
            pl.col("manufacturer").replace(self.mfr_mapping).alias("mfr_std")
        )

    # ==================== 3단계: Lookup 생성 ====================
    
    def build_lookup(self, udi_lf: pl.LazyFrame):
        """Lookup 테이블 생성"""
        print("🔧 Lookup 테이블 생성...")
        
        # Primary 직접 매칭용 (collect - 작음)
        self.udi_di_lookup = (
            udi_lf
            .select([
                "udi_di", "manufacturer", "brand",
                "model_number", "catalog_number", "publish_date"
            ])
            .unique(subset=["udi_di"])
            .collect()
        )
        
        print(f"   Primary UDI Lookup: {len(self.udi_di_lookup):,} 건")
        
        # Full info + Secondary list (LazyFrame - 큼)
        schema = udi_lf.collect_schema()
        sec_cols = [c for c in schema.names()
                   if c.startswith("identifiers_") and c.endswith("_id")]
        
        if sec_cols:
            print(f"   Secondary 컬럼: {len(sec_cols)}개")
            self.udi_full_lookup_lf = udi_lf.select([
                "udi_di", "manufacturer", "brand",
                "model_number", "catalog_number", "publish_date",
                pl.concat_list(sec_cols).alias("secondary_list")
            ])
        else:
            print("   ⚠️  Secondary 컬럼 없음")
            self.udi_full_lookup_lf = udi_lf.select([
                "udi_di", "manufacturer", "brand",
                "model_number", "catalog_number", "publish_date",
                pl.lit(None).cast(pl.List(pl.Utf8)).alias("secondary_list")
            ])
        
        print("   Full UDI Lookup: LazyFrame")

    # ==================== 4단계: Secondary 매칭 (Path 반환!) ====================
    
    def _match_secondary_with_score(
        self,
        candidates: pl.LazyFrame,
        chunk_size: int
    ) -> Path:
        """
        Secondary UDI 매칭 (Path 반환)
        
        Returns:
            매칭 결과가 저장된 parquet 경로
        """
        print("      Secondary 매칭 (Path 기반)...")
        
        output_path = self._new_temp_path(f"secondary_matched_{uuid4().hex}.parquet")
        
        # 빈 경우 빈 parquet 생성
        if candidates.select(pl.len()).collect().item() == 0:
            pl.DataFrame(schema={
                'mfr_std': pl.Utf8,
                'brand': pl.Utf8,
                'model_number': pl.Utf8,
                'catalog_number': pl.Utf8,
                'udi_combined': pl.Utf8,
                'mapped_primary_udi': pl.Utf8,
                'mapped_manufacturer': pl.Utf8,
                'mapped_brand': pl.Utf8,
                'mapped_model_number': pl.Utf8,
                'mapped_catalog_number': pl.Utf8,
                'udi_match_type': pl.Utf8,
                'match_score': pl.Int32
            }).write_parquet(output_path)
            return output_path
        
        # ========== Step 1: Secondary key parquet ==========
        key_path = self._new_temp_path(f"secondary_keys_{uuid4().hex}.parquet")
        candidates.select(
            pl.col("udi_combined").alias("secondary_key")
        ).unique().sink_parquet(key_path)
        
        keys_lf = pl.scan_parquet(key_path)
        
        # ========== Step 2: UDI DB explode + filter ==========
        lookup_path = self._new_temp_path(f"secondary_lookup_{uuid4().hex}.parquet")
        
        def explode_filter(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            return (
                chunk_lf
                .select([
                    "udi_di", "manufacturer", "brand",
                    "model_number", "catalog_number",
                    "publish_date", "secondary_list"
                ])
                .explode("secondary_list")
                .join(
                    keys_lf,
                    left_on="secondary_list",
                    right_on="secondary_key",
                    how="inner"
                )
            )
        
        process_lazyframe_in_chunks(
            lf=self.udi_full_lookup_lf,
            transform_func=explode_filter,
            output_path=lookup_path,
            chunk_size=chunk_size,
            desc="Secondary explode"
        )
        
        lookup_lf = pl.scan_parquet(lookup_path)
        
        # ========== Step 3: Score 매칭 (chunk 단위) ==========
        def match_chunk(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            return self._match_secondary_chunk_with_score(chunk_lf, lookup_lf)
        
        process_lazyframe_in_chunks(
            lf=candidates,
            transform_func=match_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
            desc="Secondary score match"
        )
        
        return output_path  # ✅ Path 반환!

    def _match_secondary_chunk_with_score(
        self,
        candidates_chunk: pl.LazyFrame,
        lookup_lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Secondary chunk별 score 매칭 - 일관성 있게 수정"""
        remaining = candidates_chunk
        results = []
        
        for min_score in [3, 2, 1]:
            if remaining.select(pl.len()).collect().item() == 0:
                break
            
            matched = (
                remaining
                .join(
                    lookup_lf,
                    left_on=["udi_combined", "mfr_std"],
                    right_on=["secondary_list", "manufacturer"],
                    how="inner"
                )
                .filter(pl.col("publish_date") < pl.col("report_date"))
                .with_columns([
                    (
                        (pl.col("brand") == pl.col("brand_right")).cast(pl.Int32) +
                        (
                            pl.when(
                                pl.col("model_number").is_not_null() &
                                pl.col("model_number_right").is_not_null()
                            )
                            .then(pl.col("model_number") == pl.col("model_number_right"))
                            .otherwise(False)
                        ).cast(pl.Int32) +
                        (
                            pl.when(
                                pl.col("catalog_number").is_not_null() &
                                pl.col("catalog_number_right").is_not_null()
                            )
                            .then(pl.col("catalog_number") == pl.col("catalog_number_right"))
                            .otherwise(False)
                        ).cast(pl.Int32)
                    ).alias("match_score")
                ])
                .filter(pl.col("match_score") >= min_score)
                .group_by([
                    "udi_combined", "mfr_std", "brand",
                    "model_number", "catalog_number"
                ])
                .agg([
                    pl.col("udi_di").n_unique().alias("n_primary"),
                    pl.col("udi_di").first().alias("mapped_primary_udi"),
                    pl.col("brand_right").first().alias("mapped_brand"),
                    pl.col("model_number_right").first().alias("mapped_model_number"),
                    pl.col("catalog_number_right").first().alias("mapped_catalog_number"),
                    pl.col("match_score").max().alias("match_score")
                ])
                .filter(pl.col("n_primary") == 1)
                .select([
                    'mfr_std',
                    'brand',
                    'model_number',
                    'catalog_number',
                    'udi_combined',
                    "mapped_primary_udi",
                    pl.col("mfr_std").alias("mapped_manufacturer"),
                    "mapped_brand",
                    "mapped_model_number",
                    "mapped_catalog_number",
                    pl.lit("udi_secondary").alias("udi_match_type"),
                    "match_score"
                ])
            )
            
            len_matched = matched.select(pl.len()).collect().item()
            if len_matched > 0:
                results.append(matched)
                
                # ✅ udi_combined으로 anti join (이건 괜찮음, null 아님)
                matched_keys = matched.select("udi_combined")
                remaining = remaining.join(matched_keys, on="udi_combined", how="anti")
        
        return pl.concat(results) if results else pl.LazyFrame()

    # ==================== 5단계: No UDI 매칭 (Path 반환!) ====================
    
    def _match_no_udi_with_score(
        self,
        candidates: pl.LazyFrame,
        chunk_size: int
    ) -> Path:
        """
        No UDI 매칭 (Path 반환)
        
        Returns:
            매칭 결과가 저장된 parquet 경로
        """
        print("      No UDI 매칭 (Path 기반)...")
        
        output_path = self._new_temp_path(f"no_udi_matched_{uuid4().hex}.parquet")
        
        if candidates.select(pl.len()).collect().item() == 0:
            pl.DataFrame(schema={
                'mfr_std': pl.Utf8,
                'brand': pl.Utf8,
                'model_number': pl.Utf8,
                'catalog_number': pl.Utf8,
                'udi_combined': pl.Utf8,
                'mapped_primary_udi': pl.Utf8,
                'mapped_manufacturer': pl.Utf8,
                'mapped_brand': pl.Utf8,
                'mapped_model_number': pl.Utf8,
                'mapped_catalog_number': pl.Utf8,
                'udi_match_type': pl.Utf8,
                'match_score': pl.Int32
            }).write_parquet(output_path)
            return output_path
        
        # ========== Step 1: 제조사 key parquet ==========
        mfr_key_path = self._new_temp_path(f"no_udi_mfr_keys_{uuid4().hex}.parquet")
        candidates.select(
            pl.col("mfr_std").alias("manufacturer")
        ).unique().sink_parquet(mfr_key_path)
        
        mfr_keys_lf = pl.scan_parquet(mfr_key_path)
        
        # ========== Step 2: UDI DB 제조사 필터링 ==========
        lookup_path = self._new_temp_path(f"no_udi_lookup_{uuid4().hex}.parquet")
        
        def filter_by_mfr(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            return (
                chunk_lf
                .select([
                    "udi_di", "manufacturer", "brand",
                    "model_number", "catalog_number", "publish_date"
                ])
                .join(mfr_keys_lf, on="manufacturer", how="inner")
            )
        
        process_lazyframe_in_chunks(
            lf=self.udi_full_lookup_lf,
            transform_func=filter_by_mfr,
            output_path=lookup_path,
            chunk_size=chunk_size,
            desc="No-UDI 제조사 필터"
        )
        
        lookup_lf = pl.scan_parquet(lookup_path)
        
        # ========== Step 3: Score 매칭 ==========
        def match_chunk(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            return self._match_no_udi_chunk_with_score(chunk_lf, lookup_lf)
        
        process_lazyframe_in_chunks(
            lf=candidates,
            transform_func=match_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
            desc="No-UDI score match"
        )
        
        return output_path  # ✅ Path 반환!

    def _match_no_udi_chunk_with_score(
        self,
        candidates_chunk: pl.LazyFrame,
        lookup_lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """No-UDI chunk별 score 매칭 - anti join 수정"""
        remaining = candidates_chunk
        results = []
        
        for min_score in [3, 2, 1]:
            if remaining.select(pl.len()).collect().item() == 0:
                break
            
            matched = (
                remaining
                .join(
                    lookup_lf,
                    left_on="mfr_std",
                    right_on="manufacturer",
                    how="inner"
                )
                .filter(pl.col("publish_date") < pl.col("report_date"))
                .with_columns([
                    (
                        (pl.col("brand") == pl.col("brand_right")).cast(pl.Int32) +
                        (
                            pl.when(
                                pl.col("model_number").is_not_null() &
                                pl.col("model_number_right").is_not_null()
                            )
                            .then(pl.col("model_number") == pl.col("model_number_right"))
                            .otherwise(False)
                        ).cast(pl.Int32) +
                        (
                            pl.when(
                                pl.col("catalog_number").is_not_null() &
                                pl.col("catalog_number_right").is_not_null()
                            )
                            .then(pl.col("catalog_number") == pl.col("catalog_number_right"))
                            .otherwise(False)
                        ).cast(pl.Int32)
                    ).alias("match_score")
                ])
                .filter(pl.col("match_score") >= min_score)
                .group_by([
                    "udi_combined", "mfr_std", "brand",
                    "model_number", "catalog_number"
                ])
                .agg([
                    pl.col("udi_di").n_unique().alias("n_primary"),
                    pl.col("udi_di").first().alias("mapped_primary_udi"),
                    pl.col("brand_right").first().alias("mapped_brand"),
                    pl.col("model_number_right").first().alias("mapped_model_number"),
                    pl.col("catalog_number_right").first().alias("mapped_catalog_number"),
                    pl.col("match_score").max().alias("match_score")
                ])
                .filter(pl.col("n_primary") == 1)
                .select([
                    'mfr_std',
                    'brand',
                    'model_number',
                    'catalog_number',
                    'udi_combined',
                    "mapped_primary_udi",
                    pl.col("mfr_std").alias("mapped_manufacturer"),
                    "mapped_brand",
                    "mapped_model_number",
                    "mapped_catalog_number",
                    pl.lit("meta_match").alias("udi_match_type"),
                    "match_score"
                ])
            )
            
            len_matched = matched.select(pl.len()).collect().item()
            if len_matched > 0:
                print(f"Score >= {min_score} → {len_matched:,} 건 성공")
                results.append(matched)
                
                # ✅ 수정: 4개 키로 anti join
                matched_keys = matched.select([
                    "mfr_std", "brand", "model_number", "catalog_number"
                ])
                remaining = remaining.join(
                    matched_keys,
                    on=["mfr_std", "brand", "model_number", "catalog_number"],
                    how="anti"
                )
        
        return pl.concat(results) if results else pl.LazyFrame()

    # ==================== 6단계: UDI 매핑 생성 (Path 반환!) ====================
    
    def build_udi_mapping(
        self,
        maude_lf: pl.LazyFrame,
        chunk_size: int
    ) -> Path:
        """
        UDI 매핑 테이블 생성 (Path 반환)
        
        Returns:
            최종 매핑 테이블이 저장된 parquet 경로
        """
        print("🔧 UDI 매핑 테이블 생성 (Score 기반)...")
        
        # ========== Unique UDI 추출 ==========
        unique_udi = maude_lf.select([
            "udi_combined", "mfr_std", "brand",
            "model_number", "catalog_number", "report_date"
        ]).unique(subset=["udi_combined"])
        
        print(f"   Unique UDI: {unique_udi.select(pl.len()).collect().item():,} 건")
        
        # ========== Case A: Primary 직접 매칭 ==========
        primary_success = unique_udi.join(
            self.udi_di_lookup.lazy(),
            left_on="udi_combined",
            right_on="udi_di",
            how="inner",  # ✅ 변경!
            suffix="_matched"
        ).select([
            'mfr_std',
            'brand',
            'model_number',
            'catalog_number',
            'udi_combined',
            pl.col("udi_combined").alias("mapped_primary_udi"),
            pl.col("manufacturer").alias("mapped_manufacturer"),
            pl.col("brand_matched").alias("mapped_brand"),
            pl.col("model_number_matched").alias("mapped_model_number"),
            pl.col("catalog_number_matched").alias("mapped_catalog_number"),
            pl.lit("udi_direct").alias("udi_match_type"),
            pl.lit(3).alias("match_score")
        ])

        primary_failed = unique_udi.join(
            primary_success.select("udi_combined"),
            on="udi_combined",
            how="anti"
        )

        # Primary → parquet
        primary_path = self._new_temp_path("primary_matched.parquet")
        primary_success.sink_parquet(primary_path)

        len_primary = pl.scan_parquet(primary_path).select(pl.len()).collect().item()
        print(f"   - Primary 직접 매칭: {len_primary:,} 건")

        # ========== Case B: Secondary 매칭 ==========
        secondary_candidates = primary_failed.filter(
            pl.col("udi_combined").is_not_null()
        )
        
        len_secondary_candidates = secondary_candidates.select(pl.len()).collect().item()
        print(f"   - Secondary 매칭 시도: {len_secondary_candidates:,} 건")
        
        if len_secondary_candidates > 0:
            secondary_path = self._match_secondary_with_score(
                secondary_candidates, chunk_size
            )  # ✅ Path 받음!
        else:
            # 빈 parquet
            secondary_path = self._new_temp_path("secondary_empty.parquet")
            pl.DataFrame(schema={
                'mfr_std': pl.Utf8,
                'brand': pl.Utf8,
                'model_number': pl.Utf8,
                'catalog_number': pl.Utf8,
                'udi_combined': pl.Utf8,
                'mapped_primary_udi': pl.Utf8,
                'mapped_manufacturer': pl.Utf8,
                'mapped_brand': pl.Utf8,
                'mapped_model_number': pl.Utf8,
                'mapped_catalog_number': pl.Utf8,
                'udi_match_type': pl.Utf8,
                'match_score': pl.Int32
            }).write_parquet(secondary_path)
        
        len_secondary = pl.scan_parquet(secondary_path).select(pl.len()).collect().item()
        print(f"   - Secondary 매칭 성공: {len_secondary:,} 건")
        
        # ========== Case C: No UDI 매칭 ==========
        no_udi_candidates = maude_lf.select([
            "udi_combined", "mfr_std", "brand",
            "model_number", "catalog_number", "report_date"
        ]).filter(
            pl.col("udi_combined").is_null()
        ).unique(subset=["mfr_std", "brand", "model_number", "catalog_number"])  # ✅ unique key 추가
        
        len_no_udi_candidates = no_udi_candidates.select(pl.len()).collect().item()
        print(f"   - No UDI 매칭 시도: {len_no_udi_candidates:,} 건")
        
        if len_no_udi_candidates > 0:
            no_udi_path = self._match_no_udi_with_score(
                no_udi_candidates, chunk_size
            )
        else:
            no_udi_path = self._new_temp_path("no_udi_empty.parquet")
            pl.DataFrame(schema={
                'mfr_std': pl.Utf8,
                'brand': pl.Utf8,
                'model_number': pl.Utf8,
                'catalog_number': pl.Utf8,
                'udi_combined': pl.Utf8,
                'mapped_primary_udi': pl.Utf8,
                'mapped_manufacturer': pl.Utf8,
                'mapped_brand': pl.Utf8,
                'mapped_model_number': pl.Utf8,
                'mapped_catalog_number': pl.Utf8,
                'udi_match_type': pl.Utf8,
                'match_score': pl.Int32
            }).write_parquet(no_udi_path)
        
        len_no_udi = pl.scan_parquet(no_udi_path).select(pl.len()).collect().item()
        print(f"   - No UDI 매칭 성공: {len_no_udi:,} 건")
        
        # ========== 매칭 실패 처리 ==========
        # Secondary 실패
        if len_secondary > 0:
            secondary_matched_udi = pl.scan_parquet(secondary_path).select(
                "udi_combined"
            ).collect().to_series().to_list()
            
            secondary_failed = secondary_candidates.filter(
                ~pl.col("udi_combined").is_in(secondary_matched_udi)
            )
        else:
            secondary_failed = secondary_candidates
        
        secondary_failed_path = self._new_temp_path("secondary_failed.parquet")
        secondary_failed.select([
            'mfr_std',
            'brand',
            'model_number',
            'catalog_number',
            'udi_combined',
            pl.col("udi_combined").alias("mapped_primary_udi"),
            pl.lit(None).cast(pl.Utf8).alias("mapped_manufacturer"),
            pl.lit(None).cast(pl.Utf8).alias("mapped_brand"),
            pl.lit(None).cast(pl.Utf8).alias("mapped_model_number"),
            pl.lit(None).cast(pl.Utf8).alias("mapped_catalog_number"),
            pl.lit("udi_no_match").alias("udi_match_type"),
            pl.lit(0).alias("match_score")
        ]).sink_parquet(secondary_failed_path)
        
        # ========== No UDI 실패 처리 (수정!) ==========
        no_udi_failed_path = self._new_temp_path("no_udi_failed.parquet")
        
        if len_no_udi > 0:
            # no_udi_path에서 매칭 성공한 키 추출
            matched_keys = pl.scan_parquet(no_udi_path).select([
                "mapped_manufacturer",
                "mapped_brand", 
                "mapped_model_number",
                "mapped_catalog_number"
            ]).unique()
            
            # 실패한 것만 필터링 (원본 키로 비교)
            no_udi_candidates.join(
                matched_keys,
                left_on=["mfr_std", "brand", "model_number", "catalog_number"],
                right_on=["mapped_manufacturer", "mapped_brand", "mapped_model_number", "mapped_catalog_number"],
                how="anti"
            ).select([
                'mfr_std',
                'brand',
                'model_number',
                'catalog_number',
                'udi_combined',
                pl.lit(None).cast(pl.Utf8).alias("mapped_primary_udi"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_manufacturer"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_brand"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_model_number"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_catalog_number"),
                pl.lit("no_match").alias("udi_match_type"),
                pl.lit(0).alias("match_score")
            ]).sink_parquet(no_udi_failed_path)
        else:
            # 전체가 실패
            no_udi_candidates.select([
                'mfr_std',
                'brand',
                'model_number',
                'catalog_number',
                'udi_combined',
                pl.lit(None).cast(pl.Utf8).alias("mapped_primary_udi"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_manufacturer"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_brand"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_model_number"),
                pl.lit(None).cast(pl.Utf8).alias("mapped_catalog_number"),
                pl.lit("no_match").alias("udi_match_type"),
                pl.lit(0).alias("match_score")
            ]).sink_parquet(no_udi_failed_path)
        
        # ========== 통합 ==========
        final_path = self._new_temp_path("udi_mapping_final.parquet")
        
        pl.concat([
            pl.scan_parquet(primary_path),
            pl.scan_parquet(secondary_path),
            pl.scan_parquet(secondary_failed_path),
            pl.scan_parquet(no_udi_path),
            pl.scan_parquet(no_udi_failed_path)
        ]).sink_parquet(final_path)
        
        total = pl.scan_parquet(final_path).select(pl.len()).collect().item()
        print(f"   ✅ 최종 UDI 매핑: {total:,} 건")
        
        # 통계
        stats = pl.scan_parquet(final_path).group_by("udi_match_type").agg([
            pl.len().alias("count")
        ]).sort("count", descending=True).collect()
        
        print(stats)
        
        return final_path

    # ==================== 7단계: 매칭 적용 ====================
    def process_all(
        self,
        maude_lf: pl.LazyFrame,
        mapping_path: Path,
        output_path: Path,
        chunk_size: int
    ):
        """전체 파이프라인 (매핑 적용) - 스키마 통일 버전"""
        print("\n🔧 매칭 적용 중...")
        
        mapping_lf = pl.scan_parquet(mapping_path)
        
        # 매핑 테이블을 UDI 있음/없음으로 분리
        mapping_with_udi = mapping_lf.filter(pl.col("udi_combined").is_not_null())
        mapping_no_udi = mapping_lf.filter(pl.col("udi_combined").is_null())
        
        def transform_chunk(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            # 원본 컬럼 리스트
            original_cols = chunk_lf.collect_schema().names()
            
            # UDI 있는 행과 없는 행 분리
            chunk_with_udi = chunk_lf.filter(pl.col("udi_combined").is_not_null())
            chunk_no_udi = chunk_lf.filter(pl.col("udi_combined").is_null())
            
            results = []
            
            # ========== Case 1: UDI 있는 경우 ==========
            if chunk_with_udi.select(pl.len()).collect().item() > 0:
                matched_with_udi = (
                    chunk_with_udi
                    .join(
                        mapping_with_udi,
                        on="udi_combined",
                        how="left",
                        suffix="_mapping"
                    )
                    .with_columns([
                        pl.coalesce(["mapped_primary_udi", "udi_combined"]).alias("device_version_id"),
                        pl.coalesce(["mapped_manufacturer", "manufacturer"]).alias("manufacturer_final"),
                        pl.coalesce(["mapped_brand", "brand"]).alias("brand_final"),
                        pl.coalesce(["mapped_model_number", "model_number"]).alias("model_number_final"),
                        pl.coalesce(["mapped_catalog_number", "catalog_number"]).alias("catalog_number_final"),
                        pl.coalesce(["udi_match_type", pl.lit("not_in_mapping")]).alias("match_source")
                    ])
                    .select([
                        *original_cols,  # 원본 컬럼 유지
                        "device_version_id",
                        "manufacturer_final",
                        "brand_final",
                        "model_number_final",
                        "catalog_number_final",
                        "match_source",
                        "match_score"
                    ])
                )
                results.append(matched_with_udi)
            
            # ========== Case 2: UDI 없는 경우 ==========
            if chunk_no_udi.select(pl.len()).collect().item() > 0:
                matched_no_udi = (
                    chunk_no_udi
                    .join(
                        mapping_no_udi,
                        on=["mfr_std", "brand", "model_number", "catalog_number"],
                        how="left",
                        suffix="_mapping"
                    )
                    .with_columns([
                        pl.coalesce(["mapped_primary_udi"]).alias("device_version_id"),
                        pl.coalesce(["mapped_manufacturer", "manufacturer"]).alias("manufacturer_final"),
                        pl.coalesce(["mapped_brand", "brand"]).alias("brand_final"),
                        pl.coalesce(["mapped_model_number", "model_number"]).alias("model_number_final"),
                        pl.coalesce(["mapped_catalog_number", "catalog_number"]).alias("catalog_number_final"),
                        pl.coalesce(["udi_match_type", pl.lit("not_in_mapping")]).alias("match_source")
                    ])
                    .select([
                        *original_cols,  # ✅ 같은 원본 컬럼
                        "device_version_id",
                        "manufacturer_final",
                        "brand_final",
                        "model_number_final",
                        "catalog_number_final",
                        "match_source",
                        "match_score"
                    ])
                )
                results.append(matched_no_udi)
            
            # ========== 통합 (스키마 동일!) ==========
            return pl.concat(results) if results else chunk_lf
        
        process_lazyframe_in_chunks(
            lf=maude_lf,
            transform_func=transform_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
            desc="UDI 매핑 적용"
        )

    # ==================== 8단계: 후처리 ====================
    
    def _post_process_complex_cases(self, input_path: Path, chunk_size: int) -> Path:
        """후처리 - Tier 3 생성 (Path 반환)"""
        print("\n🔧 후처리 (Tier 3)...")
        
        lf = pl.scan_parquet(input_path)
        
        compliance = lf.group_by("mfr_std").agg([
            (pl.col("udi_combined").is_null().sum() / pl.len()).alias("missing_rate")
        ]).collect()
        
        low_compliance_mfrs = compliance.filter(
            pl.col("missing_rate") > self.config.LOW_COMPLIANCE_THRESHOLD
        )["mfr_std"].to_list()
        
        def resolve_chunk(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            chunk_lf = chunk_lf.with_columns([
                # ✅ 매칭 실패 케이스 모두 처리
                pl.when(
                    pl.col("match_source").is_in([
                        "no_match", 
                        "not_in_mapping", 
                        # "udi_no_match"
                    ])
                )
                .then(
                    pl.when(pl.col("mfr_std").is_in(low_compliance_mfrs))
                    .then(pl.concat_str([
                        pl.lit("LOW_"), 
                        pl.col("mfr_std"), 
                        pl.lit("_"), 
                        pl.coalesce(["brand_final", pl.lit("UNKNOWN")])
                    ])
                    # .map_elements(uuid5_from_str)
                    )
                    .otherwise(pl.concat_str([
                        pl.lit("UNK_"), 
                        pl.col("mfr_std"), 
                        pl.lit("_"),
                        pl.coalesce(["brand_final", pl.lit("UNKNOWN")]), 
                        pl.lit("_"), 
                        pl.coalesce(["catalog_number_final", pl.lit("NA")])
                    ])
                    # .map_elements(uuid5_from_str)
                    )
                )
                .otherwise(pl.col("device_version_id"))
                .alias("device_version_id"),
                
                # 신뢰도 매핑
                pl.coalesce([
                    pl.col("match_source").replace(self.config.CONFIDENCE_MAP),
                    pl.lit("VERY_LOW")
                ]).alias("udi_confidence"),
                
                pl.col("match_source").alias("final_source")
            ])
            
            return chunk_lf
        
        output_path = self._new_temp_path("resolved_final.parquet")
        
        process_lazyframe_in_chunks(
            lf=lf,
            transform_func=resolve_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
            desc="Tier 3 처리"
        )
        
        print(f"✅ 최종 결과: {output_path}")
        return output_path  # ✅ Path 반환!

    # ==================== 9단계: 전체 실행 ====================
    
    def process(
        self,
        maude_lf: pl.LazyFrame,
        udi_lf: pl.LazyFrame,
        output_path: Path,
        chunk_size: int = 50_000
    ) -> Path:
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("UDI 처리 파이프라인 시작 (Path 기반)")
        print("=" * 60)
        
        try:
            # 1. 전처리
            maude_lf = self.preprocess_maude(maude_lf)
            udi_lf = self.preprocess_udi_db(udi_lf)
            
            # 2. 제조사명 정규화
            self.normalize_manufacturers(maude_lf, udi_lf)
            maude_lf = self.apply_normalization(maude_lf)
            
            # 3. Lookup 생성
            self.build_lookup(udi_lf)
            
            # 4. UDI 매핑 생성 (Path 받음!)
            mapping_path = self.build_udi_mapping(maude_lf, chunk_size)
            
            # 5. 매칭 적용
            temp_matched_path = self._new_temp_path("maude_matched.parquet")
            self.process_all(maude_lf, mapping_path, temp_matched_path, chunk_size)
            
            # 6. 후처리 (Path 받음!)
            final_temp_path = self._post_process_complex_cases(temp_matched_path, chunk_size)
            
            # join으로 늘어난 중복 제거
            final_lf = pl.scan_parquet(final_temp_path).unique(subset=['mdr_report_key'],keep='first')
            
            # 7. 최종 파일 이동
            final_lf.sink_parquet(output_path)
            
            # 통계
            print("\n" + "=" * 60)
            print("📊 최종 결과")
            print("=" * 60)
            
            result_lf = pl.scan_parquet(output_path)
            total = result_lf.select(pl.len()).collect().item()
            
            match_stats = result_lf.group_by("match_source").agg([
                pl.len().alias("count"),
                (pl.len() / total * 100).round(2).alias("percent")
            ]).collect().sort("count", descending=True)
            
            print("\n매칭 출처 분포:")
            print(match_stats)
            
            conf_stats = result_lf.group_by("udi_confidence").agg([
                pl.len().alias("count"),
                (pl.len() / total * 100).round(2).alias("percent")
            ]).collect().sort("count", descending=True)
            
            print("\n신뢰도 분포:")
            print(conf_stats)
            
            score_stats = result_lf.group_by("match_score").agg([
                pl.len().alias("count"),
                (pl.len() / total * 100).round(2).alias("percent")
            ]).collect().sort("match_score", descending=True)
            
            print("\nScore 분포:")
            print(score_stats)
            
            print(f"\n✅ 총 {total:,} 건 처리 완료!")
            print(f"📁 결과: {output_path}")
            
            return output_path
        
        finally:
            # ✅ temp 삭제는 여기서만!
            if self.config.CLEANUP_TEMP_ON_SUCCESS:
                self._cleanup_temps()
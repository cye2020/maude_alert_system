"""
UDI 매칭 SQL 생성 모듈 (Snowflake)

설계 원칙:
- 각 함수는 SQL 문자열을 반환
- 설정은 개별 인자로 외부 주입
- Snowflake 연결은 이 모듈에서 하지 않음
- 단일 패스 + Window 함수로 효율적 처리
"""

from textwrap import dedent


# ==================== 1단계: 전처리 ====================

def sql_preprocess_maude(
    source_table: str,
    maude_date_cols: list[str],
    di_regex: str
) -> str:
    """
    MAUDE 전처리 SQL

    - udi_public에서 DI 추출
    - 날짜 우선순위 적용
    - udi_combined 생성
    - udi_source 라벨링

    Args:
        source_table: 원본 MAUDE 테이블명
        maude_date_cols: 날짜 컬럼 우선순위 리스트
        di_regex: DI 추출 정규식
    """
    date_coalesce = ", ".join(maude_date_cols)

    return dedent(f"""
        SELECT *,
            COALESCE({date_coalesce}) AS report_date,
            COALESCE(udi_di, REGEXP_SUBSTR(udi_public, '{di_regex}', 1, 1, 'e')) AS udi_combined
        FROM {source_table}
    """).strip()


def sql_preprocess_udi(
    source_table: str,
    udi_date_cols: list[str]
) -> str:
    """
    UDI DB 전처리 SQL

    - 날짜 우선순위 적용하여 publish_date 생성

    Args:
        source_table: 원본 UDI 테이블명
        udi_date_cols: 날짜 컬럼 우선순위 리스트
    """
    date_coalesce = ", ".join(udi_date_cols)

    return dedent(f"""
        SELECT
            *,
            COALESCE({date_coalesce}) AS publish_date
        FROM {source_table}
    """).strip()


# ==================== 2단계: 제조사 정규화 ====================

def sql_create_fuzzy_match_udf() -> str:
    """
    Fuzzy matching Python UDF 생성 SQL

    Note: rapidfuzz 패키지 필요
    """
    return dedent("""
        CREATE OR REPLACE FUNCTION fuzzy_match_score(source VARCHAR, target VARCHAR)
        RETURNS FLOAT
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.11'
        PACKAGES = ('rapidfuzz')
        HANDLER = 'calculate_score'
        AS $$
def calculate_score(source, target):
    if source is None or target is None:
        return 0.0
    from rapidfuzz import fuzz
    return fuzz.ratio(source.upper(), target.upper()) / 100.0
$$;
    """).strip()


def sql_build_manufacturer_mapping(
    maude_table: str,
    udi_table: str,
    threshold: float
) -> str:
    """
    제조사명 퍼지 매칭 테이블 생성 SQL

    Args:
        maude_table: MAUDE 테이블명
        udi_table: UDI 테이블명
        threshold: 매칭 임계값 (0.0 ~ 1.0)
    """
    return dedent(f"""
        WITH maude_mfrs AS (
            SELECT DISTINCT manufacturer FROM {maude_table}
            WHERE manufacturer IS NOT NULL
        ),
        udi_mfrs AS (
            SELECT DISTINCT manufacturer FROM {udi_table}
            WHERE manufacturer IS NOT NULL
        ),
        cross_matched AS (
            SELECT
                m.manufacturer AS maude_mfr,
                u.manufacturer AS udi_mfr,
                fuzzy_match_score(m.manufacturer, u.manufacturer) AS score
            FROM maude_mfrs m
            CROSS JOIN udi_mfrs u
        ),
        best_match AS (
            SELECT
                maude_mfr,
                udi_mfr,
                score
            FROM cross_matched
            WHERE score >= {threshold}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY maude_mfr ORDER BY score DESC) = 1
        )
        SELECT
            m.manufacturer AS original_mfr,
            COALESCE(b.udi_mfr, m.manufacturer) AS standardized_mfr
        FROM maude_mfrs m
        LEFT JOIN best_match b ON m.manufacturer = b.maude_mfr
    """).strip()


def sql_apply_manufacturer_normalization(
    source_table: str,
    mapping_table: str
) -> str:
    """
    제조사명 정규화 적용 SQL

    Args:
        source_table: 정규화 대상 테이블
        mapping_table: 제조사 매핑 테이블
    """
    return dedent(f"""
        SELECT
            s.*,
            COALESCE(m.standardized_mfr, s.manufacturer) AS mfr_std
        FROM {source_table} s
        LEFT JOIN {mapping_table} m ON s.manufacturer = m.original_mfr
    """).strip()


# ==================== 3단계: Lookup 테이블 생성 ====================

def sql_create_primary_lookup(udi_table: str) -> str:
    """
    Primary UDI Lookup 테이블 생성 SQL

    - udi_di 기준 unique

    Args:
        udi_table: UDI 테이블명
    """
    return dedent(f"""
        SELECT DISTINCT
            udi_di,
            manufacturer,
            brand,
            model_number,
            catalog_number,
            publish_date
        FROM {udi_table}
        WHERE udi_di IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY udi_di ORDER BY publish_date DESC NULLS LAST) = 1
    """).strip()


def sql_create_full_lookup_with_secondary(
    udi_table: str,
    secondary_columns: list[str]
) -> str:
    """
    Secondary 식별자 포함 Full Lookup 생성 SQL

    Args:
        udi_table: UDI 테이블명
        secondary_columns: Secondary 식별자 컬럼명 리스트
    """
    if secondary_columns:
        cols_str = ", ".join(secondary_columns)
        array_construct = f"ARRAY_COMPACT(ARRAY_CONSTRUCT({cols_str}))"
    else:
        array_construct = "ARRAY_CONSTRUCT()"

    return dedent(f"""
        SELECT
            udi_di,
            manufacturer,
            brand,
            model_number,
            catalog_number,
            publish_date,
            {array_construct} AS secondary_list
        FROM {udi_table}
    """).strip()


# ==================== 4단계: Primary 직접 매칭 ====================

def sql_primary_match(
    maude_table: str,
    udi_lookup: str,
    match_type_direct: str
) -> str:
    """
    Primary UDI 직접 매칭 SQL

    - udi_combined = udi_di 직접 조인
    - match_score = 3 (최고 점수)

    Args:
        maude_table: MAUDE 테이블명
        udi_lookup: UDI Lookup 테이블명
        match_type_direct: 직접 매칭 타입 라벨
    """
    return dedent(f"""
        SELECT
            m.mfr_std,
            m.brand,
            m.model_number,
            m.catalog_number,
            m.udi_combined,
            m.udi_combined AS mapped_primary_udi,
            u.manufacturer AS mapped_manufacturer,
            u.brand AS mapped_brand,
            u.model_number AS mapped_model_number,
            u.catalog_number AS mapped_catalog_number,
            '{match_type_direct}' AS udi_match_type,
            3 AS match_score
        FROM {maude_table} m
        INNER JOIN {udi_lookup} u
            ON m.udi_combined = u.udi_di
    """).strip()


def sql_primary_failed(
    maude_table: str,
    primary_matched: str
) -> str:
    """
    Primary 매칭 실패 케이스 추출 SQL

    Args:
        maude_table: MAUDE 테이블명
        primary_matched: Primary 매칭 성공 테이블명
    """
    return dedent(f"""
        SELECT m.*
        FROM {maude_table} m
        LEFT JOIN {primary_matched} p ON m.udi_combined = p.udi_combined
        WHERE p.udi_combined IS NULL
    """).strip()


# ==================== 5단계: Secondary 매칭 ====================

def sql_explode_secondary(udi_full_table: str) -> str:
    """
    Secondary 식별자 FLATTEN SQL

    Args:
        udi_full_table: Secondary 포함 UDI 테이블명
    """
    return dedent(f"""
        SELECT
            udi_di,
            manufacturer,
            brand,
            model_number,
            catalog_number,
            publish_date,
            f.value::STRING AS secondary_id
        FROM {udi_full_table},
        LATERAL FLATTEN(input => secondary_list, OUTER => FALSE) f
    """).strip()


def sql_secondary_match(
    candidates_table: str,
    exploded_udi: str,
    score_weights: dict[str, int],
    require_unique_primary: bool,
    min_match_score: int,
    match_type_secondary: str
) -> str:
    """
    Secondary UDI 매칭 SQL (단일 패스 + Window)

    - udi_combined가 secondary_list에 존재하는 경우
    - Score 기반 최적 매칭 선택

    Args:
        candidates_table: 후보 테이블명
        exploded_udi: Exploded UDI 테이블명
        score_weights: 필드별 가중치 딕셔너리
        require_unique_primary: Primary UDI 유일성 요구 여부
        min_match_score: 최소 매칭 점수
        match_type_secondary: Secondary 매칭 타입 라벨
    """
    score_expr = _build_score_expr(score_weights, "c", "e")
    unique_filter = "n_primary = 1" if require_unique_primary else "TRUE"

    return dedent(f"""
        WITH scored AS (
            SELECT
                c.mfr_std,
                c.brand,
                c.model_number,
                c.catalog_number,
                c.udi_combined,
                c.report_date,
                e.udi_di,
                e.brand AS brand_r,
                e.model_number AS model_r,
                e.catalog_number AS catalog_r,
                ({score_expr}) AS match_score,
                COUNT(DISTINCT e.udi_di) OVER (
                    PARTITION BY c.udi_combined, c.mfr_std, c.brand, c.model_number, c.catalog_number
                ) AS n_primary
            FROM {candidates_table} c
            INNER JOIN {exploded_udi} e
                ON c.udi_combined = e.secondary_id
                AND c.mfr_std = e.manufacturer
            WHERE e.publish_date < c.report_date
        )
        SELECT
            mfr_std,
            brand,
            model_number,
            catalog_number,
            udi_combined,
            udi_di AS mapped_primary_udi,
            mfr_std AS mapped_manufacturer,
            brand_r AS mapped_brand,
            model_r AS mapped_model_number,
            catalog_r AS mapped_catalog_number,
            '{match_type_secondary}' AS udi_match_type,
            match_score
        FROM scored
        WHERE match_score >= {min_match_score} AND {unique_filter}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY udi_combined
            ORDER BY match_score DESC
        ) = 1
    """).strip()


def sql_secondary_failed(
    candidates_table: str,
    secondary_matched: str,
    match_type_udi_no_match: str
) -> str:
    """
    Secondary 매칭 실패 케이스 SQL

    Args:
        candidates_table: 후보 테이블명
        secondary_matched: Secondary 매칭 성공 테이블명
        match_type_udi_no_match: UDI 매칭 실패 타입 라벨
    """
    return dedent(f"""
        SELECT
            c.mfr_std,
            c.brand,
            c.model_number,
            c.catalog_number,
            c.udi_combined,
            c.udi_combined AS mapped_primary_udi,
            NULL AS mapped_manufacturer,
            NULL AS mapped_brand,
            NULL AS mapped_model_number,
            NULL AS mapped_catalog_number,
            '{match_type_udi_no_match}' AS udi_match_type,
            0 AS match_score
        FROM {candidates_table} c
        LEFT JOIN {secondary_matched} s ON c.udi_combined = s.udi_combined
        WHERE s.udi_combined IS NULL
    """).strip()


# ==================== 6단계: No UDI 매칭 ====================

def sql_no_udi_candidates(source_table: str) -> str:
    """
    No UDI 후보 추출 SQL (udi_combined IS NULL)

    Args:
        source_table: 원본 테이블명
    """
    return dedent(f"""
        SELECT DISTINCT
            mfr_std,
            brand,
            model_number,
            catalog_number,
            udi_combined,
            report_date
        FROM {source_table}
        WHERE udi_combined IS NULL
    """).strip()


def sql_no_udi_match(
    candidates_table: str,
    udi_lookup: str,
    score_weights: dict[str, int],
    require_unique_primary: bool,
    min_match_score: int,
    match_type_meta: str
) -> str:
    """
    No UDI 메타데이터 기반 매칭 SQL (단일 패스 + Window)

    - 제조사 + 브랜드 + 모델/카탈로그 조합으로 매칭

    Args:
        candidates_table: 후보 테이블명
        udi_lookup: UDI Lookup 테이블명
        score_weights: 필드별 가중치 딕셔너리
        require_unique_primary: Primary UDI 유일성 요구 여부
        min_match_score: 최소 매칭 점수
        match_type_meta: 메타 매칭 타입 라벨
    """
    score_expr = _build_score_expr(score_weights, "c", "u")
    unique_filter = "n_primary = 1" if require_unique_primary else "TRUE"

    return dedent(f"""
        WITH scored AS (
            SELECT
                c.mfr_std,
                c.brand,
                c.model_number,
                c.catalog_number,
                c.udi_combined,
                c.report_date,
                u.udi_di,
                u.brand AS brand_r,
                u.model_number AS model_r,
                u.catalog_number AS catalog_r,
                ({score_expr}) AS match_score,
                COUNT(DISTINCT u.udi_di) OVER (
                    PARTITION BY c.mfr_std, c.brand, c.model_number, c.catalog_number
                ) AS n_primary
            FROM {candidates_table} c
            INNER JOIN {udi_lookup} u
                ON c.mfr_std = u.manufacturer
            WHERE u.publish_date < c.report_date
        )
        SELECT
            mfr_std,
            brand,
            model_number,
            catalog_number,
            udi_combined,
            udi_di AS mapped_primary_udi,
            mfr_std AS mapped_manufacturer,
            brand_r AS mapped_brand,
            model_r AS mapped_model_number,
            catalog_r AS mapped_catalog_number,
            '{match_type_meta}' AS udi_match_type,
            match_score
        FROM scored
        WHERE match_score >= {min_match_score} AND {unique_filter}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY mfr_std, brand, model_number, catalog_number
            ORDER BY match_score DESC
        ) = 1
    """).strip()


def sql_no_udi_failed(
    candidates_table: str,
    no_udi_matched: str,
    match_type_no_match: str
) -> str:
    """
    No UDI 매칭 실패 케이스 SQL

    Args:
        candidates_table: 후보 테이블명
        no_udi_matched: No UDI 매칭 성공 테이블명
        match_type_no_match: 매칭 실패 타입 라벨
    """
    return dedent(f"""
        SELECT
            c.mfr_std,
            c.brand,
            c.model_number,
            c.catalog_number,
            c.udi_combined,
            NULL AS mapped_primary_udi,
            NULL AS mapped_manufacturer,
            NULL AS mapped_brand,
            NULL AS mapped_model_number,
            NULL AS mapped_catalog_number,
            '{match_type_no_match}' AS udi_match_type,
            0 AS match_score
        FROM {candidates_table} c
        LEFT JOIN {no_udi_matched} n
            ON c.mfr_std = n.mfr_std
            AND c.brand = n.brand
            AND c.model_number = n.model_number
            AND c.catalog_number = n.catalog_number
        WHERE n.mfr_std IS NULL
    """).strip()


# ==================== 7단계: 매핑 통합 ====================

def sql_union_all_mappings(
    primary_matched: str,
    secondary_matched: str,
    secondary_failed: str,
    no_udi_matched: str,
    no_udi_failed: str
) -> str:
    """
    모든 매핑 결과 통합 SQL

    Args:
        primary_matched: Primary 매칭 테이블명
        secondary_matched: Secondary 매칭 테이블명
        secondary_failed: Secondary 실패 테이블명
        no_udi_matched: No UDI 매칭 테이블명
        no_udi_failed: No UDI 실패 테이블명
    """
    return dedent(f"""
        SELECT * FROM {primary_matched}
        UNION ALL
        SELECT * FROM {secondary_matched}
        UNION ALL
        SELECT * FROM {secondary_failed}
        UNION ALL
        SELECT * FROM {no_udi_matched}
        UNION ALL
        SELECT * FROM {no_udi_failed}
    """).strip()


# ==================== 8단계: 매핑 적용 ====================

def sql_apply_mapping_with_udi(
    maude_table: str,
    mapping_table: str,
    match_type_not_in_mapping: str
) -> str:
    """
    UDI 있는 케이스에 매핑 적용 SQL

    Args:
        maude_table: MAUDE 테이블명
        mapping_table: 매핑 테이블명
        match_type_not_in_mapping: 매핑 없음 타입 라벨
    """
    return dedent(f"""
        SELECT
            m.*,
            COALESCE(mp.mapped_primary_udi, m.udi_combined) AS device_version_id,
            COALESCE(mp.mapped_manufacturer, m.manufacturer) AS manufacturer_final,
            COALESCE(mp.mapped_brand, m.brand) AS brand_final,
            COALESCE(mp.mapped_model_number, m.model_number) AS model_number_final,
            COALESCE(mp.mapped_catalog_number, m.catalog_number) AS catalog_number_final,
            COALESCE(mp.udi_match_type, '{match_type_not_in_mapping}') AS match_source,
            mp.match_score
        FROM {maude_table} m
        LEFT JOIN {mapping_table} mp ON m.udi_combined = mp.udi_combined
        WHERE m.udi_combined IS NOT NULL
    """).strip()


def sql_apply_mapping_no_udi(
    maude_table: str,
    mapping_table: str,
    match_type_not_in_mapping: str
) -> str:
    """
    UDI 없는 케이스에 매핑 적용 SQL

    Args:
        maude_table: MAUDE 테이블명
        mapping_table: 매핑 테이블명
        match_type_not_in_mapping: 매핑 없음 타입 라벨
    """
    return dedent(f"""
        SELECT
            m.*,
            mp.mapped_primary_udi AS device_version_id,
            COALESCE(mp.mapped_manufacturer, m.manufacturer) AS manufacturer_final,
            COALESCE(mp.mapped_brand, m.brand) AS brand_final,
            COALESCE(mp.mapped_model_number, m.model_number) AS model_number_final,
            COALESCE(mp.mapped_catalog_number, m.catalog_number) AS catalog_number_final,
            COALESCE(mp.udi_match_type, '{match_type_not_in_mapping}') AS match_source,
            mp.match_score
        FROM {maude_table} m
        LEFT JOIN {mapping_table} mp
            ON m.mfr_std = mp.mfr_std
            AND m.brand = mp.brand
            AND m.model_number = mp.model_number
            AND m.catalog_number = mp.catalog_number
        WHERE m.udi_combined IS NULL
    """).strip()


def sql_apply_mapping_combined(
    maude_table: str,
    mapping_table: str,
    match_type_not_in_mapping: str
) -> str:
    """
    전체 매핑 적용 SQL (UDI 유무 통합)

    Args:
        maude_table: MAUDE 테이블명
        mapping_table: 매핑 테이블명
        match_type_not_in_mapping: 매핑 없음 타입 라벨
    """
    with_udi = sql_apply_mapping_with_udi(maude_table, mapping_table, match_type_not_in_mapping)
    no_udi = sql_apply_mapping_no_udi(maude_table, mapping_table, match_type_not_in_mapping)

    return dedent(f"""
        WITH with_udi AS (
            {with_udi}
        ),
        no_udi AS (
            {no_udi}
        )
        SELECT * FROM with_udi
        UNION ALL
        SELECT * FROM no_udi
    """).strip()


# ==================== 9단계: 후처리 (Tier 3) ====================

def sql_identify_low_compliance_manufacturers(
    source_table: str,
    low_compliance_threshold: float
) -> str:
    """
    Low Compliance 제조사 식별 SQL

    Args:
        source_table: 원본 테이블명
        low_compliance_threshold: UDI 누락률 임계값
    """
    return dedent(f"""
        SELECT mfr_std
        FROM {source_table}
        GROUP BY mfr_std
        HAVING SUM(IFF(udi_combined IS NULL, 1, 0))::FLOAT / COUNT(*) > {low_compliance_threshold}
    """).strip()


def sql_tier3_fallback(
    input_table: str,
    low_compliance_table: str,
    confidence_map: dict[str, str],
    match_type_no_match: str,
    match_type_not_in_mapping: str,
    match_type_udi_no_match: str
) -> str:
    """
    Tier 3 폴백 ID 생성 + 신뢰도 매핑 SQL

    Args:
        input_table: 입력 테이블명
        low_compliance_table: Low compliance 제조사 테이블명
        confidence_map: 매칭 타입 -> 신뢰도 매핑
        match_type_no_match: 매칭 실패 타입 라벨
        match_type_not_in_mapping: 매핑 없음 타입 라벨
        match_type_udi_no_match: UDI 매칭 실패 타입 라벨
    """
    # 신뢰도 CASE 문 생성
    confidence_cases = " ".join(
        f"WHEN match_source = '{k}' THEN '{v}'"
        for k, v in confidence_map.items()
    )

    # 매칭 실패 타입들
    no_match_types = [match_type_no_match, match_type_not_in_mapping, match_type_udi_no_match]
    no_match_list = ", ".join(f"'{t}'" for t in no_match_types)

    return dedent(f"""
        SELECT
            t.*,
            CASE
                WHEN match_source IN ({no_match_list}) THEN
                    CASE
                        WHEN t.mfr_std IN (SELECT mfr_std FROM {low_compliance_table})
                        THEN CONCAT('LOW_', t.mfr_std, '_', COALESCE(t.brand_final, 'UNKNOWN'))
                        ELSE CONCAT('UNK_', t.mfr_std, '_', COALESCE(t.brand_final, 'UNKNOWN'),
                                    '_', COALESCE(t.catalog_number_final, 'NA'))
                    END
                ELSE t.device_version_id
            END AS device_version_id_final,
            CASE {confidence_cases} ELSE 'VERY_LOW' END AS udi_confidence,
            t.match_source AS final_source
        FROM {input_table} t
    """).strip()


# ==================== 10단계: 최종 정리 ====================

def sql_deduplicate_final(
    input_table: str,
    unique_key: str
) -> str:
    """
    최종 결과 중복 제거 SQL

    Args:
        input_table: 입력 테이블명
        unique_key: 중복 제거 기준 컬럼
    """
    return dedent(f"""
        SELECT *
        FROM {input_table}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY {unique_key}
            ORDER BY match_score DESC NULLS LAST
        ) = 1
    """).strip()


def sql_final_statistics(result_table: str) -> str:
    """
    최종 통계 SQL

    Args:
        result_table: 결과 테이블명
    """
    return dedent(f"""
        WITH stats AS (
            SELECT
                COUNT(*) AS total_count,
                match_source,
                udi_confidence,
                match_score
            FROM {result_table}
            GROUP BY match_source, udi_confidence, match_score
        )
        SELECT
            'match_source' AS stat_type,
            match_source AS category,
            SUM(total_count) AS count,
            ROUND(SUM(total_count) * 100.0 / SUM(SUM(total_count)) OVER (), 2) AS percent
        FROM stats
        GROUP BY match_source

        UNION ALL

        SELECT
            'udi_confidence' AS stat_type,
            udi_confidence AS category,
            SUM(total_count) AS count,
            ROUND(SUM(total_count) * 100.0 / SUM(SUM(total_count)) OVER (), 2) AS percent
        FROM stats
        GROUP BY udi_confidence

        UNION ALL

        SELECT
            'match_score' AS stat_type,
            match_score::VARCHAR AS category,
            SUM(total_count) AS count,
            ROUND(SUM(total_count) * 100.0 / SUM(SUM(total_count)) OVER (), 2) AS percent
        FROM stats
        GROUP BY match_score

        ORDER BY stat_type, count DESC
    """).strip()


# ==================== 헬퍼 함수 ====================

def _build_score_expr(
    weights: dict[str, int],
    left_alias: str,
    right_alias: str
) -> str:
    """
    Score 계산 SQL 표현식 생성

    Args:
        weights: 필드별 가중치 딕셔너리
        left_alias: 왼쪽 테이블 alias (MAUDE)
        right_alias: 오른쪽 테이블 alias (UDI)
    """
    parts = []

    if weights.get("brand"):
        parts.append(
            f"IFF({left_alias}.brand = {right_alias}.brand_r, {weights['brand']}, 0)"
        )

    if weights.get("model_number"):
        parts.append(
            f"IFF({left_alias}.model_number IS NOT NULL "
            f"AND {right_alias}.model_r IS NOT NULL "
            f"AND {left_alias}.model_number = {right_alias}.model_r, "
            f"{weights['model_number']}, 0)"
        )

    if weights.get("catalog_number"):
        parts.append(
            f"IFF({left_alias}.catalog_number IS NOT NULL "
            f"AND {right_alias}.catalog_r IS NOT NULL "
            f"AND {left_alias}.catalog_number = {right_alias}.catalog_r, "
            f"{weights['catalog_number']}, 0)"
        )

    return " + ".join(parts) if parts else "0"


# ==================== 전체 파이프라인 (단일 쿼리) ====================

def sql_full_matching_pipeline(
    maude_preprocessed: str,
    udi_primary_lookup: str,
    udi_full_lookup: str,
    score_weights: dict[str, int],
    require_unique_primary: bool,
    min_match_score: int,
    match_type_direct: str,
    match_type_secondary: str,
    match_type_udi_no_match: str,
    match_type_meta: str,
    match_type_no_match: str
) -> str:
    """
    전체 매칭 파이프라인 (단일 CTE 체인)

    단계:
    1. Primary 직접 매칭
    2. Secondary 매칭 (UDI 있음)
    3. No UDI 매칭 (UDI 없음)
    4. 실패 케이스 처리
    5. 통합

    Args:
        maude_preprocessed: 전처리된 MAUDE 테이블
        udi_primary_lookup: Primary UDI Lookup 테이블
        udi_full_lookup: Secondary 포함 Full Lookup 테이블
        score_weights: 필드별 가중치 딕셔너리
        require_unique_primary: Primary UDI 유일성 요구 여부
        min_match_score: 최소 매칭 점수
        match_type_direct: 직접 매칭 타입 라벨
        match_type_secondary: Secondary 매칭 타입 라벨
        match_type_udi_no_match: UDI 매칭 실패 타입 라벨
        match_type_meta: 메타 매칭 타입 라벨
        match_type_no_match: 매칭 실패 타입 라벨
    """
    unique_filter = "n_primary = 1" if require_unique_primary else "TRUE"

    brand_w = score_weights.get("brand", 0)
    model_w = score_weights.get("model_number", 0)
    catalog_w = score_weights.get("catalog_number", 0)

    return dedent(f"""
        WITH
        -- ========== Unique UDI 추출 ==========
        unique_udi AS (
            SELECT DISTINCT
                mfr_std, brand, model_number, catalog_number,
                udi_combined, report_date
            FROM {maude_preprocessed}
        ),

        -- ========== 1. Primary 직접 매칭 ==========
        primary_matched AS (
            SELECT
                u.mfr_std, u.brand, u.model_number, u.catalog_number, u.udi_combined,
                u.udi_combined AS mapped_primary_udi,
                p.manufacturer AS mapped_manufacturer,
                p.brand AS mapped_brand,
                p.model_number AS mapped_model_number,
                p.catalog_number AS mapped_catalog_number,
                '{match_type_direct}' AS udi_match_type,
                3 AS match_score
            FROM unique_udi u
            INNER JOIN {udi_primary_lookup} p ON u.udi_combined = p.udi_di
        ),

        primary_failed AS (
            SELECT u.*
            FROM unique_udi u
            LEFT JOIN primary_matched pm ON u.udi_combined = pm.udi_combined
            WHERE pm.udi_combined IS NULL
        ),

        -- ========== 2. Secondary 매칭 ==========
        secondary_candidates AS (
            SELECT * FROM primary_failed WHERE udi_combined IS NOT NULL
        ),

        exploded_secondary AS (
            SELECT
                udi_di, manufacturer, brand, model_number, catalog_number, publish_date,
                f.value::STRING AS secondary_id
            FROM {udi_full_lookup},
            LATERAL FLATTEN(input => secondary_list, OUTER => FALSE) f
        ),

        secondary_scored AS (
            SELECT
                c.mfr_std, c.brand, c.model_number, c.catalog_number,
                c.udi_combined, c.report_date,
                e.udi_di,
                e.brand AS brand_r,
                e.model_number AS model_r,
                e.catalog_number AS catalog_r,
                (
                    IFF(c.brand = e.brand, {brand_w}, 0) +
                    IFF(c.model_number IS NOT NULL AND e.model_number IS NOT NULL
                        AND c.model_number = e.model_number, {model_w}, 0) +
                    IFF(c.catalog_number IS NOT NULL AND e.catalog_number IS NOT NULL
                        AND c.catalog_number = e.catalog_number, {catalog_w}, 0)
                ) AS match_score,
                COUNT(DISTINCT e.udi_di) OVER (
                    PARTITION BY c.udi_combined, c.mfr_std, c.brand, c.model_number, c.catalog_number
                ) AS n_primary
            FROM secondary_candidates c
            INNER JOIN exploded_secondary e
                ON c.udi_combined = e.secondary_id AND c.mfr_std = e.manufacturer
            WHERE e.publish_date < c.report_date
        ),

        secondary_matched AS (
            SELECT
                mfr_std, brand, model_number, catalog_number, udi_combined,
                udi_di AS mapped_primary_udi,
                mfr_std AS mapped_manufacturer,
                brand_r AS mapped_brand,
                model_r AS mapped_model_number,
                catalog_r AS mapped_catalog_number,
                '{match_type_secondary}' AS udi_match_type,
                match_score
            FROM secondary_scored
            WHERE match_score >= {min_match_score} AND {unique_filter}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY udi_combined ORDER BY match_score DESC) = 1
        ),

        secondary_failed AS (
            SELECT
                c.mfr_std, c.brand, c.model_number, c.catalog_number, c.udi_combined,
                c.udi_combined AS mapped_primary_udi,
                NULL AS mapped_manufacturer,
                NULL AS mapped_brand,
                NULL AS mapped_model_number,
                NULL AS mapped_catalog_number,
                '{match_type_udi_no_match}' AS udi_match_type,
                0 AS match_score
            FROM secondary_candidates c
            LEFT JOIN secondary_matched sm ON c.udi_combined = sm.udi_combined
            WHERE sm.udi_combined IS NULL
        ),

        -- ========== 3. No UDI 매칭 ==========
        no_udi_candidates AS (
            SELECT DISTINCT mfr_std, brand, model_number, catalog_number, udi_combined, report_date
            FROM primary_failed
            WHERE udi_combined IS NULL
        ),

        no_udi_scored AS (
            SELECT
                c.mfr_std, c.brand, c.model_number, c.catalog_number,
                c.udi_combined, c.report_date,
                p.udi_di,
                p.brand AS brand_r,
                p.model_number AS model_r,
                p.catalog_number AS catalog_r,
                (
                    IFF(c.brand = p.brand, {brand_w}, 0) +
                    IFF(c.model_number IS NOT NULL AND p.model_number IS NOT NULL
                        AND c.model_number = p.model_number, {model_w}, 0) +
                    IFF(c.catalog_number IS NOT NULL AND p.catalog_number IS NOT NULL
                        AND c.catalog_number = p.catalog_number, {catalog_w}, 0)
                ) AS match_score,
                COUNT(DISTINCT p.udi_di) OVER (
                    PARTITION BY c.mfr_std, c.brand, c.model_number, c.catalog_number
                ) AS n_primary
            FROM no_udi_candidates c
            INNER JOIN {udi_primary_lookup} p ON c.mfr_std = p.manufacturer
            WHERE p.publish_date < c.report_date
        ),

        no_udi_matched AS (
            SELECT
                mfr_std, brand, model_number, catalog_number, udi_combined,
                udi_di AS mapped_primary_udi,
                mfr_std AS mapped_manufacturer,
                brand_r AS mapped_brand,
                model_r AS mapped_model_number,
                catalog_r AS mapped_catalog_number,
                '{match_type_meta}' AS udi_match_type,
                match_score
            FROM no_udi_scored
            WHERE match_score >= {min_match_score} AND {unique_filter}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY mfr_std, brand, model_number, catalog_number
                ORDER BY match_score DESC
            ) = 1
        ),

        no_udi_failed AS (
            SELECT
                c.mfr_std, c.brand, c.model_number, c.catalog_number, c.udi_combined,
                NULL AS mapped_primary_udi,
                NULL AS mapped_manufacturer,
                NULL AS mapped_brand,
                NULL AS mapped_model_number,
                NULL AS mapped_catalog_number,
                '{match_type_no_match}' AS udi_match_type,
                0 AS match_score
            FROM no_udi_candidates c
            LEFT JOIN no_udi_matched nm
                ON c.mfr_std = nm.mfr_std
                AND c.brand = nm.brand
                AND c.model_number = nm.model_number
                AND c.catalog_number = nm.catalog_number
            WHERE nm.mfr_std IS NULL
        )

        -- ========== 4. 통합 ==========
        SELECT * FROM primary_matched
        UNION ALL SELECT * FROM secondary_matched
        UNION ALL SELECT * FROM secondary_failed
        UNION ALL SELECT * FROM no_udi_matched
        UNION ALL SELECT * FROM no_udi_failed
    """).strip()


# ==================== 테스트 ====================

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename='temp.log',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 테스트용 설정값
    TEST_CONFIG = {
        "maude_date_cols": ["date_report", "date_event", "date_received"],
        "udi_date_cols": ["publish_date", "version_date"],
        "di_regex": r"\(01\)(\d{14})",
        "fuzzy_threshold": 0.85,
        "secondary_columns": ["secondary_di_1", "secondary_di_2"],
        "score_weights": {"brand": 1, "model_number": 1, "catalog_number": 1},
        "require_unique_primary": True,
        "min_match_score": 2,
        "low_compliance_threshold": 0.5,
        "confidence_map": {
            "DIRECT": "HIGH",
            "SECONDARY": "MEDIUM",
            "META": "LOW",
            "UDI_NO_MATCH": "VERY_LOW",
            "NO_MATCH": "VERY_LOW",
        },
        "match_types": {
            "direct": "DIRECT",
            "secondary": "SECONDARY",
            "meta": "META",
            "udi_no_match": "UDI_NO_MATCH",
            "no_match": "NO_MATCH",
            "not_in_mapping": "NOT_IN_MAPPING",
        },
    }

    def test_step(step_name: str, sql_func, *args, **kwargs):
        """단계별 SQL 생성 테스트 및 로깅"""
        logger.info(f"{'='*60}")
        logger.info(f"[{step_name}] 시작")
        try:
            sql = sql_func(*args, **kwargs)
            logger.info(f"[{step_name}] SQL 생성 완료")
            logger.info(f"생성된 SQL:\n{sql}")
            logger.info(f"[{step_name}] 성공 ✓")
            return sql
        except Exception as e:
            logger.error(f"[{step_name}] 실패: {e}")
            raise

    # ========== 1단계: 전처리 ==========
    logger.info("\n" + "="*70)
    logger.info("1단계: 전처리")
    logger.info("="*70)

    test_step(
        "1-1. MAUDE 전처리",
        sql_preprocess_maude,
        source_table="raw_maude",
        maude_date_cols=TEST_CONFIG["maude_date_cols"],
        di_regex=TEST_CONFIG["di_regex"],
    )

    test_step(
        "1-2. UDI 전처리",
        sql_preprocess_udi,
        source_table="raw_udi",
        udi_date_cols=TEST_CONFIG["udi_date_cols"],
    )

    # ========== 2단계: 제조사 정규화 ==========
    logger.info("\n" + "="*70)
    logger.info("2단계: 제조사 정규화")
    logger.info("="*70)

    test_step(
        "2-1. Fuzzy Match UDF 생성",
        sql_create_fuzzy_match_udf,
    )

    test_step(
        "2-2. 제조사 매핑 테이블 생성",
        sql_build_manufacturer_mapping,
        maude_table="maude_preprocessed",
        udi_table="udi_preprocessed",
        threshold=TEST_CONFIG["fuzzy_threshold"],
    )

    test_step(
        "2-3. 제조사 정규화 적용",
        sql_apply_manufacturer_normalization,
        source_table="maude_preprocessed",
        mapping_table="manufacturer_mapping",
    )

    # ========== 3단계: Lookup 테이블 생성 ==========
    logger.info("\n" + "="*70)
    logger.info("3단계: Lookup 테이블 생성")
    logger.info("="*70)

    test_step(
        "3-1. Primary Lookup 생성",
        sql_create_primary_lookup,
        udi_table="udi_normalized",
    )

    test_step(
        "3-2. Full Lookup (Secondary 포함) 생성",
        sql_create_full_lookup_with_secondary,
        udi_table="udi_normalized",
        secondary_columns=TEST_CONFIG["secondary_columns"],
    )

    # ========== 4단계: Primary 직접 매칭 ==========
    logger.info("\n" + "="*70)
    logger.info("4단계: Primary 직접 매칭")
    logger.info("="*70)

    test_step(
        "4-1. Primary 매칭",
        sql_primary_match,
        maude_table="maude_normalized",
        udi_lookup="udi_primary_lookup",
        match_type_direct=TEST_CONFIG["match_types"]["direct"],
    )

    test_step(
        "4-2. Primary 매칭 실패 추출",
        sql_primary_failed,
        maude_table="maude_normalized",
        primary_matched="primary_matched",
    )

    # ========== 5단계: Secondary 매칭 ==========
    logger.info("\n" + "="*70)
    logger.info("5단계: Secondary 매칭")
    logger.info("="*70)

    test_step(
        "5-1. Secondary Explode",
        sql_explode_secondary,
        udi_full_table="udi_full_lookup",
    )

    test_step(
        "5-2. Secondary 매칭",
        sql_secondary_match,
        candidates_table="primary_failed",
        exploded_udi="exploded_secondary",
        score_weights=TEST_CONFIG["score_weights"],
        require_unique_primary=TEST_CONFIG["require_unique_primary"],
        min_match_score=TEST_CONFIG["min_match_score"],
        match_type_secondary=TEST_CONFIG["match_types"]["secondary"],
    )

    test_step(
        "5-3. Secondary 매칭 실패 추출",
        sql_secondary_failed,
        candidates_table="secondary_candidates",
        secondary_matched="secondary_matched",
        match_type_udi_no_match=TEST_CONFIG["match_types"]["udi_no_match"],
    )

    # ========== 6단계: No UDI 매칭 ==========
    logger.info("\n" + "="*70)
    logger.info("6단계: No UDI 매칭")
    logger.info("="*70)

    test_step(
        "6-1. No UDI 후보 추출",
        sql_no_udi_candidates,
        source_table="maude_normalized",
    )

    test_step(
        "6-2. No UDI 매칭",
        sql_no_udi_match,
        candidates_table="no_udi_candidates",
        udi_lookup="udi_primary_lookup",
        score_weights=TEST_CONFIG["score_weights"],
        require_unique_primary=TEST_CONFIG["require_unique_primary"],
        min_match_score=TEST_CONFIG["min_match_score"],
        match_type_meta=TEST_CONFIG["match_types"]["meta"],
    )

    test_step(
        "6-3. No UDI 매칭 실패 추출",
        sql_no_udi_failed,
        candidates_table="no_udi_candidates",
        no_udi_matched="no_udi_matched",
        match_type_no_match=TEST_CONFIG["match_types"]["no_match"],
    )

    # ========== 7단계: 매핑 통합 ==========
    logger.info("\n" + "="*70)
    logger.info("7단계: 매핑 통합")
    logger.info("="*70)

    test_step(
        "7-1. 모든 매핑 통합",
        sql_union_all_mappings,
        primary_matched="primary_matched",
        secondary_matched="secondary_matched",
        secondary_failed="secondary_failed",
        no_udi_matched="no_udi_matched",
        no_udi_failed="no_udi_failed",
    )

    # ========== 8단계: 매핑 적용 ==========
    logger.info("\n" + "="*70)
    logger.info("8단계: 매핑 적용")
    logger.info("="*70)

    test_step(
        "8-1. UDI 있는 케이스 매핑 적용",
        sql_apply_mapping_with_udi,
        maude_table="maude_normalized",
        mapping_table="unified_mapping",
        match_type_not_in_mapping=TEST_CONFIG["match_types"]["not_in_mapping"],
    )

    test_step(
        "8-2. UDI 없는 케이스 매핑 적용",
        sql_apply_mapping_no_udi,
        maude_table="maude_normalized",
        mapping_table="unified_mapping",
        match_type_not_in_mapping=TEST_CONFIG["match_types"]["not_in_mapping"],
    )

    test_step(
        "8-3. 전체 매핑 적용 (통합)",
        sql_apply_mapping_combined,
        maude_table="maude_normalized",
        mapping_table="unified_mapping",
        match_type_not_in_mapping=TEST_CONFIG["match_types"]["not_in_mapping"],
    )

    # ========== 9단계: 후처리 (Tier 3) ==========
    logger.info("\n" + "="*70)
    logger.info("9단계: 후처리 (Tier 3)")
    logger.info("="*70)

    test_step(
        "9-1. Low Compliance 제조사 식별",
        sql_identify_low_compliance_manufacturers,
        source_table="maude_normalized",
        low_compliance_threshold=TEST_CONFIG["low_compliance_threshold"],
    )

    test_step(
        "9-2. Tier 3 폴백 적용",
        sql_tier3_fallback,
        input_table="maude_mapped",
        low_compliance_table="low_compliance_mfrs",
        confidence_map=TEST_CONFIG["confidence_map"],
        match_type_no_match=TEST_CONFIG["match_types"]["no_match"],
        match_type_not_in_mapping=TEST_CONFIG["match_types"]["not_in_mapping"],
        match_type_udi_no_match=TEST_CONFIG["match_types"]["udi_no_match"],
    )

    # ========== 10단계: 최종 정리 ==========
    logger.info("\n" + "="*70)
    logger.info("10단계: 최종 정리")
    logger.info("="*70)

    test_step(
        "10-1. 중복 제거",
        sql_deduplicate_final,
        input_table="maude_tier3",
        unique_key="mdr_report_key",
    )

    test_step(
        "10-2. 최종 통계",
        sql_final_statistics,
        result_table="final_result",
    )

    # ========== 전체 파이프라인 ==========
    logger.info("\n" + "="*70)
    logger.info("전체 파이프라인 (단일 쿼리)")
    logger.info("="*70)

    test_step(
        "전체 파이프라인",
        sql_full_matching_pipeline,
        maude_preprocessed="maude_preprocessed",
        udi_primary_lookup="udi_primary_lookup",
        udi_full_lookup="udi_full_lookup",
        score_weights=TEST_CONFIG["score_weights"],
        require_unique_primary=TEST_CONFIG["require_unique_primary"],
        min_match_score=TEST_CONFIG["min_match_score"],
        match_type_direct=TEST_CONFIG["match_types"]["direct"],
        match_type_secondary=TEST_CONFIG["match_types"]["secondary"],
        match_type_udi_no_match=TEST_CONFIG["match_types"]["udi_no_match"],
        match_type_meta=TEST_CONFIG["match_types"]["meta"],
        match_type_no_match=TEST_CONFIG["match_types"]["no_match"],
    )

    logger.info("\n" + "="*70)
    logger.info("모든 테스트 완료!")
    logger.info("="*70)

"""
UDI 매칭 SQL 생성 모듈 (Snowflake)

설계 원칙:
- 각 함수는 SQL 문자열을 반환
- 설정은 개별 인자로 외부 주입
- Snowflake 연결은 이 모듈에서 하지 않음

매칭 흐름:
  1. 대표 날짜 컬럼 생성 (TARGET_DATE, SOURCE_DATE)
  2. UDI_DI 유무로 분기
  3. 1단계: UDI_DI → PRIMARY_UDI_DI (고유 시 UDI SUCCESS)
  4. 2단계: MANUFACTURER_NAME + device_cols 기반 메타 매칭 (고유 시 META SUCCESS)
  5. 정산: 신뢰도 부여 + HIGH일 때 컬럼 대체

MATCH_STATUS / CONFIDENCE:
  - UDI SUCCESS  → HIGH      (UDI_DI 매칭, PRIMARY_UDI_DI 고유)
  - META SUCCESS → MEDIUM    (메타데이터 매칭 성공)
  - UDI FAILED   → LOW       (UDI_DI 있으나 매칭 실패)
  - NO UDI       → VERY LOW  (UDI_DI 없고 매칭 실패)
"""
import structlog

from maude_early_alert.utils.sql_builder import build_cte_sql

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ==================== 표현식 헬퍼 ====================

def _coalesce_expr(cols: list[str], alias: str) -> str:
    return f"COALESCE({', '.join(cols)}) AS {alias}"


def _device_match_count(
    device_cols: list[str],
    left_prefix: str,
    right_prefix: str,
) -> str:
    if not device_cols:
        return "0"
    parts = [
        f"IFF({left_prefix}{c} IS NOT NULL AND {right_prefix}{c} IS NOT NULL "
        f"AND {left_prefix}{c} = {right_prefix}{c}, 1, 0)"
        for c in device_cols
    ]
    return " + ".join(parts)


def _build_confidence_case(
    confidence: dict,
    confidence_col: str,
    coalesce_expr: str,
) -> str:
    when_lines = "\n                    ".join(
        f"WHEN '{s}' THEN '{c}'"
        for s, c in confidence.items()
    )
    return (
        f"CASE {coalesce_expr}\n"
        f"                    {when_lines}\n"
        f"                END AS {confidence_col}"
    )


# ==================== 단계별 CTE 빌더 ====================

def _date_ctes(
    target: str, source: str,
    target_date: str, source_date: str,
) -> list:
    """1. 대표 날짜 컬럼 생성"""
    return [
        {'name': 'target_base', 'query':
            f"SELECT *, {target_date} FROM {target}"},
        {'name': 'source_base', 'query':
            f"SELECT *, {source_date} FROM {source}"},
    ]


def _split_ctes(udi_di: str) -> list:
    """2. UDI_DI 유무 분기"""
    return [
        {'name': 'has_udi', 'query':
            f"SELECT * FROM target_base WHERE {udi_di} IS NOT NULL"},
        {'name': 'no_udi', 'query':
            f"SELECT * FROM target_base WHERE {udi_di} IS NULL"},
    ]


def _primary_match_ctes(
    sel: str,
    udi_di: str, primary_udi_di: str,
    source_date_alias: str,
    status_udi_success: str, match_status_col: str,
) -> list:
    """3. 1단계: UDI_DI 직접 매칭"""
    return [
        {'name': 'step1_joined', 'query': f"""\
            SELECT
                h.*,
                {sel},
                u.{source_date_alias},
                COUNT(DISTINCT u.{primary_udi_di}) OVER (PARTITION BY h.{udi_di}) AS N_PRIMARY
            FROM has_udi h
            INNER JOIN source_base u ON h.{udi_di} = u.{udi_di}"""},

        {'name': 'udi_success', 'query': f"""\
            SELECT *, '{status_udi_success}' AS {match_status_col}
            FROM step1_joined
            WHERE N_PRIMARY = 1
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY {udi_di} ORDER BY {source_date_alias} DESC NULLS LAST
            ) = 1"""},
    ]


def _meta_match_ctes(
    sel: str,
    dm_cte: str, dm_join: str,
    part_no_udi: str,
    manufacturer: str, source_manufacturer: str,
    target_date_alias: str, source_date_alias: str,
    udi_di: str, primary_udi_di: str,
    min_device_match: int,
    status_meta_success: str, match_status_col: str,
) -> list:
    """4. 2단계: 메타데이터 매칭"""
    # 4a. UDI_DI 있으나 PRIMARY_UDI_DI 비고유 → 후보: 매칭된 source 행
    from_udi = [
        {'name': 'step2_udi_filtered', 'query': f"""\
            SELECT *
            FROM step1_joined
            WHERE N_PRIMARY > 1
              AND {manufacturer} = {source_manufacturer}
              AND {target_date_alias} > {source_date_alias}
              AND ({dm_cte}) >= {min_device_match}"""},

        {'name': 'step2_udi_counted', 'query': f"""\
            SELECT *,
                COUNT(DISTINCT {primary_udi_di}) OVER (PARTITION BY {udi_di}) AS N_PRIMARY_META
            FROM step2_udi_filtered"""},

        {'name': 'meta_from_udi', 'query': f"""\
            SELECT *, '{status_meta_success}' AS {match_status_col}
            FROM step2_udi_counted
            WHERE N_PRIMARY_META = 1
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY {udi_di} ORDER BY {source_date_alias} DESC NULLS LAST
            ) = 1"""},
    ]

    # 4b. UDI_DI 없음 → 후보: source 전체
    from_no_udi = [
        {'name': 'step2_no_udi_joined', 'query': f"""\
            SELECT
                n.*,
                {sel},
                u.{source_date_alias}
            FROM no_udi n
            INNER JOIN source_base u ON n.{manufacturer} = u.{manufacturer}
            WHERE n.{target_date_alias} > u.{source_date_alias}
              AND ({dm_join}) >= {min_device_match}"""},

        {'name': 'step2_no_udi_counted', 'query': f"""\
            SELECT *,
                COUNT(DISTINCT {primary_udi_di}) OVER (
                    PARTITION BY {part_no_udi}
                ) AS N_PRIMARY_META
            FROM step2_no_udi_joined"""},

        {'name': 'meta_from_no_udi', 'query': f"""\
            SELECT *, '{status_meta_success}' AS {match_status_col}
            FROM step2_no_udi_counted
            WHERE N_PRIMARY_META = 1
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY {part_no_udi}
                ORDER BY {source_date_alias} DESC NULLS LAST
            ) = 1"""},
    ]

    return from_udi + from_no_udi


def _mapping_ctes(map_cols: str, noudi_map_cols: str) -> list:
    """5. 매핑 통합"""
    return [
        {'name': 'udi_mapping', 'query': f"""\
            SELECT
                {map_cols}
            FROM udi_success
            UNION ALL
            SELECT
                {map_cols}
            FROM meta_from_udi"""},

        {'name': 'no_udi_mapping', 'query': f"""\
            SELECT
                {noudi_map_cols}
            FROM meta_from_no_udi"""},
    ]


def _settlement_ctes(
    excl: str,
    repl_udi: str, repl_mfr: str, repl_dev: str,
    noudi_orig: str, noudi_join: str,
    udi_di: str, match_status_col: str,
    status_udi_failed: str, status_no_udi: str,
    has_udi_conf: str, no_udi_conf: str,
) -> list:
    """6. 정산"""
    return [
        {'name': 'result_has_udi', 'query': f"""\
            SELECT
                e.* EXCLUDE ({excl}),
                {repl_udi},
                {repl_mfr},
                {repl_dev},
                COALESCE(m.{match_status_col}, '{status_udi_failed}') AS {match_status_col},
                {has_udi_conf}
            FROM has_udi e
            LEFT JOIN udi_mapping m ON e.{udi_di} = m.{udi_di}"""},

        {'name': 'result_no_udi', 'query': f"""\
            SELECT
                e.* EXCLUDE ({excl}),
                {noudi_orig},
                COALESCE(m_no.{match_status_col}, '{status_no_udi}') AS {match_status_col},
                {no_udi_conf}
            FROM no_udi e
            LEFT JOIN no_udi_mapping m_no ON {noudi_join}"""},
    ]


# ==================== 전체 파이프라인 ====================

def build_matching_sql(
    target: str,
    source: str,
    device_cols: list[str],
    target_date_cols: list[str],
    source_date_cols: list[str],
    udi_di: str,
    primary_udi_di: str,
    manufacturer: str,
    match_status_col: str,
    confidence_col: str,
    udi_col_prefix: str,
    status: dict,
    confidence: dict,
    min_device_match: int,
) -> str:
    """
    전체 UDI 매칭 파이프라인 SQL 생성 (단일 CTE 체인)

    Args:
        target: 매칭 대상 테이블 (event)
        source: 참조 데이터 테이블 (udi)
        device_cols: 기기 메타데이터 컬럼 리스트
        target_date_cols: target 대표 날짜 컬럼들 (COALESCE 순서)
        source_date_cols: source 대표 날짜 컬럼들 (COALESCE 순서)
        udi_di: UDI Device Identifier 컬럼명
        primary_udi_di: 정규화된 Primary UDI_DI 컬럼명
        manufacturer: 제조사명 컬럼명
        match_status_col: 매칭 상태 출력 컬럼명
        confidence_col: 신뢰도 출력 컬럼명
        udi_col_prefix: source 컬럼 alias prefix
        status: 매칭 상태 레이블 dict
        confidence: 신뢰도 매핑 dict (status → confidence)
        min_device_match: 기기 컬럼 최소 일치 개수
    """
    # --- 파생 값 ---
    target_date_alias  = "TARGET_DATE"
    source_date_alias  = "SOURCE_DATE"
    source_manufacturer = f"{udi_col_prefix}{manufacturer}"

    # --- 공통 표현식 ---
    target_date = _coalesce_expr(target_date_cols, target_date_alias)
    source_date = _coalesce_expr(source_date_cols, source_date_alias)

    sel = ",\n                ".join(
        [f"u.{primary_udi_di}", f"u.{manufacturer} AS {source_manufacturer}"]
        + [f"u.{c} AS {udi_col_prefix}{c}" for c in device_cols]
    )
    dm_cte  = _device_match_count(device_cols, "", udi_col_prefix)
    dm_join = _device_match_count(device_cols, "n.", "u.")

    part_no_udi = ", ".join([manufacturer] + list(device_cols))
    excl        = ", ".join([udi_di, manufacturer] + list(device_cols))

    map_cols = ",\n                ".join(
        [udi_di, primary_udi_di, source_manufacturer]
        + [f"{udi_col_prefix}{c}" for c in device_cols]
        + [match_status_col]
    )
    noudi_map_cols = ",\n                ".join(
        [manufacturer] + list(device_cols)
        + [primary_udi_di, source_manufacturer]
        + [f"{udi_col_prefix}{c}" for c in device_cols]
        + [match_status_col]
    )

    repl_udi = (
        f"CASE WHEN m.{match_status_col} = '{status['udi_success']}' "
        f"THEN m.{primary_udi_di} ELSE e.{udi_di} END AS {udi_di}"
    )
    repl_mfr = (
        f"CASE WHEN m.{match_status_col} = '{status['udi_success']}' "
        f"THEN m.{source_manufacturer} ELSE e.{manufacturer} END AS {manufacturer}"
    )
    repl_dev = ",\n                ".join(
        f"CASE WHEN m.{match_status_col} = '{status['udi_success']}' "
        f"THEN m.{udi_col_prefix}{c} ELSE e.{c} END AS {c}"
        for c in device_cols
    )
    noudi_orig = ",\n                ".join(
        [f"e.{udi_di}", f"e.{manufacturer}"] + [f"e.{c}" for c in device_cols]
    )
    noudi_join = " AND ".join(
        [f"e.{manufacturer} = m_no.{manufacturer}"]
        + [f"e.{c} IS NOT DISTINCT FROM m_no.{c}" for c in device_cols]
    )
    has_udi_conf = _build_confidence_case(
        confidence, confidence_col,
        f"COALESCE(m.{match_status_col}, '{status['udi_failed']}')",
    )
    no_udi_conf = _build_confidence_case(
        confidence, confidence_col,
        f"COALESCE(m_no.{match_status_col}, '{status['no_udi']}')",
    )

    # --- CTE 조립 ---
    ctes = (
        _date_ctes(target, source, target_date, source_date)
        + _split_ctes(udi_di)
        + _primary_match_ctes(
            sel, udi_di, primary_udi_di,
            source_date_alias,
            status['udi_success'], match_status_col,
        )
        + _meta_match_ctes(
            sel, dm_cte, dm_join, part_no_udi,
            manufacturer, source_manufacturer,
            target_date_alias, source_date_alias,
            udi_di, primary_udi_di,
            min_device_match,
            status['meta_success'], match_status_col,
        )
        + _mapping_ctes(map_cols, noudi_map_cols)
        + _settlement_ctes(
            excl, repl_udi, repl_mfr, repl_dev,
            noudi_orig, noudi_join,
            udi_di, match_status_col,
            status['udi_failed'], status['no_udi'],
            has_udi_conf, no_udi_conf,
        )
    )

    return build_cte_sql(
        ctes=ctes,
        final_query="""\
            SELECT * FROM result_has_udi
            UNION ALL
            SELECT * FROM result_no_udi""",
    )


# ==================== 테스트 ====================

if __name__ == "__main__":
    print("=== build_matching_sql ===")
    sql = build_matching_sql(
        target='target_table',
        source='source_table',
        device_cols=['brand_name', 'generic_name'],
        target_date_cols=['date_received', 'date_of_event'],
        source_date_cols=['publish_date', 'version_date'],
        udi_di='udi_di',
        primary_udi_di='primary_udi_di',
        manufacturer='manufacturer_name',
        match_status_col='match_status',
        confidence_col='confidence',
        udi_col_prefix='udi_',
        status={
            'udi_success': 'UDI SUCCESS',
            'meta_success': 'META SUCCESS',
            'udi_failed': 'UDI FAILED',
            'no_udi': 'NO UDI',
        },
        confidence={
            'UDI SUCCESS': 'HIGH',
            'META SUCCESS': 'MEDIUM',
            'UDI FAILED': 'LOW',
            'NO UDI': 'VERY LOW',
        },
        min_device_match=1,
    )
    print(sql)

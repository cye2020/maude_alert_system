"""
UDI 매칭 SQL 생성 모듈 (Snowflake)

설계 원칙:
- 각 함수는 SQL 문자열을 반환
- 설정은 개별 인자로 외부 주입
- Snowflake 연결은 이 모듈에서 하지 않음

매칭 흐름:
  1. 대표 날짜 컬럼 생성 (EVENT_DATE, UDI_DATE)
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


# ==================== 단계별 CTE 빌더 ====================

def _date_ctes(event_table, udi_table, ev_date, udi_date):
    """1. 대표 날짜 컬럼 생성"""
    return [
        {'name': 'event_base', 'query':
            f"SELECT *, {ev_date} FROM {event_table}"},
        {'name': 'udi_base', 'query':
            f"SELECT *, {udi_date} FROM {udi_table}"},
    ]


def _split_ctes():
    """2. UDI_DI 유무 분기"""
    return [
        {'name': 'has_udi', 'query':
            "SELECT * FROM event_base WHERE UDI_DI IS NOT NULL"},
        {'name': 'no_udi', 'query':
            "SELECT * FROM event_base WHERE UDI_DI IS NULL"},
    ]


def _primary_match_ctes(udi_sel):
    """3. 1단계: UDI_DI 직접 매칭"""
    return [
        {'name': 'step1_joined', 'query': f"""\
            SELECT
                h.*,
                {udi_sel},
                u.UDI_DATE,
                COUNT(DISTINCT u.PRIMARY_UDI_DI) OVER (PARTITION BY h.UDI_DI) AS N_PRIMARY
            FROM has_udi h
            INNER JOIN udi_base u ON h.UDI_DI = u.UDI_DI"""},

        {'name': 'udi_success', 'query': """\
            SELECT *, 'UDI SUCCESS' AS MATCH_STATUS
            FROM step1_joined
            WHERE N_PRIMARY = 1
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY UDI_DI ORDER BY UDI_DATE DESC NULLS LAST
            ) = 1"""},
    ]


def _meta_match_ctes(udi_sel, dm_cte, dm_join, part_no_udi):
    """4. 2단계: 메타데이터 매칭"""
    # 4a. UDI_DI 있으나 PRIMARY_UDI_DI 비고유 → 후보: 매칭된 UDI 행
    from_udi = [
        {'name': 'step2_udi_filtered', 'query': f"""\
            SELECT *
            FROM step1_joined
            WHERE N_PRIMARY > 1
              AND MANUFACTURER_NAME = UDI_MANUFACTURER_NAME
              AND EVENT_DATE > UDI_DATE
              AND ({dm_cte}) >= 1"""},

        {'name': 'step2_udi_counted', 'query': """\
            SELECT *,
                COUNT(DISTINCT PRIMARY_UDI_DI) OVER (PARTITION BY UDI_DI) AS N_PRIMARY_META
            FROM step2_udi_filtered"""},

        {'name': 'meta_from_udi', 'query': """\
            SELECT *, 'META SUCCESS' AS MATCH_STATUS
            FROM step2_udi_counted
            WHERE N_PRIMARY_META = 1
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY UDI_DI ORDER BY UDI_DATE DESC NULLS LAST
            ) = 1"""},
    ]

    # 4b. UDI_DI 없음 → 후보: UDI 전체
    from_no_udi = [
        {'name': 'step2_no_udi_joined', 'query': f"""\
            SELECT
                n.*,
                {udi_sel},
                u.UDI_DATE
            FROM no_udi n
            INNER JOIN udi_base u ON n.MANUFACTURER_NAME = u.MANUFACTURER_NAME
            WHERE n.EVENT_DATE > u.UDI_DATE
              AND ({dm_join}) >= 1"""},

        {'name': 'step2_no_udi_counted', 'query': f"""\
            SELECT *,
                COUNT(DISTINCT PRIMARY_UDI_DI) OVER (
                    PARTITION BY {part_no_udi}
                ) AS N_PRIMARY_META
            FROM step2_no_udi_joined"""},

        {'name': 'meta_from_no_udi', 'query': f"""\
            SELECT *, 'META SUCCESS' AS MATCH_STATUS
            FROM step2_no_udi_counted
            WHERE N_PRIMARY_META = 1
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY {part_no_udi}
                ORDER BY UDI_DATE DESC NULLS LAST
            ) = 1"""},
    ]

    return from_udi + from_no_udi


def _mapping_ctes(map_cols, noudi_map_cols):
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


def _settlement_ctes(excl, repl_udi, repl_mfr, repl_dev, noudi_orig, noudi_join):
    """6. 정산"""
    return [
        {'name': 'result_has_udi', 'query': f"""\
            SELECT
                e.* EXCLUDE ({excl}),
                {repl_udi},
                {repl_mfr},
                {repl_dev},
                COALESCE(m.MATCH_STATUS, 'UDI FAILED') AS MATCH_STATUS,
                CASE COALESCE(m.MATCH_STATUS, 'UDI FAILED')
                    WHEN 'UDI SUCCESS' THEN 'HIGH'
                    WHEN 'META SUCCESS' THEN 'MEDIUM'
                    ELSE 'LOW'
                END AS CONFIDENCE
            FROM has_udi e
            LEFT JOIN udi_mapping m ON e.UDI_DI = m.UDI_DI"""},

        {'name': 'result_no_udi', 'query': f"""\
            SELECT
                e.* EXCLUDE ({excl}),
                {noudi_orig},
                COALESCE(m_no.MATCH_STATUS, 'NO UDI') AS MATCH_STATUS,
                CASE COALESCE(m_no.MATCH_STATUS, 'NO UDI')
                    WHEN 'META SUCCESS' THEN 'MEDIUM'
                    ELSE 'VERY LOW'
                END AS CONFIDENCE
            FROM no_udi e
            LEFT JOIN no_udi_mapping m_no ON {noudi_join}"""},
    ]


# ==================== 전체 파이프라인 ====================

def sql_full_matching_pipeline(
    event_table_name: str,
    udi_table_name: str,
    device_cols: list[str],
    event_date_cols: list[str],
    udi_date_cols: list[str],
) -> str:
    """
    전체 UDI 매칭 파이프라인 (단일 CTE 체인)

    Args:
        event_table_name: EVENT 테이블 이름
        udi_table_name: UDI 테이블 이름
        device_cols: 기기 정보 컬럼 리스트 (e.g. ['BRAND_NAME', 'MODEL_NUMBER', 'CATALOG_NUMBER'])
        event_date_cols: EVENT 날짜 컬럼들
        udi_date_cols: UDI 날짜 컬럼들
    """
    # --- 공통 표현식 ---
    ev_date = _coalesce_expr(event_date_cols, "EVENT_DATE")
    udi_date = _coalesce_expr(udi_date_cols, "UDI_DATE")

    udi_sel = ",\n                ".join(
        ["u.PRIMARY_UDI_DI", "u.MANUFACTURER_NAME AS UDI_MANUFACTURER_NAME"]
        + [f"u.{c} AS UDI_{c}" for c in device_cols]
    )
    dm_cte = _device_match_count(device_cols, "", "UDI_")
    dm_join = _device_match_count(device_cols, "n.", "u.")
    part_no_udi = ", ".join(["MANUFACTURER_NAME"] + list(device_cols))
    excl = ", ".join(["UDI_DI", "MANUFACTURER_NAME"] + list(device_cols))

    map_cols = ",\n                ".join(
        ["UDI_DI", "PRIMARY_UDI_DI", "UDI_MANUFACTURER_NAME"]
        + [f"UDI_{c}" for c in device_cols]
        + ["MATCH_STATUS"]
    )
    noudi_map_cols = ",\n                ".join(
        ["MANUFACTURER_NAME"] + list(device_cols)
        + ["PRIMARY_UDI_DI", "UDI_MANUFACTURER_NAME"]
        + [f"UDI_{c}" for c in device_cols]
        + ["MATCH_STATUS"]
    )

    repl_udi = (
        "CASE WHEN m.MATCH_STATUS = 'UDI SUCCESS' "
        "THEN m.PRIMARY_UDI_DI ELSE e.UDI_DI END AS UDI_DI"
    )
    repl_mfr = (
        "CASE WHEN m.MATCH_STATUS = 'UDI SUCCESS' "
        "THEN m.UDI_MANUFACTURER_NAME ELSE e.MANUFACTURER_NAME END AS MANUFACTURER_NAME"
    )
    repl_dev = ",\n                ".join(
        f"CASE WHEN m.MATCH_STATUS = 'UDI SUCCESS' "
        f"THEN m.UDI_{c} ELSE e.{c} END AS {c}"
        for c in device_cols
    )
    noudi_join = " AND ".join(
        ["e.MANUFACTURER_NAME = m_no.MANUFACTURER_NAME"]
        + [f"e.{c} IS NOT DISTINCT FROM m_no.{c}" for c in device_cols]
    )
    noudi_orig = ",\n                ".join(
        ["e.UDI_DI", "e.MANUFACTURER_NAME"] + [f"e.{c}" for c in device_cols]
    )

    # --- CTE 조립 ---
    ctes = (
        _date_ctes(event_table_name, udi_table_name, ev_date, udi_date)
        + _split_ctes()
        + _primary_match_ctes(udi_sel)
        + _meta_match_ctes(udi_sel, dm_cte, dm_join, part_no_udi)
        + _mapping_ctes(map_cols, noudi_map_cols)
        + _settlement_ctes(excl, repl_udi, repl_mfr, repl_dev, noudi_orig, noudi_join)
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
    from maude_early_alert.logging_config import configure_logging

    configure_logging('DEBUG', 'temp.log')

    sql = sql_full_matching_pipeline(
        event_table_name="EVENT_STAGE_09",
        udi_table_name="UDI_STAGE_06",
        device_cols=["BRAND_NAME", "MODEL_NUMBER", "CATALOG_NUMBER"],
        event_date_cols=["DATE_OCCURRED", "DATE_RECEIVED"],
        udi_date_cols=["PUBLISH_DATE", "PUBLIC_VERSION_DATE"],
    )

    logger.debug("전체 파이프라인 SQL")
    print(sql)
    logger.debug("테스트 완료!")

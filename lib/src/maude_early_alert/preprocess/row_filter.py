"""필터링 SQL 생성

WHERE / QUALIFY 절을 조립하여 Snowflake SQL을 생성한다.
- WHERE 조건: 값 필터링, 컬럼 비교, NOT EXISTS 등
- QUALIFY 조건: 중복 제거 (ROW_NUMBER)
"""

from typing import Dict, List

from maude_early_alert.utils.helpers import validate_identifier, ensure_list
from maude_early_alert.utils.sql_builder import build_cte_sql


def build_filter_sql(
    source: str, alias: str = "",
    where: List[str] = None, qualify: List[str] = None
) -> str:
    """WHERE / QUALIFY 절 → Snowflake SQL

    Args:
        source: 소스 테이블명
        alias: 테이블 별칭
        where: WHERE 조건 리스트 (AND로 결합)
        qualify: QUALIFY 조건 리스트 (AND로 결합)

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)

    sql = f"SELECT *\nFROM {source} {alias}"

    if where:
        where = ensure_list(where)
        sql += "\nWHERE " + "\n  AND ".join(c.strip() for c in where)

    if qualify:
        qualify = ensure_list(qualify)
        sql += "\nQUALIFY " + "\n  AND ".join(c.strip() for c in qualify)

    return sql


def build_filter_pipeline(cfg: Dict, source: str, alias: str = "s") -> str:
    """필터링 스텝들 → CTE 체인 SQL

    CTE 흐름:
        source → low_quality → dedup_step1 → dup_to_drop (QUALIFY)
                                    ↓               ↓
                               dedup_clean (NOT EXISTS)
                                    ↓
                                 logical

    Args:
        cfg: filtering.yaml 로드 결과
        source: 최초 소스 테이블/CTE 이름
        alias: 테이블 별칭
    """
    steps = [
        ('low_quality',    cfg['LOW_QUALITY'],         source),
        ('dedup_step1',    cfg['REPORT_DEDUP_STEP1'],  'low_quality'),
        ('dup_to_drop',    cfg['REPORT_DEDUP_STEP2'],  'dedup_step1'),
        ('dedup_clean',    cfg['REPORT_DEDUP_STEP3'],  'low_quality'),
        ('logical',        cfg['LOGICAL'],             'dedup_clean'),
    ]

    ctes = []
    for cte_name, step_cfg, from_source in steps:
        query = build_filter_sql(
            from_source, alias=alias,
            where=step_cfg.get('where'),
            qualify=step_cfg.get('qualify'),
        )
        ctes.append({'name': cte_name, 'query': query})

    return build_cte_sql(ctes=ctes, final_query="SELECT * FROM logical")


if __name__ == '__main__':
    from maude_early_alert.utils.config_loader import load_config

    cfg = load_config('preprocess/filtering')
    sql = build_filter_pipeline(cfg, source='EVENT_STAGE_10')
    print(sql)
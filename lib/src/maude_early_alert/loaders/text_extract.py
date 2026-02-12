# ======================
# 표준 라이브러리
# ======================
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Union

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.utils.helpers import validate_identifier


def build_mdr_text_extract_sql(
    table_name: str = "EVENT_STAGE_12",
    columns: Optional[List[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """EVENT_STAGE_12(또는 지정 테이블)에서 MDR_TEXT 추출용 SELECT SQL 생성.

    파이프라인에서 cursor.execute(sql) 후 fetch로 MDR_TEXT 리스트를 얻을 때 사용합니다.
    테스트 시에는 __main__ 에서 이 SQL을 실행해 보며 검증할 수 있습니다.

    Args:
        table_name: 대상 테이블 (FQDN 가능, 예: MAUDE.SILVER.EVENT_STAGE_12)
        columns: SELECT 컬럼 목록 (None이면 MDR_TEXT 만)
        where: WHERE 절 (선택, 예: "MDR_TEXT IS NOT NULL")
        limit: LIMIT 값 (None이면 미포함)

    Returns:
        실행 가능한 SELECT SQL 문자열
    """
    validate_identifier(table_name)
    select_cols = columns if columns is not None else ["MDR_TEXT"]
    for c in select_cols:
        validate_identifier(c)

    select_clause = ", ".join(select_cols)
    sql = f"SELECT {select_clause}\nFROM {table_name}"
    if where:
        sql += f"\nWHERE {dedent(where).strip()}"
    if limit is not None:
        sql += f"\nLIMIT {limit}"
    return sql


def records_to_rows(
    records: List[Union[str, dict]],
) -> List[dict]:
    """리스트 입력을 추출기 내부에서 쓰는 행 리스트로 변환.

    - str 이면 mdr_text 로만 사용, product_problems 는 빈 문자열.
    - dict 이면 'mdr_text' 필수, 'product_problems' / 'product_problems_str' 선택.
    """
    rows = []
    for i, r in enumerate(records):
        if isinstance(r, str):
            rows.append({"mdr_text": r, "product_problems": ""})
        elif isinstance(r, dict):
            text = r.get("mdr_text") or r.get("MDR_TEXT")
            if text is None:
                raise ValueError(f"record[{i}] has no mdr_text")
            probs = r.get("product_problems") or r.get("product_problems_str") or ""
            rows.append({"mdr_text": str(text), "product_problems": str(probs)})
        else:
            raise TypeError(f"record[{i}] must be str or dict, got {type(r)}")
    return rows


# -----------------------------------------------------------------------------
# 리스트 기반 추출기 (파이프라인: unique_mdr_text → extractor.process_batch → df)
# 실제 vLLM 추출 로직은 src.preprocess.extractor.MAUDEExtractor 에 두고,
# 리스트 입력만 여기서 받아 DataFrame 으로 넘겨 호출합니다.
# -----------------------------------------------------------------------------


def get_extractor_class():
    """
    MAUDEExtractor를 lazy import (지연 로딩).

    이 함수가 호출되기 전까지 vLLM, torch 등 무거운 라이브러리가
    로딩되지 않습니다. 덕분에 SQL 빌드만 할 때는 GPU가 없는 환경에서도
    이 모듈을 안전하게 import할 수 있습니다.
    """
    try:
        from maude_early_alert.analyze.mdr_extractor import MAUDEExtractor
        return MAUDEExtractor
    except ImportError:
        raise ImportError(
            "MAUDEExtractor를 찾을 수 없습니다. "
            "maude_early_alert.analyze.mdr_extractor 경로를 확인하세요."
        )


class MDRExtractor:
    """MDR 텍스트 리스트를 받아 추출 결과 DataFrame 을 반환하는 파사드.

    파이프라인 예:
        sql = build_mdr_text_extract_sql(table_name="EVENT_STAGE_12", limit=100)
        result = cursor.execute(sql)
        rows = result.fetchall()
        unique_mdr_text = list(set(r[0] for r in rows))  # 또는 dict 리스트
        extractor = MDRExtractor()
        df = extractor.process_batch(unique_mdr_text)
    """

    def __init__(self, **kwargs):
        """MAUDEExtractor 와 동일한 인자 전달 (model_path, tensor_parallel_size 등)."""
        self._kwargs = kwargs
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is None:
            Klass = get_extractor_class()
            self._extractor = Klass(**self._kwargs)
        return self._extractor

    def process_batch(
        self,
        records: List[Union[str, dict]],
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 5000,
        checkpoint_prefix: str = "checkpoint",
    ):
        """리스트(또는 unique_mdr_text)를 받아 추출 후 DataFrame 반환.

        Args:
            records: MDR 텍스트 문자열 리스트, 또는 dict 리스트 (mdr_text, product_problems 등)
            checkpoint_dir: None 이면 체크포인트 없이 process_with_retry 만 수행
            checkpoint_interval: 체크포인트 간격
            checkpoint_prefix: 체크포인트 파일 접두사

        Returns:
            추출 결과 DataFrame (파이프라인에서 df 적재 SQL 등에 사용)
        """
        # str 또는 dict 리스트를 {'mdr_text': ..., 'product_problems': ...} 형태로 통일
        rows = records_to_rows(records)

        impl = self._get_extractor()
        if checkpoint_dir is None:
            return impl.process_with_retry(rows)
        return impl.process_batch(
            rows,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint_prefix=checkpoint_prefix,
        )


if __name__ == "__main__":
    # 테스트: MDR_TEXT 추출 SQL만 빌드 (실제 DB 없이 검증)
    sql = build_mdr_text_extract_sql(table_name="EVENT_STAGE_12", limit=100)
    print("=== build_mdr_text_extract_sql (limit=100) ===")
    print(sql)
    print()

    # 리스트 → 행 변환만 테스트
    rows = records_to_rows(["text1", "text2", {"mdr_text": "t3", "product_problems": "p3"}])
    print("=== records_to_rows sample ===")
    for r in rows:
        print(r)

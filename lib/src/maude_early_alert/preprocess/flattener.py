# ==================================================
# JSON 평탄화 - Snowflake SQL 구문 자동 생성기
# 목적:
# - MAUDE 원천 JSON(results 배열 구조)을 Snowflake에서
#   SELECT 가능한 평탄화 SQL로 자동 변환
# ==================================================

from typing import Dict, Any, List, Tuple
import json


# ==================================================
# 특수 필드 처리 전략 정의
# ==================================================

def first_only_field(col: str) -> Dict[str, Any]:
    """
    배열이지만 첫 번째 요소만 의미가 있는 필드
    예) patient[0]
    """
    return {'field': col, 'type': 'first_only'}


def list_field(col: str) -> Dict[str, Any]:
    """
    배열 전체를 유지하는 필드
    예) mdr_text[*]
    """
    return {'field': col, 'type': 'list'}


def row_split_field(col: str) -> Dict[str, Any]:
    """
    배열을 행 단위로 분리해야 하는 필드
    예) device → LATERAL FLATTEN
    """
    return {'field': col, 'type': 'row_split'}


# ==================================================
# JSON 필드 재귀 추출 로직
# ==================================================

def _extract_fields(data: Dict, prefix: str = '', sep: str = '_') -> List[Tuple[str, str]]:
    """
    JSON dict를 재귀적으로 순회하여
    - Snowflake 컬럼명
    - JSON 접근 경로
    를 쌍으로 반환

    Returns:
        [
          ("device_brand_name", "brand_name"),
          ("device_model", "model")
        ]
    """

    fields = []

    for key, value in data.items():
        # 컬럼명은 상위 key를 prefix로 누적
        col_name = f"{prefix}{sep}{key}" if prefix else key
        json_path = key

        if isinstance(value, dict):
            # 중첩 JSON → 재귀 호출
            nested = _extract_fields(value, col_name, sep)

            # JSON path는 Snowflake ':' 문법으로 누적
            fields.extend([
                (cn, f"{json_path}:{jp}")
                for cn, jp in nested
            ])
        else:
            # leaf node → 바로 컬럼화
            fields.append((col_name, json_path))

    return fields


# ==================================================
# Snowflake SQL 생성 메인 함수
# ==================================================

def _collect_all_fields(records: List[Dict[str, Any]], special_configs: List[Dict[str, Any]], sep: str = '_') -> Dict[str, Any]:
    """
    여러 레코드를 스캔하여 전체 필드 구조 수집
    """
    special_map = {cfg['field']: cfg['type'] for cfg in special_configs}
    
    # 모든 필드 수집용 set
    all_base_fields = set()
    all_first_only_fields = set()
    all_list_fields = set()
    all_device_fields = set()
    row_split_field = None
    
    for record in records:
        for key, value in record.items():
            spec_type = special_map.get(key)
            
            if spec_type == 'first_only':
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    for cn, jp in _extract_fields(value[0], f"{key}{sep}0", sep):
                        all_first_only_fields.add((cn, f"{key}[0]:{jp}"))
            
            elif spec_type == 'list':
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    for field_name in value[0].keys():
                        all_list_fields.add((f"{key}{sep}{field_name}", f"{key}[*]:{field_name}"))
            
            elif spec_type == 'row_split':
                row_split_field = key
                if isinstance(value, list) and len(value) > 0:
                    for device_item in value:
                        if isinstance(device_item, dict):
                            for cn, jp in _extract_fields(device_item, row_split_field, sep):
                                all_device_fields.add((cn, jp))
            
            else:
                if isinstance(value, dict):
                    for cn, jp in _extract_fields(value, key, sep):
                        all_base_fields.add((cn, jp))
                else:
                    all_base_fields.add((key, key))
    
    return {
        'base_fields': sorted(all_base_fields),
        'first_only_fields': sorted(all_first_only_fields),
        'list_fields': sorted(all_list_fields),
        'device_fields': sorted(all_device_fields),
        'row_split_field': row_split_field
    }


def generate_flatten_sql(
    raw_table_name: str,
    records: List[Dict[str, Any]],
    special_configs: List[Dict[str, Any]],
    sep: str = '_'
) -> str:
    """
    여러 JSON 레코드를 스캔하여
    Snowflake용 평탄화 SELECT SQL 생성
    """

    # 모든 레코드에서 필드 수집
    collected = _collect_all_fields(records, special_configs, sep)
    
    base_fields = collected['base_fields']
    first_only_fields = collected['first_only_fields']
    list_fields = collected['list_fields']
    device_fields = collected['device_fields']
    row_split_field = collected['row_split_field']
    
    if row_split_field and not device_fields:
        raise ValueError(f"{row_split_field} 필드 데이터가 없습니다")

    # --------------------------------------------------
    # 3. SELECT 절 생성
    # --------------------------------------------------
    select_lines = []

    # 일반 필드
    for col_name, json_path in base_fields:
        select_lines.append(
            f"value:{json_path}::STRING AS {col_name}"
        )

    # patient
    for col_name, json_path in first_only_fields:
        select_lines.append(
            f"value:{json_path}::STRING AS {col_name}"
        )

    # device (row split)
    for col_name, json_path in device_fields:
        # device_brand → brand
        field_path = json_path.replace(f"{row_split_field}{sep}", "")
        select_lines.append(
            f"device_item:{field_path}::STRING AS {col_name}"
        )

    # mdr_text (array 유지)
    for col_name, json_path in list_fields:
        select_lines.append(
            f"value:{json_path} AS {col_name}"
        )

    # --------------------------------------------------
    # 4. FROM 절
    # --------------------------------------------------
    if row_split_field:
        from_clause = f"""FROM {raw_table_name},
LATERAL FLATTEN(input => raw_json:results) AS value,
LATERAL FLATTEN(input => value:{row_split_field}) AS device_item"""
    else:
        from_clause = f"""FROM {raw_table_name},
LATERAL FLATTEN(input => raw_json:results) AS value"""

    # --------------------------------------------------
    # 5. 최종 SQL 조립
    # --------------------------------------------------
    nl = '\n'
    return f"""SELECT
    {f',{nl}    '.join(select_lines)}
{from_clause}"""


# ==================================================
# JSON 파일 기반 SQL 생성
# ==================================================

def generate_flatten_sql_from_file(
    file_path: str,
    raw_table_name: str,
    special_configs: List[Dict[str, Any]],
    sep: str = '_',
    max_records: int = 1000
) -> str:
    """
    실제 JSON 파일을 읽어
    모든 results 레코드를 스캔하여 전체 스키마 수집
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results or len(results) == 0:
        raise ValueError("results 배열이 비어있습니다")
    
    # 성능을 위해 최대 max_records개만 스캔
    records_to_scan = results[:max_records] if len(results) > max_records else results
    
    return generate_flatten_sql(raw_table_name, records_to_scan, special_configs, sep)


# ==================================================
# CLI 실행 인터페이스
# ==================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MAUDE JSON 평탄화 SQL 생성')
    parser.add_argument('--file', '-f', type=str,
                       default=r'D:\data\raw\maude_json\2024q1_device-event-0001-of-0007.json',
                       help='샘플 JSON 파일 경로')
    parser.add_argument('--table', '-t', type=str, default='RAW_TEMP_TABLE',
                       help='Snowflake 테이블명')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='SQL 파일 저장 경로 (지정하지 않으면 콘솔 출력)')
    
    args = parser.parse_args()
    
    special_configs = [
        first_only_field('patient'),
        list_field('mdr_text'),
        row_split_field('device')
    ]
    
    print("=" * 60)
    print("MAUDE JSON 평탄화 SQL 생성")
    print("=" * 60)
    print(f"샘플 파일: {args.file}")
    print(f"테이블명: {args.table}")
    print("=" * 60)
    
    try:
        sql = generate_flatten_sql_from_file(
            file_path=args.file,
            raw_table_name=args.table,
            special_configs=special_configs
        )
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(sql)
            print(f"\nSQL 저장: {args.output}")
        else:
            print("\n생성된 SQL:")
            print("=" * 60)
            print(sql)
        
    except FileNotFoundError:
        print(f"\n파일 없음: {args.file}")
    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()

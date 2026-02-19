import structlog
import pendulum
from typing import Any, Dict, List, Union
from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.loaders.snowflake_load import (
    SnowflakeLoader,
    get_staging_table_name,
    get_raw_table_name,
)
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.utils.helpers import ensure_list, validate_identifier

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class BronzePipeline(SnowflakeBase):
    def __init__(self, logical_date: pendulum.DateTime):
        self.cfg = get_config().bronze
        self.logical_date = logical_date
        self.loader = SnowflakeLoader()

        if not self.cfg.get_snowflake_enabled():
            logger.warning('Snowflake 로드 비활성화 상태, 건너뜀')
            return

        database = self.cfg.get_snowflake_load_database()
        schema = self.cfg.get_snowflake_load_schema()
        super().__init__(database, schema)

    def load_folder(
        self,
        cursor: SnowflakeCursor,
        table_name: str,
        s3_stage_path: str,
        primary_key: Union[str, List[str]],
        metadata: Dict[str, Any] = None,
        business_primary_key: Union[str, List[str]] = None,
    ) -> Dict[str, Any]:
        """단일 테이블 S3→Snowflake 적재 (전체 플로우 실행)

        Args:
            cursor: Snowflake cursor
            table_name: 목적 테이블명
            s3_stage_path: S3 스테이지 전체 경로 (stage/folder)
            primary_key: MERGE 기본 키
            metadata: 메타데이터 (ingest_time 등)
            business_primary_key: 비즈니스 기본 키
        """
        validate_identifier(table_name)
        for pk in ensure_list(primary_key):
            validate_identifier(pk)
        if business_primary_key:
            for bpk in ensure_list(business_primary_key):
                validate_identifier(bpk)

        logger.info('S3 데이터 적재 시작',
                     table_name=table_name, s3_stage=s3_stage_path)

        # 1. 테이블 스키마 조회
        table_schema = self.get_table_schema(cursor, table_name)
        stg_table_name = get_staging_table_name(table_name)
        raw_table_name = get_raw_table_name(s3_stage_path)

        # 2. 스테이징 테이블 생성
        sql = self.loader.build_create_staging_sql(table_name, table_schema)
        logger.debug('스테이징 테이블 생성', stg_table_name=stg_table_name)
        cursor.execute(sql)

        # 3. Raw 임시 테이블 생성
        sql = self.loader.build_create_raw_temp_sql(s3_stage_path)
        logger.debug('Raw 임시 테이블 생성', raw_table_name=raw_table_name)
        cursor.execute(sql)

        # 4. S3 → Raw 임시 테이블 COPY INTO
        sql = self.loader.build_copy_into_sql(raw_table_name, s3_stage_path)
        logger.debug('COPY INTO 실행', s3_stage=s3_stage_path)
        cursor.execute(sql)
        copy_result = self._parse_copy_result(cursor.fetchall())

        # 5. FLATTEN INSERT (Raw → 스테이징)
        sql = self.loader.build_flatten_insert_sql(
            raw_table_name, stg_table_name,
            metadata=metadata,
            business_primary_key=business_primary_key,
        )
        logger.debug('FLATTEN INSERT 실행', stg_table_name=stg_table_name)
        cursor.execute(sql)
        rows_inserted = cursor.rowcount

        # 6. MERGE (스테이징 → 목적 테이블)
        column_names = [col_name for col_name, _ in table_schema]
        sql = self.loader.build_merge_sql(
            table_name, stg_table_name, primary_key, column_names,
        )
        logger.debug('MERGE 실행', table_name=table_name)
        with self.transaction(cursor):
            cursor.execute(sql)

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_count = cursor.fetchone()[0]

        logger.info('S3 데이터 적재 완료',
                     table_name=table_name,
                     rows_inserted=rows_inserted,
                     total_rows=total_count)

        return {
            'files_loaded': copy_result.get('files_loaded', 0),
            'rows_inserted': rows_inserted,
            'total_rows': total_count,
        }

    @with_context
    def load_all(
        self, cursor: SnowflakeCursor,
        batch_id: str,
    ) -> List[Dict[str, Any]]:
        """config의 모든 테이블 순회하며 S3→Snowflake 적재

        Args:
            cursor: Snowflake cursor
            batch_id: 배치 식별자
        """

        tables = self.cfg.get_snowflake_load_tables()
        stage = self.cfg.get_snowflake_load_stage()
        ym = self.logical_date.strftime('%Y%m')
        results = []

        metadata = {
            'source_system': 's3',
            'ingest_time': self.logical_date,
            'batch_id': batch_id,
        }

        for table in tables:
            s3_stage_path = f"{stage}/{ym}/device/{table.lower()}"
            primary_key = self.cfg.get_snowflake_load_primary_key(table)
            business_primary_key = self.cfg.get_snowflake_load_business_primary_key(table)

            result = self.load_folder(
                cursor,
                table_name=table,
                s3_stage_path=s3_stage_path,
                primary_key=primary_key,
                metadata=metadata,
                business_primary_key=business_primary_key,
            )
            results.append(result)

        return results

    @staticmethod
    def _parse_copy_result(results) -> Dict[str, Any]:
        """COPY INTO 실행 결과 파싱

        Args:
            results: cursor.fetchall() 결과
                     (file, status, rows_parsed, rows_loaded, error_limit, errors_seen, ...)
        """
        files_loaded = sum(1 for r in results if r[1] == 'LOADED')
        rows_loaded = sum(r[3] for r in results if r and len(r) > 3)
        errors_seen = sum(r[5] for r in results if r and len(r) > 5)

        if errors_seen > 0:
            logger.warning('COPY INTO 중 에러 발생', errors_seen=errors_seen)
            for r in results:
                if r[5] > 0:
                    logger.warning(
                        'COPY INTO 에러 상세',
                        file=r[0], status=r[1],
                        errors_seen=r[5], first_error=r[6],
                        first_error_line=r[7]
                    )

        return {
            'files_loaded': files_loaded,
            'rows_loaded': rows_loaded,
            'errors_seen': errors_seen,
        }



if __name__ == '__main__':
    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret

    secret = get_secret('snowflake/de')
    
    conn = snowflake.connector.connect(
        user=secret['user'],
        password=secret['password'],
        account=secret['account'],
        warehouse=secret['warehouse'],
    )

    logical_date = pendulum.now()
    pipeline = BronzePipeline(logical_date)
    batch_id = f"bronze_{logical_date.strftime('%Y%m%d_%H%M%S')}"

    try:
        cursor = conn.cursor()
        results = pipeline.load_all(cursor, batch_id=batch_id)

        for i, result in enumerate(results):
            print(f"\n[테이블 {i+1}] {result}")
    finally:
        cursor.close()
        conn.close()
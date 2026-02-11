"""
FDA MAUDE 의료기기 부작용 데이터를 Snowflake에서 가져와 
로컬 vLLM(qwen3-14b)로 분석하고 결과를 다시 Snowflake에 저장
"""

import asyncio
import json
import time
import os
import configparser
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.pandas_tools import write_pandas
from openai import AsyncOpenAI, APIError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog

from prompt import GeneralPrompt


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SnowflakeConfig:
    """Snowflake 연결 설정"""
    account: str
    user: str
    password: str
    database: str
    schema: str
    warehouse: str
    role: Optional[str] = None
    timeout: int = 300
    login_timeout: int = 30

    @classmethod
    def from_config(cls, config: configparser.ConfigParser):
        sf = config['snowflake']
        return cls(
            account=sf['account'],
            user=sf['user'],
            password=sf['password'],
            database=sf['database'],
            schema=sf['schema'],
            warehouse=sf['warehouse'],
            role=sf.get('role') or None,
            timeout=int(sf.get('timeout', 300)),
            login_timeout=int(sf.get('login_timeout', 30))
        )


@dataclass
class VLLMConfig:
    """vLLM 서버 설정"""
    model: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.3
    max_tokens: int = 1500
    timeout: int = 60

    @classmethod
    def from_config(cls, config: configparser.ConfigParser):
        vllm = config['vllm']
        return cls(
            model=vllm['model'],
            base_url=vllm['base_url'],
            api_key=vllm.get('api_key', 'EMPTY'),
            temperature=float(vllm.get('temperature', 0.3)),
            max_tokens=int(vllm.get('max_tokens', 1500)),
            timeout=int(vllm.get('timeout', 60))
        )


@dataclass
class ProcessingConfig:
    """처리 설정"""
    batch_size: int
    concurrent_requests: int
    checkpoint_interval: int
    source_table: str
    cache_table: str
    final_table: str
    output_dir: Path
    checkpoint_file: Path
    log_file: Path

    @classmethod
    def from_config(cls, config: configparser.ConfigParser):
        proc = config['processing']
        tables = config['tables']
        output = config['output']

        output_dir = Path(output.get('output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            batch_size=int(proc.get('batch_size', 5000)),
            concurrent_requests=int(proc.get('concurrent_requests', 100)),
            checkpoint_interval=int(proc.get('checkpoint_interval', 1000)),
            source_table=tables.get('event_stage_12', 'EVENT_STAGE_12'),
            cache_table=tables.get('cache_table', 'EVENT_STAGE_12_CACHE'),
            final_table=tables.get('final_table', 'EVENT_STAGE_12_ANALYSIS'),
            output_dir=output_dir,
            checkpoint_file=output_dir / 'checkpoint.json',
            log_file=output_dir / 'maude_processor.log'
        )


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """처리 메트릭 수집"""

    def __init__(self):
        self.metrics = {
            'total_processed': 0,
            'total_failed': 0,
            'processing_times': [],
            'error_types': {},
            'start_time': time.time()
        }

    def record_success(self, processing_time: float):
        self.metrics['total_processed'] += 1
        self.metrics['processing_times'].append(processing_time)

    def record_failure(self, error_type: str):
        self.metrics['total_failed'] += 1
        self.metrics['error_types'][error_type] = \
            self.metrics['error_types'].get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.metrics['start_time']
        times = self.metrics['processing_times']
        avg_time = sum(times) / len(times) if times else 0

        total = self.metrics['total_processed'] + self.metrics['total_failed']
        success_rate = (self.metrics['total_processed'] / total * 100) if total > 0 else 0
        throughput = self.metrics['total_processed'] / elapsed if elapsed > 0 else 0

        return {
            'total_processed': self.metrics['total_processed'],
            'total_failed': self.metrics['total_failed'],
            'success_rate': round(success_rate, 2),
            'avg_processing_time': round(avg_time, 3),
            'throughput': round(throughput, 2),
            'elapsed_seconds': round(elapsed, 1),
            'error_types': self.metrics['error_types']
        }


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """체크포인트 관리 (멱등성 + 재개 기능)"""

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.state = self._load()
        self.logger = structlog.get_logger(__name__)

    def _load(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    state = json.load(f)
                    if isinstance(state.get('processed_keys'), list):
                        state['processed_keys'] = set(state['processed_keys'])
                    if isinstance(state.get('processed_hashes'), list):
                        state['processed_keys'] = set(state['processed_hashes'])
                    return state
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            'processed_keys': set(),
            'last_batch_id': 0,
            'total_processed': 0
        }

    def save(self):
        """Atomic write"""
        state_copy = self.state.copy()
        state_copy['processed_keys'] = list(self.state['processed_keys'])
        state_copy['last_updated'] = datetime.utcnow().isoformat() + 'Z'

        temp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(state_copy, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(self.checkpoint_file)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

    def mark_processed(self, report_key: int):
        """MDR_REPORT_KEY 처리 완료 마킹"""
        self.state['processed_keys'].add(str(report_key))
        self.state['total_processed'] += 1

    def is_processed(self, report_key: int) -> bool:
        """MDR_REPORT_KEY 처리 여부 확인"""
        return str(report_key) in self.state['processed_keys']

    def update_batch_id(self, batch_id: int):
        self.state['last_batch_id'] = batch_id


# =============================================================================
# Snowflake Manager
# =============================================================================

class SnowflakeManager:
    """Snowflake 연결 및 쿼리 관리"""

    def __init__(self, config: SnowflakeConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.conn = None
        self._connect()

    def _connect(self):
        """Snowflake 연결"""
        try:
            conn_params = {
                'account': self.config.account,
                'user': self.config.user,
                'password': self.config.password,
                'database': self.config.database,
                'schema': self.config.schema,
                'warehouse': self.config.warehouse,
                'login_timeout': self.config.login_timeout,
                'network_timeout': self.config.timeout,
                'session_parameters': {
                    'QUERY_TAG': 'MAUDE_PROCESSOR_VLLM',
                    'TIMEZONE': 'UTC'
                }
            }

            if self.config.role:
                conn_params['role'] = self.config.role

            self.conn = snowflake.connector.connect(**conn_params)

            self.logger.info(
                "Snowflake connected",
                database=self.config.database,
                schema=self.config.schema
            )
        except Exception as e:
            self.logger.error("Snowflake connection failed", error=str(e))
            raise

    def fetch_unprocessed_batch(
        self,
        batch_size: int,
        checkpoint: CheckpointManager,
        source_table: str
    ) -> pd.DataFrame:
        """미처리 레코드 배치 가져오기 (중복 제거 없음)"""
        try:
            processed_keys = list(checkpoint.state['processed_keys'])[:10000]

            if processed_keys:
                processed_keys_int = [int(k) for k in processed_keys]
                placeholders = ', '.join(['%s'] * len(processed_keys_int))
                exclude_clause = f"AND MDR_REPORT_KEY NOT IN ({placeholders})"
            else:
                exclude_clause = ""
                processed_keys_int = []

            query = f"""
            SELECT 
                MDR_REPORT_KEY,
                MDR_TEXT,
                ARRAY_TO_STRING(PRODUCT_PROBLEMS, '; ') AS product_problems_str,
                BRAND_NAME,
                MANUFACTURER_NAME,
                PRODUCT_NAME,
                MODEL_NUMBER
            FROM {source_table}
            WHERE MDR_TEXT IS NOT NULL
                {exclude_clause}
            ORDER BY MDR_REPORT_KEY
            LIMIT {batch_size}
            """

            start_time = time.time()
            cursor = self.conn.cursor(DictCursor)
            cursor.execute(query, processed_keys_int)
            results = cursor.fetchall()

            data = []
            for row in results:
                data.append({
                    'mdr_report_key': int(row['MDR_REPORT_KEY']),
                    'mdr_text': row['MDR_TEXT'],
                    'product_problems_str': row.get('PRODUCT_PROBLEMS_STR', ''),
                    'brand_name': row.get('BRAND_NAME', ''),
                    'manufacturer_name': row.get('MANUFACTURER_NAME', ''),
                    'product_name': row.get('PRODUCT_NAME', ''),
                    'model_number': row.get('MODEL_NUMBER', '')
                })

            df = pd.DataFrame(data)
            cursor.close()

            self.logger.info(
                "Batch fetched",
                size=len(df),
                time_sec=round(time.time() - start_time, 2)
            )
            return df

        except Exception as e:
            self.logger.error("Fetch failed", error=str(e))
            raise

    def bulk_upload_results(self, df: pd.DataFrame, table_name: str):
        """COPY INTO로 대량 업로드"""
        if len(df) == 0:
            return

        try:
            start_time = time.time()
            success, _, num_rows, _ = write_pandas(
                conn=self.conn,
                df=df,
                table_name=table_name,
                auto_create_table=False,
                overwrite=False,
                quote_identifiers=False
            )

            if success:
                self.logger.info(
                    "Upload successful",
                    table=table_name,
                    records=num_rows,
                    time_sec=round(time.time() - start_time, 2)
                )
            else:
                self.logger.error("Upload failed", table=table_name)

        except Exception as e:
            self.logger.error("Upload error", error=str(e))
            raise

    def create_final_results(self, cache_table: str, final_table: str):
        """캐시 테이블을 최종 테이블로 복사 (JOIN 불필요)"""
        cursor = self.conn.cursor()
        try:
            self.logger.info("Creating final table (direct copy)")

            cursor.execute(f"""
                CREATE OR REPLACE TABLE {final_table} AS
                SELECT * FROM {cache_table}
            """)

            cursor.execute(f"SELECT COUNT(*) FROM {final_table}")
            count = cursor.fetchone()[0]
            self.logger.info("Final table created", records=count)

        finally:
            cursor.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Snowflake connection closed")


# =============================================================================
# vLLM Processor
# =============================================================================

class VLLMProcessor:
    """vLLM 비동기 처리기"""

    def __init__(
        self,
        config: VLLMConfig,
        metrics: MetricsCollector,
        concurrent_requests: int = 100
    ):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.metrics = metrics
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self.semaphore = asyncio.Semaphore(concurrent_requests)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _call_vllm(self, mdr_text: str, product_problems: str) -> Dict:
        """vLLM API 호출 (Exponential Backoff 재시도)"""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": GeneralPrompt.SYSTEM_INSTRUCTION},
                {"role": "user", "content": GeneralPrompt.format_user_prompt(
                    text=mdr_text,
                    product_problem=product_problems or "Not Available"
                )}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )
        return json.loads(response.choices[0].message.content)

    async def process_single(self, record: Dict) -> Dict:
        """단일 레코드 처리"""
        async with self.semaphore:
            start_time = time.time()
            mdr_report_key = record['mdr_report_key']

            try:
                result = await self._call_vllm(
                    record['mdr_text'],
                    record.get('product_problems_str', 'Not Available')
                )

                processing_time = (time.time() - start_time) * 1000
                self._validate_result(result)

                flat_result = {
                    'mdr_report_key': mdr_report_key,
                    'mdr_text': record['mdr_text'][:1000],
                    'product_problems_str': record.get('product_problems_str', ''),
                    'brand_name': record.get('brand_name', ''),
                    'manufacturer_name': record.get('manufacturer_name', ''),
                    'product_name': record.get('product_name', ''),
                    'model_number': record.get('model_number', ''),
                    'patient_harm': result['incident_details']['patient_harm'],
                    'patient_harm_original_text': result['incident_details']['patient_harm_original_text'][:200],
                    'problem_components': json.dumps(result['incident_details']['problem_components']),
                    'problem_components_original_text': result['incident_details']['problem_components_original_text'][:200],
                    'defect_confirmed': result['manufacturer_inspection']['defect_confirmed'],
                    'defect_confirmed_original_text': result['manufacturer_inspection']['defect_confirmed_original_text'][:200],
                    'defect_type': result['manufacturer_inspection']['defect_type'],
                    'defect_type_original_text': result['manufacturer_inspection']['defect_type_original_text'][:200],
                    'processing_time_ms': int(processing_time),
                    'processed_at': datetime.utcnow().isoformat() + 'Z',
                    'status': 'success'
                }

                self.metrics.record_success(processing_time / 1000)
                return flat_result

            except json.JSONDecodeError as e:
                self.logger.error("JSON parse error", key=mdr_report_key, error=str(e))
                self.metrics.record_failure('json_parse_error')
                return self._error_result(mdr_report_key, 'json_parse_error', str(e))

            except (APIError, APITimeoutError) as e:
                self.logger.error("API error", key=mdr_report_key, error=str(e))
                self.metrics.record_failure('api_error')
                return self._error_result(mdr_report_key, 'api_error', str(e))

            except Exception as e:
                self.logger.error("Unexpected error", key=mdr_report_key, error=str(e))
                self.metrics.record_failure('unknown_error')
                return self._error_result(mdr_report_key, 'unknown_error', str(e))

    def _validate_result(self, result: Dict):
        """LLM 응답 검증"""
        required = ['incident_details', 'manufacturer_inspection']
        for key in required:
            if key not in result:
                raise ValueError(f"Missing key: {key}")

        incident_keys = ['patient_harm', 'patient_harm_original_text',
                        'problem_components', 'problem_components_original_text']
        for key in incident_keys:
            if key not in result['incident_details']:
                raise ValueError(f"Missing incident detail: {key}")

        inspection_keys = ['defect_confirmed', 'defect_confirmed_original_text',
                          'defect_type', 'defect_type_original_text']
        for key in inspection_keys:
            if key not in result['manufacturer_inspection']:
                raise ValueError(f"Missing inspection detail: {key}")

    def _error_result(self, report_key: int, error_type: str, error_msg: str) -> Dict:
        return {
            'mdr_report_key': report_key,
            'status': 'error',
            'error_type': error_type,
            'error_message': error_msg
        }

    async def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """배치 비동기 처리"""
        records = df.to_dict('records')
        tasks = [self.process_single(record) for record in records]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return pd.DataFrame(results)


# =============================================================================
# Main Orchestrator
# =============================================================================

class MAUDEProcessor:
    """전체 처리 흐름 조율"""

    def __init__(self, sf_config: SnowflakeConfig, vllm_config: VLLMConfig, proc_config: ProcessingConfig):
        self.proc_config = proc_config
        proc_config.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = structlog.get_logger(__name__)
        self.metrics = MetricsCollector()
        self.checkpoint = CheckpointManager(proc_config.checkpoint_file)
        self.sf_manager = SnowflakeManager(sf_config)
        self.vllm_processor = VLLMProcessor(
            vllm_config,
            self.metrics,
            proc_config.concurrent_requests
        )

        self.logger.info("MAUDE Processor initialized")

    async def process_all(self):
        """전체 처리 루프"""
        self.logger.info("Starting processing")
        batch_id = self.checkpoint.state['last_batch_id']

        while True:
            try:
                df = self.sf_manager.fetch_unprocessed_batch(
                    self.proc_config.batch_size,
                    self.checkpoint,
                    self.proc_config.source_table
                )

                if len(df) == 0:
                    self.logger.info("No more unprocessed records")
                    break

                self.logger.info(f"Processing batch {batch_id}", size=len(df))

                results_df = await self.vllm_processor.process_batch(df)
                success_df = results_df[results_df['status'] == 'success'].copy()
                failed_df = results_df[results_df['status'] == 'error']

                if len(success_df) > 0:
                    self.sf_manager.bulk_upload_results(success_df, self.proc_config.cache_table)
                    for report_key in success_df['mdr_report_key']:
                        self.checkpoint.mark_processed(int(report_key))

                if len(failed_df) > 0:
                    self.logger.warning(f"Batch had {len(failed_df)} failures")

                self.checkpoint.update_batch_id(batch_id)
                if batch_id % (self.proc_config.checkpoint_interval // self.proc_config.batch_size) == 0:
                    self.checkpoint.save()

                metrics = self.metrics.get_summary()
                self.logger.info("Batch completed", **metrics)
                batch_id += 1

            except KeyboardInterrupt:
                self.logger.info("Interrupted by user")
                self.checkpoint.save()
                break

            except Exception as e:
                self.logger.error("Batch failed", error=str(e), batch_id=batch_id)
                batch_id += 1
                continue

        self.logger.info("Creating final results table")
        self.sf_manager.create_final_results(
            self.proc_config.cache_table,
            self.proc_config.final_table
        )

        final_metrics = self.metrics.get_summary()
        self.logger.info("Processing completed", **final_metrics)
        return final_metrics

    def close(self):
        self.sf_manager.close()
        self.checkpoint.save()


# =============================================================================
# structlog 설정
# =============================================================================

def setup_logging(log_file: Path):
    """structlog 설정"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """메인 진입점"""
    print("=" * 80)
    print("MAUDE Processor - vLLM Edition")
    print("=" * 80)
    print()

    # 프로젝트 루트 찾기
    def find_project_root():
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / '.env').exists():
                return current
            if (current / '.git').exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        raise FileNotFoundError(
            f"Could not find .env file. Searched from {Path(__file__).resolve()}"
        )

    try:
        project_root = find_project_root()
        config_file = project_root / '.env'
        print(f"Project root: {project_root}")
        print(f"Config file: {config_file}")
        print()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not config_file.exists():
        print(f"ERROR: Configuration file not found: {config_file}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    sf_config = SnowflakeConfig.from_config(config)
    vllm_config = VLLMConfig.from_config(config)
    proc_config = ProcessingConfig.from_config(config)

    # structlog 설정
    setup_logging(proc_config.log_file)

    processor = MAUDEProcessor(sf_config, vllm_config, proc_config)

    try:
        metrics = await processor.process_all()

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(f"Total Processed: {metrics['total_processed']:,}")
        print(f"Total Failed: {metrics['total_failed']:,}")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
        print(f"Throughput: {metrics['throughput']:.2f} texts/sec")
        print(f"Total Time: {metrics['elapsed_seconds']/3600:.2f} hours")
        print("=" * 80)

    finally:
        processor.close()


if __name__ == "__main__":
    asyncio.run(main())
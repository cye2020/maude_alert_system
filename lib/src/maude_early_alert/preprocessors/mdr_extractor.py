# ============================================================================
# MAUDEExtractor
# vLLM 기반 MDR 텍스트 구조화 추출기
# ============================================================================
# Windows vllm-windows wheel 사용 시 PyTorch 소스 빌드 없이 쓰려면 필요 (libuv 미지원 빌드)
import os
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ======================
# 표준 라이브러리
# ======================
import gc
import json
import shutil
import sqlite3
import time
from functools import partial
import structlog

import pendulum
from pathlib import Path
from typing import Any, Dict, List, Union

# ======================
# 서드파티 라이브러리
# ======================
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.preprocessors.prompt import Prompt, GeneralPrompt

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class MAUDEExtractor:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_retries: int,
        enable_prefix_caching: bool,
        sampling_config: dict,
        prompt: Prompt = None,
    ):
        """vLLM 최적화 배치 추출기

        Args:
            model_path: 모델 경로
            tensor_parallel_size: 사용할 GPU 수
            gpu_memory_utilization: GPU 메모리 사용률 (0.0-1.0)
            max_model_len: 최대 시퀀스 길이 (토큰 수)
            max_num_batched_tokens: Chunked prefill 토큰 수 (throughput 핵심)
            max_num_seqs: 동시 처리 시퀀스 수
            max_retries: 실패 시 재시도 횟수
            enable_prefix_caching: 반복 프롬프트 캐싱 활성화
            sampling_config: 샘플링 파라미터 dict (temperature, max_tokens, top_p)
            prompt: 사용할 Prompt 인스턴스 (None이면 GeneralPrompt)
        """
        self.max_retries = max_retries
        self.model_path = model_path
        self.prompt = prompt if prompt is not None else GeneralPrompt()
        self.extraction_model = self.prompt.get_extraction_model()

        logger.debug(
            "Loading vLLM model",
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            max_retries=max_retries,
        )

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
            enforce_eager=False,
            enable_prefix_caching=enable_prefix_caching,
            disable_log_stats=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        logger.debug("Model loaded successfully")

        self.json_schema = self.extraction_model.model_json_schema()
        self.sampling_params = SamplingParams(
            temperature=sampling_config['temperature'],
            max_tokens=sampling_config['max_tokens'],
            top_p=sampling_config['top_p'],
            structured_outputs=StructuredOutputsParams(json=self.json_schema),
            truncate_prompt_tokens=max_model_len - sampling_config['max_tokens'],
        )

    # --------------------------------------------------------------------------
    # 내부 메서드 (파이프라인 단계별)
    # --------------------------------------------------------------------------

    def _create_prompts(self, mdr_records: List[Dict[str, Any]]) -> List[str]:
        """Chat template을 적용한 프롬프트 문자열 리스트 생성."""
        prompts = []
        for record in mdr_records:
            user_content = self.prompt.format_user_prompt(
                text=record.get('mdr_text', ''),
                product_problem=record.get('product_problems', ''),
            )
            messages = [
                {"role": "system", "content": self.prompt.SYSTEM_INSTRUCTION},
                {"role": "user",   "content": user_content},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(formatted)
        return prompts

    def _parse_and_validate(self, response_text: str) -> dict:
        """LLM 응답 JSON 파싱 + Pydantic 모델로 검증."""
        data = json.loads(response_text)
        validated = self.extraction_model(**data)
        return validated.model_dump(mode='json')

    def _generate_and_parse(
        self,
        mdr_records: List[Dict[str, Any]],
    ) -> tuple:
        """
        vLLM 배치 추론 및 결과 파싱.

        Returns:
            (results, stats)
            - results: 각 레코드의 추출 결과 dict 리스트
            - stats: 처리 시간·토큰 통계 dict
        """
        batch_start = time.time()

        prompts = self._create_prompts(mdr_records)
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=partial(tqdm, mininterval=10.0),
        )

        batch_time = time.time() - batch_start

        results = []
        total_input_tokens = 0
        total_output_tokens = 0

        for record, output in zip(mdr_records, outputs):
            try:
                response_text = output.outputs[0].text
                validated_data = self._parse_and_validate(response_text)

                input_tokens  = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)
                total_input_tokens  += input_tokens
                total_output_tokens += output_tokens

                results.append({
                    **validated_data,
                    '_mdr_text':      record.get('mdr_text', ''),
                    '_success':       True,
                    '_input_tokens':  input_tokens,
                    '_output_tokens': output_tokens,
                    '_total_tokens':  input_tokens + output_tokens,
                })
            except Exception as e:
                results.append({
                    '_mdr_text':      record.get('mdr_text', ''),
                    '_success':       False,
                    '_error':         str(e)[:200],
                    '_raw_response':  output.outputs[0].text[:200],
                })

        stats = {
            'batch_time':          batch_time,
            'num_samples':         len(mdr_records),
            'total_input_tokens':  total_input_tokens,
            'total_output_tokens': total_output_tokens,
        }
        return results, stats

    def _print_batch_stats(self, stats: dict):
        """배치 처리 통계 출력."""
        throughput   = stats['num_samples'] / stats['batch_time'] if stats['batch_time'] > 0 else 0
        total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
        tokens_per_sec = total_tokens / stats['batch_time'] if stats['batch_time'] > 0 else 0
        logger.debug(
            "Batch stats",
            num_samples=stats['num_samples'],
            batch_time=f"{stats['batch_time']:.2f}s",
            throughput=f"{throughput:.1f} samples/s",
            tokens_per_sec=f"{tokens_per_sec:.0f} tokens/s",
        )

    def _open_cache_db(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str,
    ) -> sqlite3.Connection:
        """SQLite 캐시 DB를 열고 테이블을 초기화."""
        db_path = checkpoint_dir / f'{checkpoint_name}.db'
        conn = sqlite3.connect(str(db_path))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS extraction_cache (
                mdr_text TEXT PRIMARY KEY,
                result   TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS input_records (
                idx    INTEGER PRIMARY KEY,
                record TEXT NOT NULL
            )
        ''')
        conn.commit()
        return conn

    def _load_cache_from_db(self, conn: sqlite3.Connection) -> Dict[str, dict]:
        """DB에서 {mdr_text: result} 캐시 딕셔너리 로드."""
        cache = {
            mdr_text: json.loads(result_json)
            for mdr_text, result_json in conn.execute(
                'SELECT mdr_text, result FROM extraction_cache'
            )
        }
        logger.debug("Cache loaded from DB", cached_count=len(cache))
        return cache

    def _save_to_cache_db(self, conn: sqlite3.Connection, results: List[dict]) -> None:
        """추론 결과를 DB에 upsert."""
        conn.executemany(
            'INSERT OR REPLACE INTO extraction_cache (mdr_text, result) VALUES (?, ?)',
            [
                (res['_mdr_text'], json.dumps(res, ensure_ascii=False))
                for res in results
                if res.get('_mdr_text')
            ],
        )
        conn.commit()

    # --------------------------------------------------------------------------
    # 공개 메서드
    # --------------------------------------------------------------------------

    def process_with_retry(
        self,
        mdr_records: List[Dict[str, Any]],
    ) -> List[dict]:
        """
        재시도 로직 포함 배치 처리.

        실패한 레코드만 골라서 최대 max_retries 회 재시도합니다.
        체크포인트 없이 소량 처리하거나 단위 테스트 시 사용합니다.

        Args:
            mdr_records: {'mdr_text': ..., 'product_problems': ...} 형태의 dict 리스트

        Returns:
            입력과 동일한 순서의 추출 결과 dict 리스트
        """
        all_results: dict = {}
        pending = mdr_records.copy()
        attempt = 1

        while pending and attempt <= self.max_retries:
            if attempt > 1:
                logger.debug("Retry attempt", attempt=attempt, pending_count=len(pending))

            results, stats = self._generate_and_parse(pending)
            self._print_batch_stats(stats)

            next_pending = []
            for record, result in zip(pending, results):
                result['_attempts'] = attempt
                mdr_key = record.get('mdr_text', '')
                all_results[mdr_key] = result
                if not result['_success']:
                    next_pending.append(record)

            pending = next_pending
            attempt += 1

        # 재시도 횟수 초과한 항목 처리
        for record in pending:
            mdr_key = record.get('mdr_text', '')
            if mdr_key not in all_results:
                all_results[mdr_key] = {
                    '_mdr_text':  mdr_key,
                    '_success':   False,
                    '_error':     'Max retries exceeded',
                    '_attempts':  self.max_retries,
                }

        # 원래 입력 순서 유지
        return [all_results[r.get('mdr_text', '')] for r in mdr_records]

    def cleanup_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """체크포인트 디렉토리 삭제."""
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.debug('체크포인트 삭제 완료', checkpoint_dir=str(checkpoint_dir))

    def process_batch(
        self,
        mdr_records: List[Dict[str, Any]],
        checkpoint_dir: Union[str, Path],
        checkpoint_interval: int = 5000,
        checkpoint_name: str = 'checkpoint',
    ) -> List[Dict[str, Any]]:
        """
        전체 레코드 배치 처리 (체크포인트 포함).

        대량 데이터를 checkpoint_interval 단위로 나눠 처리하고,
        추론 결과를 SQLite DB({name}.db)에 mdr_text 기준으로 upsert합니다.
        중단 후 재개 시 DB를 캐시로 재활용합니다.
        처리 완료 후 체크포인트를 삭제하려면 cleanup_checkpoint()를 명시적으로 호출하세요.

        Args:
            mdr_records: {'mdr_text': ..., 'product_problems': ...} dict 리스트
            checkpoint_dir: SQLite DB 저장 디렉토리
            checkpoint_interval: 한 청크당 레코드 수
            checkpoint_name: DB 파일명 ({name}.db)

        Returns:
            전체 추출 결과 dict 리스트 (입력 순서 유지)
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        conn = self._open_cache_db(checkpoint_dir, checkpoint_name)
        cache = self._load_cache_from_db(conn)

        # 재개 시 records를 복원할 수 있도록 저장 (이미 있으면 유지)
        if not conn.execute('SELECT COUNT(*) FROM input_records').fetchone()[0]:
            conn.executemany(
                'INSERT INTO input_records (idx, record) VALUES (?, ?)',
                [(i, json.dumps(r, ensure_ascii=False)) for i, r in enumerate(mdr_records)],
            )
            conn.commit()

        num_chunks  = (len(mdr_records) - 1) // checkpoint_interval + 1
        all_results: List[list] = []

        logger.debug(
            "vLLM Batch Processing Started",
            total_records=len(mdr_records),
            num_chunks=num_chunks,
            checkpoint_interval=checkpoint_interval,
            max_retries=self.max_retries,
            cache_hits_available=len(cache),
        )

        overall_start = time.time()
        infer_chunks  = 0  # 실제 LLM 추론이 발생한 청크 수 (ETA 계산용)

        try:
            for chunk_idx in trange(num_chunks, desc="Processing chunks"):
                start_idx = chunk_idx * checkpoint_interval
                end_idx   = min((chunk_idx + 1) * checkpoint_interval, len(mdr_records))
                chunk     = mdr_records[start_idx:end_idx]

                # 캐시 미스 레코드만 LLM 추론
                to_infer = [r for r in chunk if r['mdr_text'] not in cache]

                chunk_start = time.time()
                if to_infer:
                    new_results = self.process_with_retry(to_infer)
                    self._save_to_cache_db(conn, new_results)
                    for r, res in zip(to_infer, new_results):
                        cache[r['mdr_text']] = res
                    infer_chunks += 1

                chunk_result = [cache[r['mdr_text']] for r in chunk]
                all_results.append(chunk_result)

                elapsed   = time.time() - chunk_start
                success   = sum(1 for r in chunk_result if r.get('_success', False))
                cache_hit = len(chunk) - len(to_infer)

                elapsed_session = time.time() - overall_start
                if infer_chunks > 0:
                    avg_infer     = elapsed_session / infer_chunks
                    remaining     = num_chunks - chunk_idx - 1
                    eta_seconds   = avg_infer * remaining
                    eta_kst       = pendulum.now("Asia/Seoul").add(seconds=int(eta_seconds))
                    remaining_str = f"{eta_seconds / 3600:.1f}h"
                    eta_str       = eta_kst.format("MM-DD HH:mm")
                else:
                    remaining_str = "N/A"
                    eta_str       = "N/A"

                logger.debug(
                    "Chunk completed",
                    chunk=chunk_idx + 1,
                    total_chunks=num_chunks,
                    success=success,
                    cache_hit=cache_hit,
                    inferred=len(to_infer),
                    success_rate=f"{100 * success / len(chunk):.1f}%",
                    elapsed=f"{elapsed:.1f}s",
                    elapsed_session=f"{elapsed_session / 3600:.2f}h",
                    remaining_time=remaining_str,
                    eta_kst=eta_str,
                )

            # 청크 리스트를 단일 flat 리스트로 합치기
            flat_results = [r for chunk in all_results for r in chunk]

            # 최종 통계
            total_time    = time.time() - overall_start
            success_count = sum(1 for r in flat_results if r.get('_success', False))
            successful    = [r for r in flat_results if r.get('_success', False)]
            total_tokens  = sum(r.get('_total_tokens', 0) for r in successful)
            avg_input     = sum(r.get('_input_tokens', 0) for r in successful) / max(success_count, 1)
            avg_output    = sum(r.get('_output_tokens', 0) for r in successful) / max(success_count, 1)

            logger.debug(
                "Batch processing complete",
                total_processed=len(flat_results),
                success_count=success_count,
                success_rate=f"{100 * success_count / len(flat_results):.1f}%",
                failed_count=len(flat_results) - success_count,
                total_time=f"{total_time / 60:.1f} min ({total_time / 3600:.2f} hours)",
                throughput=f"{len(flat_results) / total_time:.2f} samples/s",
                total_tokens=total_tokens,
                avg_input_tokens=f"{avg_input:.1f}",
                avg_output_tokens=f"{avg_output:.1f}",
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            conn.close()
            return flat_results

        except KeyboardInterrupt:
            logger.debug("Processing interrupted, cache preserved in DB", checkpoint_dir=str(checkpoint_dir))
            conn.close()
            if all_results:
                return [r for chunk in all_results for r in chunk]
            raise


def load_checkpoint_state(
    checkpoint_dir: Union[str, Path],
    checkpoint_name: str,
) -> tuple[List[dict], List[dict]]:
    """checkpoint DB에서 input_records와 extraction_cache를 읽어 복원.

    run_llm_extraction 완료 후 run_failure_model_retry가 중단된 경우,
    이전 단계 재실행 없이 상태를 복원할 때 사용합니다.

    Args:
        checkpoint_dir: SQLite DB가 위치한 디렉토리
        checkpoint_name: DB 파일명 (확장자 제외, 예: 'checkpoint_event')

    Returns:
        (records, results) 튜플. 원본 입력 순서가 유지됩니다.

    Raises:
        FileNotFoundError: DB 파일이 없을 경우
    """
    db_path = Path(checkpoint_dir) / f'{checkpoint_name}.db'
    if not db_path.exists():
        raise FileNotFoundError(
            f'체크포인트 DB 없음: {db_path}. '
            'run_llm_extraction이 완료되지 않았거나 이미 정리된 상태입니다.'
        )

    conn = sqlite3.connect(str(db_path))

    records = [
        json.loads(record_json)
        for _, record_json in conn.execute(
            'SELECT idx, record FROM input_records ORDER BY idx'
        )
    ]
    cache = {
        mdr_text: json.loads(result_json)
        for mdr_text, result_json in conn.execute(
            'SELECT mdr_text, result FROM extraction_cache'
        )
    }
    conn.close()

    results = [
        cache.get(r['mdr_text'], {
            '_mdr_text': r['mdr_text'],
            '_success': False,
            '_error': 'Not found in checkpoint DB',
        })
        for r in records
    ]

    logger.info(
        '체크포인트 상태 복원',
        db=str(db_path),
        records=len(records),
        cached_results=len(cache),
    )
    return records, results


# ============================================================================
# 테스트 실행 (__main__)
# ============================================================================

if __name__ == "__main__":
    import os
    import warnings
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    SAMPLE_RECORDS = [
        {
            'mdr_text': "Patient reported that the device failed to deliver the correct insulin dose. "
                        "The pump displayed an error code E3. No patient harm was observed.",
            'product_problems': "Incorrect dose delivered",
        },
        {
            'mdr_text': "The catheter tip broke off during insertion and was retrieved endoscopically. "
                        "Patient experienced minor bleeding. Defect confirmed by manufacturer.",
            'product_problems': "Device breakage",
        },
        {
            'mdr_text': "Software malfunction caused the ventilator to stop cycling. "
                        "Nurse intervened manually. Patient required additional oxygen therapy.",
            'product_problems': "Software failure",
        },
    ]

    extractor = MAUDEExtractor(
        model_path="Qwen/Qwen3-14B-AWQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=16384,
        max_num_batched_tokens=32768,
        max_num_seqs=128,
        max_retries=2,
        enable_prefix_caching=True,
        sampling_config={"temperature": 0.1, "max_tokens": 512, "top_p": 0.95},
    )

    results = extractor.process_batch(
        SAMPLE_RECORDS,
        checkpoint_dir="./checkpoints",
        checkpoint_interval=1000,
        checkpoint_name="test",
    )

    logger.debug("Results")
    for record, result in zip(SAMPLE_RECORDS, results):
        logger.debug("Result", input=record['mdr_text'][:80], output=result)

    del extractor

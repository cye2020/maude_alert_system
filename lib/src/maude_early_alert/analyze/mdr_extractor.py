# ============================================================================
# MAUDEExtractor
# vLLM 기반 MDR 텍스트 구조화 추출기
# ============================================================================
# Windows vllm-windows wheel 사용 시 PyTorch 소스 빌드 없이 쓰려면 필요 (libuv 미지원 빌드)
import os
os.environ.setdefault("USE_LIBUV", "0")

# ======================
# 표준 라이브러리
# ======================
import gc
import json
import shutil
import time
from functools import partial
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
from maude_early_alert.analyze.prompt import Prompt, GeneralPrompt


class MAUDEExtractor:
    def __init__(
        self,
        model_path: str = 'Qwen/Qwen3-8B',
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 256,
        max_retries: int = 2,
        enable_prefix_caching: bool = True,
        prompt: Prompt = None,
    ):
        """
        vLLM 최적화 배치 추출기 (Qwen3-8B)

        Args:
            model_path: 모델 경로
            tensor_parallel_size: 사용할 GPU 수
            gpu_memory_utilization: GPU 메모리 사용률 (0.0-1.0)
            max_model_len: 최대 시퀀스 길이 (토큰 수)
            max_num_batched_tokens: Chunked prefill 토큰 수 (throughput 핵심)
            max_num_seqs: 동시 처리 시퀀스 수
            max_retries: 실패 시 재시도 횟수
            enable_prefix_caching: 반복 프롬프트 캐싱 활성화
            prompt: 사용할 Prompt 인스턴스 (기본: 기본 Prompt 클래스)
        """
        self.max_retries = max_retries
        self.model_path = model_path
        # None이면 기본 Prompt 사용 (mutable default argument 방지)
        self.prompt = prompt if prompt is not None else GeneralPrompt()
        self.extraction_model = self.prompt.get_extraction_model()

        print("=" * 70)
        print(f"Loading vLLM model: {model_path}")
        print("=" * 70)
        print(f"  - Tensor Parallel Size:  {tensor_parallel_size}")
        print(f"  - GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"  - Max Model Length:       {max_model_len}")
        print(f"  - Max Batched Tokens:     {max_num_batched_tokens}")
        print(f"  - Max Num Seqs:           {max_num_seqs}")
        print(f"  - Prefix Caching:         {enable_prefix_caching}")
        print(f"  - Max Retries:            {max_retries}")

        # vLLM 모델 초기화
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,  # Chunked prefill: GPU 활용률 극대화
            max_num_seqs=max_num_seqs,                      # 동시 처리 시퀀스 수
            trust_remote_code=True,
            enforce_eager=False,                            # CUDA graph 사용 (추론 속도 향상)
            enable_prefix_caching=enable_prefix_caching,    # 동일 시스템 프롬프트 캐싱
            disable_log_stats=False,
            swap_space=4,                                   # Preemption 발생 시 CPU로 swap
        )

        # Tokenizer 로드 (chat template 적용에 사용)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        print(f"\n✓ Model loaded successfully!\n")

        # JSON 스키마 기반 structured output 설정
        # → LLM이 항상 지정한 JSON 형식으로만 응답하도록 강제
        self.json_schema = self.extraction_model.model_json_schema()
        self.sampling_params = SamplingParams(
            temperature=0.1,       # 낮을수록 결정적 출력 (분류 태스크에 적합)
            max_tokens=512,
            top_p=0.95,
            structured_outputs=StructuredOutputsParams(json=self.json_schema),
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
                enable_thinking=True,   # Qwen3 thinking mode 활성화
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
        """배치 처리 통계 한 줄 출력."""
        throughput  = stats['num_samples'] / stats['batch_time'] if stats['batch_time'] > 0 else 0
        total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
        tokens_per_sec = total_tokens / stats['batch_time'] if stats['batch_time'] > 0 else 0
        print(
            f"  Batch: {stats['num_samples']:4d} samples | "
            f"Time: {stats['batch_time']:6.2f}s | "
            f"Throughput: {throughput:5.1f} samples/s | "
            f"{tokens_per_sec:6.0f} tokens/s"
        )

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
                print(f"\n  Retry attempt {attempt}: {len(pending)} failed samples")

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

    def process_batch(
        self,
        mdr_records: List[Dict[str, Any]],
        checkpoint_dir: Union[str, Path],
        checkpoint_interval: int = 5000,
        checkpoint_prefix: str = 'checkpoint',
    ) -> List[Dict[str, Any]]:
        """
        전체 레코드 배치 처리 (체크포인트 포함).

        대량 데이터를 checkpoint_interval 단위로 나눠 처리하고,
        각 청크 결과를 JSON 파일로 저장합니다.
        완료 후 체크포인트 디렉토리는 자동 삭제됩니다.

        Args:
            mdr_records: {'mdr_text': ..., 'product_problems': ...} dict 리스트
            checkpoint_dir: 체크포인트 JSON 저장 디렉토리
            checkpoint_interval: 한 청크당 레코드 수
            checkpoint_prefix: 체크포인트 파일명 접두사

        Returns:
            전체 추출 결과 dict 리스트 (입력 순서 유지)
        """
        print("=" * 70)
        print("vLLM Batch Processing Started")
        print("=" * 70)
        print(f"  Total records:       {len(mdr_records):,}")
        print(f"  Checkpoint interval: {checkpoint_interval:,} rows")
        print(f"  Max retries/chunk:   {self.max_retries}")
        print("=" * 70)

        overall_start = time.time()
        all_results: List[list] = []

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            num_chunks = (len(mdr_records) - 1) // checkpoint_interval + 1

            for chunk_idx in trange(num_chunks, desc="Processing chunks"):
                start_idx = chunk_idx * checkpoint_interval
                end_idx   = min((chunk_idx + 1) * checkpoint_interval, len(mdr_records))
                chunk     = mdr_records[start_idx:end_idx]

                print(f"\nChunk {chunk_idx + 1}/{num_chunks} | Rows {start_idx:,}-{end_idx - 1:,}")

                chunk_start  = time.time()
                chunk_result = self.process_with_retry(chunk)
                all_results.append(chunk_result)

                elapsed  = time.time() - chunk_start
                success  = sum(1 for r in chunk_result if r.get('_success', False))
                throughput = len(chunk) / elapsed if elapsed > 0 else 0

                print(
                    f"  Chunk completed: {success}/{len(chunk)} success "
                    f"({100 * success / len(chunk):.1f}%) | "
                    f"{elapsed:.1f}s | {throughput:.2f} samples/s"
                )

                # 청크 결과를 JSON으로 저장 (장애 발생 시 복구 가능)
                checkpoint_file = f'{checkpoint_prefix}_chunk{chunk_idx + 1:03d}.json'
                checkpoint_path = checkpoint_dir / checkpoint_file
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_result, f, ensure_ascii=False, indent=2)
                print(f"  Saved: {checkpoint_file}")

            # 청크 리스트를 단일 flat 리스트로 합치기
            flat_results = [r for chunk in all_results for r in chunk]

            # 최종 통계
            total_time    = time.time() - overall_start
            success_count = sum(1 for r in flat_results if r.get('_success', False))
            successful    = [r for r in flat_results if r.get('_success', False)]
            total_tokens  = sum(r.get('_total_tokens', 0) for r in successful)
            avg_input     = sum(r.get('_input_tokens', 0) for r in successful) / max(success_count, 1)
            avg_output    = sum(r.get('_output_tokens', 0) for r in successful) / max(success_count, 1)

            print(f"\n{'=' * 70}")
            print("FINAL RESULTS")
            print(f"{'=' * 70}")
            print(f"  Total processed: {len(flat_results):,}")
            print(f"  Success:         {success_count:,} ({100 * success_count / len(flat_results):.1f}%)")
            print(f"  Failed:          {len(flat_results) - success_count:,}")
            print(f"  Total time:      {total_time / 60:.1f} min ({total_time / 3600:.2f} hours)")
            print(f"  Throughput:      {len(flat_results) / total_time:.2f} samples/s")
            print(f"  Total tokens:    {total_tokens:,}")
            print(f"  Avg input:       {avg_input:.1f} tokens")
            print(f"  Avg output:      {avg_output:.1f} tokens")
            print(f"{'=' * 70}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 정상 완료 시에만 체크포인트 삭제
            # (KeyboardInterrupt 등 비정상 종료 시 복구 가능하도록 보존)
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            return flat_results

        except KeyboardInterrupt:
            print("\n\n처리가 중단되었습니다.")
            print(f"부분 결과는 체크포인트에 보존: {checkpoint_dir}")
            if all_results:
                return [r for chunk in all_results for r in chunk]
            raise


# ============================================================================
# 테스트 실행 (__main__)
# ============================================================================

if __name__ == "__main__":
    import os
    import warnings
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret
    from maude_early_alert.loaders.text_extract import (
        MDRExtractor,
        build_mdr_text_extract_sql,
    )
    from maude_early_alert.loaders.snowflake_load import SnowflakeLoader
    # ------------------------------------------------------------------
    # 1. Snowflake 연결
    # ------------------------------------------------------------------
    secret = get_secret('snowflake/silver/credentials')
    conn = snowflake.connector.connect(
        user=secret['user'],
        password=secret['password'],
        account=secret['account'],
        warehouse=secret['warehouse'],
        database=secret['database'],
        schema=secret['schema'],
    )
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 2. MDR_TEXT 추출 SQL 빌드 및 실행 (파이프라인과 동일한 코드)
    # ------------------------------------------------------------------
    sql = build_mdr_text_extract_sql(table_name="EVENT_STAGE_12", limit=100)
    print("=== MDR_TEXT 추출 SQL ===")
    print(sql)
    print()

    # 아래 3줄이 파이프라인에서 실행되는 코드입니다
    result = cursor.execute(sql)
    mdr_text = result.fetchall()
    unique_mdr_text = list(set(r[0] for r in mdr_text if r[0]))

    print(f"Fetched {len(mdr_text)} rows → {len(unique_mdr_text)} unique MDR texts")

    # ------------------------------------------------------------------
    # 3. MDRExtractor(파사드)로 배치 처리
    # ------------------------------------------------------------------
    extractor = MDRExtractor(
        model_path='Qwen/Qwen3-14B-AWQ',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=16384,
        max_num_batched_tokens=32768,
        max_num_seqs=128,
        max_retries=2,
        enable_prefix_caching=True,
    )
    results = extractor.process_batch(
        unique_mdr_text,
        checkpoint_dir='./checkpoints',
        checkpoint_interval=5000,
        checkpoint_prefix='maude',
    )

    # ------------------------------------------------------------------
    # 4. Snowflake 적재 (EVENT_STAGE_12_EXTRACTED 에 MERGE)
    # ------------------------------------------------------------------
    loader = SnowflakeLoader(secret['database'], secret['schema'])
    count = loader.load_extraction_results(
        cursor=cursor,
        results=results,
        base_table_name='EVENT_STAGE_12',
    )
    print(f"\n최종 적재 완료: {count:,}건")

    # ------------------------------------------------------------------
    # 5. JOIN SQL 확인 (EVENT_STAGE_12 LEFT JOIN EVENT_STAGE_12_EXTRACTED)
    # ------------------------------------------------------------------
    join_sql = loader.build_extracted_join_sql(base_table_name='EVENT_STAGE_12')
    print("\n=== 조회용 JOIN SQL ===")
    print(join_sql)

    # vLLM 엔진을 명시적으로 먼저 종료 (GC 순서 문제로 인한 spurious 에러 방지)
    del extractor

    cursor.close()
    conn.close()
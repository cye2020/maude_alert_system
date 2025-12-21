# ============================================================================
# MAUDEExtractor
# ============================================================================
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer
from pydantic import BaseModel
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Union
from functools import partial
from tqdm import tqdm, trange
import shutil

from src.preprocess.prompt import Prompt


class MAUDEExtractor:
    def __init__(self, 
                 model_path='Qwen/Qwen3-8B-Instruct',
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.85,
                 max_model_len=8192,
                 max_num_batched_tokens=16384,
                 max_num_seqs=256,
                 max_retries=2,
                 enable_prefix_caching=True,
                 prompt: Prompt = Prompt()
        ):
        """
        vLLM 최적화 배치 추출기 (Qwen3-8B)
        
        Args:
            model_path: 모델 경로
            tensor_parallel_size: 사용할 GPU 수
            gpu_memory_utilization: GPU 메모리 사용률 (0.0-1.0)
            max_model_len: 최대 시퀀스 길이
            max_num_batched_tokens: Chunked prefill 토큰 수 (throughput 핵심)
            max_num_seqs: 동시 처리 시퀀스 수
            max_retries: 실패 시 재시도 횟수
            enable_prefix_caching: 반복 프롬프트 캐싱 활성화
        """
        self.max_retries = max_retries
        self.model_path = model_path
        
        print(f"="*70)
        print(f"Loading vLLM model: {model_path}")
        print(f"="*70)
        print(f"Configuration:")
        print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
        print(f"  - GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"  - Max Model Length: {max_model_len}")
        print(f"  - Max Batched Tokens: {max_num_batched_tokens}")
        print(f"  - Max Num Seqs: {max_num_seqs}")
        print(f"  - Prefix Caching: {enable_prefix_caching}")
        print(f"  - Max Retries: {max_retries}")
        
        self.prompt = prompt
        self.extraction_model = prompt.get_extraction_model()
        
        # vLLM 모델 초기화 (최적화 설정)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,  # Chunked prefill
            max_num_seqs=max_num_seqs,                      # 동시 시퀀스 수
            trust_remote_code=True,
            enforce_eager=False,                            # CUDA graph 사용
            enable_prefix_caching=enable_prefix_caching,    # 프롬프트 캐싱
            disable_log_stats=False,                        # 통계 로깅
            swap_space=4,                                   # Preemption 대비
        )
        
        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"\n✓ Model loaded successfully!\n")
        
        # JSON 스키마 및 샘플링 파라미터
        self.json_schema = self.extraction_model.model_json_schema()
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            top_p=0.95,
            structured_outputs=StructuredOutputsParams(
                json=self.json_schema,
            )
        )

    def _create_prompts(self, rows: List[pd.Series]) -> List[str]:
        """Chat template을 적용한 프롬프트 생성"""
        prompts = []
        
        for row in rows:
            text = row['mdr_text']
            product_problem = row['product_problems']
            
            user_content = self.prompt.format_user_prompt(
                text=text,
                product_problem=product_problem
            )
            
            messages = [
                {"role": "system", "content": self.prompt.SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content}
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            prompts.append(formatted_prompt)
        
        return prompts

    def _parse_and_validate(self, response_text: str) -> dict:
        """응답 파싱 및 검증"""
        data = json.loads(response_text)
        validated = self.extraction_model(**data)
        return validated.model_dump()

    def _generate_and_parse(self, rows: List[pd.Series]) -> tuple[List[dict], dict]:
        """
        vLLM 추론 및 결과 파싱
        
        Returns:
            (results, stats): 파싱된 결과 리스트와 통계 딕셔너리
        """
        batch_start = time.time()
        
        # 프롬프트 생성 및 vLLM 추론
        prompts = self._create_prompts(rows)
        outputs = self.llm.generate(
            prompts, 
            self.sampling_params, 
            use_tqdm=partial(tqdm, mininterval=10.0)
        )
        
        batch_time = time.time() - batch_start
        
        # 결과 파싱
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for _, (row, output) in enumerate(zip(rows, outputs)):
            try:
                response_text = output.outputs[0].text
                validated_data = self._parse_and_validate(response_text)
                
                input_tokens = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                result = {
                    **validated_data,
                    '_row_id': row.name,
                    '_success': True,
                    '_input_tokens': input_tokens,
                    '_output_tokens': output_tokens,
                    '_total_tokens': input_tokens + output_tokens
                }
                results.append(result)
                
            except Exception as e:
                results.append({
                    '_row_id': row.name,
                    '_success': False,
                    '_error': str(e)[:200],
                    '_raw_response': output.outputs[0].text[:200]
                })
        
        # 통계 수집
        stats = {
            'batch_time': batch_time,
            'num_samples': len(rows),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
        }
        
        return results, stats
    
    def _print_batch_stats(self, stats: dict):
        """배치 처리 통계 출력"""
        throughput = stats['num_samples'] / stats['batch_time'] if stats['batch_time'] > 0 else 0
        total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
        tokens_per_sec = total_tokens / stats['batch_time'] if stats['batch_time'] > 0 else 0
        
        print(f"  Batch: {stats['num_samples']:4d} samples | "
              f"Time: {stats['batch_time']:6.2f}s | "
              f"Throughput: {throughput:5.1f} samples/s | "
              f"{tokens_per_sec:6.0f} tokens/s")

    def process_with_retry(self, df: pd.DataFrame) -> pd.DataFrame:
        """재시도 로직 포함 배치 처리"""
        all_results = {}
        pending_df = df.copy()
        attempt = 1

        while not pending_df.empty and attempt <= self.max_retries:
            if attempt > 1:
                print(f"\n  Retry attempt {attempt}: {len(pending_df)} failed samples")

            # vLLM 추론 및 파싱
            rows = [row for _, row in pending_df.iterrows()]
            results, stats = self._generate_and_parse(rows)
            
            # 통계 출력
            self._print_batch_stats(stats)
            
            # 결과 저장 및 실패 항목 추적
            failed_indices = []
            for row, result in zip(pending_df.itertuples(), results):
                result['_attempts'] = attempt
                all_results[row.Index] = result
                
                if not result['_success']:
                    failed_indices.append(row.Index)
            
            # 다음 재시도 준비
            pending_df = df.loc[failed_indices] if failed_indices else pd.DataFrame()
            attempt += 1

        # retry 초과 항목 처리
        for idx in pending_df.index:
            if idx not in all_results:
                all_results[idx] = {
                    '_row_id': idx,
                    '_success': False,
                    '_error': 'Max retries exceeded',
                    '_attempts': self.max_retries
                }

        # 원래 순서로 정렬
        ordered = [all_results[idx] for idx in sorted(all_results.keys())]
        return pd.json_normalize(ordered)

    def process_batch(self, 
                     df: pd.DataFrame, 
                     checkpoint_dir: Union[str, Path], 
                     checkpoint_interval: int = 5000,
                     checkpoint_prefix: str = 'checkpoint') -> pd.DataFrame:
        """
        전체 데이터프레임 처리 with 체크포인트
        
        Args:
            df: 처리할 데이터프레임
            checkpoint_dir: 체크포인트 저장 디렉토리
            checkpoint_interval: 체크포인트 저장 주기 (행 수)
            checkpoint_prefix: 체크포인트 파일 접두사
        """
        print(f"="*70)
        print(f"vLLM Batch Processing Started")
        print(f"="*70)
        print(f"Total records: {len(df):,}")
        print(f"Checkpoint interval: {checkpoint_interval:,} rows")
        print(f"Max retries per chunk: {self.max_retries}")
        print(f"="*70)
        
        overall_start = time.time()
        all_results = []
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            num_chunks = (len(df) - 1) // checkpoint_interval + 1
            
            for chunk_idx in trange((num_chunks), desc="Processing chunks"):
                start_idx = chunk_idx * checkpoint_interval
                end_idx = min((chunk_idx + 1) * checkpoint_interval, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                print(f"\nChunk {chunk_idx + 1}/{num_chunks} | Rows {start_idx:,}-{end_idx-1:,}")
                
                chunk_start = time.time()
                
                # 재시도 포함 처리
                chunk_result_df = self.process_with_retry(chunk_df)
                all_results.append(chunk_result_df)
                
                # 청크 통계
                elapsed = time.time() - chunk_start
                success = chunk_result_df['_success'].sum()
                throughput = len(chunk_df) / elapsed if elapsed > 0 else 0
                
                print(f"  Chunk completed: {success}/{len(chunk_df)} success "
                      f"({100*success/len(chunk_df):.1f}%) | "
                      f"{elapsed:.1f}s | {throughput:.2f} samples/s")
                
                # 체크포인트 저장
                checkpoint_file = f'{checkpoint_prefix}_chunk{chunk_idx+1:03d}.csv'
                checkpoint_path = checkpoint_dir / checkpoint_file
                chunk_result_df.to_csv(checkpoint_path, index=False)
                print(f"  Saved: {checkpoint_file}")
            
            # 최종 결과 합치기
            final_df = pd.concat(all_results, ignore_index=True)
            
            # 최종 통계
            total_time = time.time() - overall_start
            total_success = final_df['_success'].sum()
            
            print(f"\n{'='*70}")
            print(f"FINAL RESULTS")
            print(f"{'='*70}")
            print(f"Total processed:  {len(final_df):,}")
            print(f"Success:          {total_success:,} ({100*total_success/len(final_df):.1f}%)")
            print(f"Failed:           {len(final_df)-total_success:,}")
            print(f"Total time:       {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
            print(f"Throughput:       {len(final_df)/total_time:.2f} samples/s")
            print(f"Total tokens:     {final_df['_total_tokens'].sum():,}")
            print(f"Avg input:        {final_df['_input_tokens'].mean():.1f} tokens")
            print(f"Avg output:       {final_df['_output_tokens'].mean():.1f} tokens")
            print(f"{'='*70}")
            
            return final_df
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Processing interrupted by user")
            print(f"Partial results saved in: {checkpoint_dir}")
            if all_results:
                partial_df = pd.concat(all_results, ignore_index=True)
                return partial_df
            raise
        
        finally:
            # 체크포인트 파일 정리 (선택적)
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            pass


# ============================================================================
# 실행 코드
# ============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # 1. 데이터 로드
    print("Loading data...")
    df = pd.read_csv('your_maude_data.csv')  # 실제 파일 경로로 변경
    
    # 테스트용: 일부만 처리
    # df = df.head(1000)
    
    print(f"Loaded {len(df):,} rows")
    
    # 2. Extractor 초기화
    extractor = MAUDEExtractor(
        model_path='Qwen/Qwen3-8B-Instruct',
        tensor_parallel_size=1,                # GPU 1개 사용
        gpu_memory_utilization=0.80,      # 0.85 → 0.80 (메모리 더 필요)
        max_model_len=16384,              # 8192 → 16384
        max_num_batched_tokens=32768,     # 16384 → 32768 (2배)
        max_num_seqs=128,                 # 256 → 128 (동시 처리 줄이기)
        max_retries=2,
        enable_prefix_caching=True             # 시스템 프롬프트 캐싱
    )
    
    # 3. 배치 처리 실행
    result_df = extractor.process_batch(
        df=df,
        checkpoint_dir='./checkpoints',
        checkpoint_interval=5000,              # 5천 개마다 체크포인트
        checkpoint_prefix='maude'
    )
    
    # 4. 최종 결과 저장
    output_path = 'maude_extraction_results.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Final results saved to: {output_path}")
    
    # 5. 간단한 통계 출력
    print(f"\nQuick Stats:")
    print(f"  - Total: {len(result_df):,}")
    print(f"  - Success: {result_df['_success'].sum():,}")
    print(f"  - Failed: {(~result_df['_success']).sum():,}")
    
    if '_attempts' in result_df.columns:
        print(f"\nRetry Statistics:")
        print(result_df['_attempts'].value_counts().sort_index())
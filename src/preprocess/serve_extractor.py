# ============================================================================
# MAUDEExtractor (vLLM Serve 방식)
# ============================================================================
from openai import OpenAI
from transformers import AutoTokenizer
from pydantic import BaseModel
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm, trange
import shutil
import gc
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.preprocess.prompt import Prompt


class MAUDEExtractor:
    def __init__(self, 
                 base_url: str = "http://localhost:7000/v1",
                 model_name: str = 'Qwen/Qwen3-8B',
                 api_key: str = "EMPTY",
                 max_retries: int = 2,
                 timeout: int = 60,
                 enable_reasoning: bool = True,
                 save_reasoning: bool = True,
                 max_workers: int = 4,  # API 병렬 호출 수
                 prompt: Prompt = Prompt()
        ):
        """
        vLLM Serve 기반 배치 추출기 (OpenAI Compatible API)
        
        Args:
            base_url: vLLM 서버 URL (기본: http://localhost:7000/v1)
            model_name: 모델 이름 (서버에 로드된 모델과 일치해야 함)
            api_key: API 키 (vLLM은 "EMPTY" 사용)
            max_retries: 실패 시 재시도 횟수
            timeout: API 호출 타임아웃 (초)
            enable_reasoning: Reasoning 기능 활성화 (Qwen3)
            save_reasoning: Reasoning 내용을 결과에 저장할지 여부
            max_workers: 동시 API 호출 수 (병렬 처리)
            prompt: 프롬프트 템플릿
        """
        self.max_retries = max_retries
        self.model_name = model_name
        self.enable_reasoning = enable_reasoning
        self.save_reasoning = save_reasoning
        self.max_workers = max_workers
        self.timeout = timeout
        
        print(f"="*70)
        print(f"Connecting to vLLM Server")
        print(f"="*70)
        print(f"Configuration:")
        print(f"  - Server URL: {base_url}")
        print(f"  - Model: {model_name}")
        print(f"  - Reasoning Enabled: {enable_reasoning}")
        print(f"  - Save Reasoning: {save_reasoning}")
        print(f"  - Max Workers (parallel): {max_workers}")
        print(f"  - Max Retries: {max_retries}")
        print(f"  - Timeout: {timeout}s")
        
        # OpenAI Client 초기화
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        
        # 서버 연결 테스트
        try:
            models = self.client.models.list()
            print(f"\n✓ Server connected successfully!")
            print(f"  Available models: {[m.id for m in models.data]}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to vLLM server at {base_url}\n"
                f"Error: {e}\n"
                f"Make sure the server is running:\n"
                f"  vllm serve {model_name} --reasoning-parser qwen3 --port 7000"
            )
        
        self.prompt = prompt
        self.extraction_model = prompt.get_extraction_model()
        
        # Tokenizer 로드 (토큰 카운팅용)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # JSON 스키마
        self.json_schema = self.extraction_model.model_json_schema()
        
        print(f"\n✓ Extractor initialized successfully!\n")

    def _create_messages(self, text: str, product_problem: str) -> List[dict]:
        """Chat 메시지 생성"""
        user_content = self.prompt.format_user_prompt(
            text=text,
            product_problem=product_problem
        )
        
        return [
            {"role": "system", "content": self.prompt.SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content}
        ]

    def _parse_and_validate(self, response_text: str) -> dict:
        """응답 파싱 및 검증"""
        data = json.loads(response_text)
        validated = self.extraction_model(**data)
        return validated.model_dump()

    def _extract_single(self, row: pd.Series, row_idx: int) -> dict:
        """
        단일 샘플 추출 (API 호출)
        
        Returns:
            결과 딕셔너리 (성공 시 추출 데이터 + 메타데이터)
        """
        try:
            messages = self._create_messages(
                text=row['mdr_text'],
                product_problem=row['product_problems']
            )
            
            # API 호출
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                top_p=0.95,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_schema",
                        "strict": True,
                        "schema": self.json_schema
                    }
                },
                # Qwen3 reasoning 비활성화하려면:
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            api_time = time.time() - start_time
            
            # 응답 추출
            message = response.choices[0].message
            content = message.content
            reasoning = getattr(message, 'reasoning', None) if self.enable_reasoning else None
            
            # JSON 파싱 및 검증
            validated_data = self._parse_and_validate(content)
            
            # 토큰 카운팅
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # 결과 구성
            result = {
                **validated_data,
                '_row_id': row_idx,
                '_success': True,
                '_input_tokens': input_tokens,
                '_output_tokens': output_tokens,
                '_total_tokens': input_tokens + output_tokens,
                '_api_time': api_time,
            }
            
            # Reasoning 저장 (옵션)
            if self.save_reasoning and reasoning:
                result['_reasoning'] = reasoning
                result['_reasoning_length'] = len(reasoning)
            
            return result
            
        except Exception as e:
            return {
                '_row_id': row_idx,
                '_success': False,
                '_error': str(e)[:200],
                '_error_type': type(e).__name__
            }

    def _generate_and_parse_parallel(self, rows: List[tuple]) -> tuple[List[dict], dict]:
        """
        병렬 API 호출 및 결과 파싱
        
        Args:
            rows: [(row_idx, row_data), ...] 튜플 리스트
            
        Returns:
            (results, stats): 파싱된 결과 리스트와 통계 딕셔너리
        """
        batch_start = time.time()
        results = [None] * len(rows)
        
        # 병렬 API 호출
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Future 제출
            future_to_idx = {
                executor.submit(self._extract_single, row, idx): i 
                for i, (idx, row) in enumerate(rows)
            }
            
            # 진행 표시줄과 함께 결과 수집
            with tqdm(total=len(rows), desc="API calls", leave=False) as pbar:
                for future in as_completed(future_to_idx):
                    result_idx = future_to_idx[future]
                    results[result_idx] = future.result()
                    pbar.update(1)
        
        batch_time = time.time() - batch_start
        
        # 통계 수집
        successful = [r for r in results if r['_success']]
        total_input_tokens = sum(r.get('_input_tokens', 0) for r in successful)
        total_output_tokens = sum(r.get('_output_tokens', 0) for r in successful)
        
        stats = {
            'batch_time': batch_time,
            'num_samples': len(rows),
            'num_success': len(successful),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
        }
        
        return results, stats
    
    def _print_batch_stats(self, stats: dict):
        """배치 처리 통계 출력"""
        throughput = stats['num_samples'] / stats['batch_time'] if stats['batch_time'] > 0 else 0
        total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
        tokens_per_sec = total_tokens / stats['batch_time'] if stats['batch_time'] > 0 else 0
        success_rate = stats['num_success'] / stats['num_samples'] * 100 if stats['num_samples'] > 0 else 0
        
        print(f"  Batch: {stats['num_samples']:4d} samples | "
              f"Success: {stats['num_success']:4d} ({success_rate:.1f}%) | "
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

            # 행 인덱스와 함께 튜플 리스트 생성
            rows = [(idx, row) for idx, row in pending_df.iterrows()]
            
            # 병렬 API 호출
            results, stats = self._generate_and_parse_parallel(rows)
            
            # 통계 출력
            self._print_batch_stats(stats)
            
            # 결과 저장 및 실패 항목 추적
            failed_indices = []
            for (idx, _), result in zip(rows, results):
                result['_attempts'] = attempt
                all_results[idx] = result
                
                if not result['_success']:
                    failed_indices.append(idx)
            
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
        print(f"vLLM API Batch Processing Started")
        print(f"="*70)
        print(f"Total records: {len(df):,}")
        print(f"Checkpoint interval: {checkpoint_interval:,} rows")
        print(f"Max retries per chunk: {self.max_retries}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"="*70)
        
        overall_start = time.time()
        all_results = []
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            num_chunks = (len(df) - 1) // checkpoint_interval + 1
            
            for chunk_idx in trange(num_chunks, desc="Processing chunks"):
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
            
            if '_total_tokens' in final_df.columns:
                print(f"Total tokens:     {final_df['_total_tokens'].sum():,}")
                print(f"Avg input:        {final_df['_input_tokens'].mean():.1f} tokens")
                print(f"Avg output:       {final_df['_output_tokens'].mean():.1f} tokens")
            
            if self.save_reasoning and '_reasoning_length' in final_df.columns:
                print(f"Avg reasoning:    {final_df['_reasoning_length'].mean():.1f} chars")
            
            print(f"{'='*70}")
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
            # if checkpoint_dir.exists():
            #     shutil.rmtree(checkpoint_dir)
            pass


# ============================================================================
# 실행 코드
# ============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    import sys
    from pathlib import Path

    # 상대 경로 사용
    PROJECT_ROOT = Path.cwd().parent
    DATA_DIR = PROJECT_ROOT / 'data'

    # 맨 앞에 추가
    sys.path.insert(0, str(PROJECT_ROOT))

    # 이제 import
    from src.loading import DataLoader
    from src.preprocess.text_preprocess import execute


    loader = DataLoader(
        output_file= DATA_DIR / 'silver' / 'maude_All_2025-12-20.parquet',
    )
    
    # 0. vLLM 서버 시작 (별도 터미널에서 먼저 실행)
    """
    vllm serve Qwen/Qwen3-8B \\
        --reasoning-parser qwen3 \\
        --port 7000 \\
        --tensor-parallel-size 1 \\
        --gpu-memory-utilization 0.85 \\
        --max-model-len 16384 \\
        --max-num-batched-tokens 32768 \\
        --max-num-seqs 128 \\
        --enable-prefix-caching
    """
    
    # 1. 데이터 로드
    print("Loading data...")
    df = pd.read_csv('your_maude_data.csv')  # 실제 파일 경로로 변경
    
    # 테스트용: 일부만 처리
    # df = df.head(1000)
    
    print(f"Loaded {len(df):,} rows")
    
    # 2. Extractor 초기화 (vLLM 서버에 연결)
    extractor = MAUDEExtractor(
        base_url="http://localhost:7000/v1",
        model_name='Qwen/Qwen3-8B',
        max_retries=2,
        timeout=60,
        enable_reasoning=True,      # Reasoning 활성화
        save_reasoning=True,         # Reasoning 내용 저장
        max_workers=4,               # API 병렬 호출 수 (조정 가능)
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
    
    # 5. Reasoning 샘플 출력 (옵션)
    if '_reasoning' in result_df.columns:
        print(f"\n{'='*70}")
        print("Sample Reasoning Output:")
        print(f"{'='*70}")
        sample = result_df[result_df['_success'] == True].iloc[0]
        print(f"Input: {sample.get('mdr_text', 'N/A')[:100]}...")
        print(f"\nReasoning:\n{sample['_reasoning'][:500]}...")
        print(f"\nExtracted Data:")
        print(json.dumps({k: v for k, v in sample.items() 
                         if not k.startswith('_')}, indent=2))
    
    # 6. 통계 출력
    print(f"\n{'='*70}")
    print(f"Quick Stats:")
    print(f"{'='*70}")
    print(f"  - Total: {len(result_df):,}")
    print(f"  - Success: {result_df['_success'].sum():,}")
    print(f"  - Failed: {(~result_df['_success']).sum():,}")
    
    if '_attempts' in result_df.columns:
        print(f"\nRetry Statistics:")
        print(result_df['_attempts'].value_counts().sort_index())
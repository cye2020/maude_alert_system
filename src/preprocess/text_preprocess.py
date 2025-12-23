from typing import Union
import time
import gc
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import argparse

import torch

import sys
from pathlib import Path

# 로컬 모듈
from src.loading import DataLoader
from src.utils import increment_path
from src.preprocess.prompt import get_prompt, Prompt
from src.preprocess.extractor import MAUDEExtractor
from src.preprocess.config import get_config


class LLMPreprocessor:
    """LLM 기반 텍스트 전처리 파이프라인"""
    
    def __init__(self):
        # 통합 config 사용
        self.cfg = get_config()
        self.config = self.cfg.llm_extraction
        
        # 경로는 cfg에서
        self.data_dir = Path(self.cfg.base['paths']['local']['root'])
        self.temp_dir = Path(self.cfg.base['paths']['local']['temp'])
        
        # 성능 설정 적용
        if self.config['performance']['memory']['clear_cuda_cache']:
            torch.cuda.empty_cache()
    
    def prepare_data(
        self, 
        input_path: Union[str, Path], 
        n_rows: int = None, 
        random: bool = False
    ) -> pl.LazyFrame:
        """데이터 로드 및 샘플링"""
        
        loader_config = self.config['data_loader']
        
        loader = DataLoader(output_file=Path(input_path))
        
        lf = loader.load(
            adapter=loader_config['adapter'],
            **loader_config['polars_options']
        )
        
        if not n_rows:
            return lf
        
        if random:
            sampled_lf = lf.select(
                pl.all().sample(
                    n=n_rows,
                    with_replacement=False,
                    shuffle=True,
                    seed=self.config['input']['sampling']['random_seed']
                )
            )
        else:
            sort_cols = self.config['preprocessing']['sorting']['columns']
            sampled_lf = lf.sort(sort_cols).head(n_rows)
        
        return sampled_lf
    
    def save_data(
        self,
        lf: pl.LazyFrame,
        result_df: pd.DataFrame,
        prompt: Prompt,
        prompt_path: Union[str, Path],
        result_path: Union[str, Path]
    ):
        """결과 저장"""
        
        prompt_config = self.config['prompt']['save_format']
        
        # 프롬프트 저장
        prompt_text_parts = []
        if prompt_config['include_system']:
            prompt_text_parts.append('[SYSTEM]\n' + prompt.SYSTEM_INSTRUCTION)
        if prompt_config['include_user']:
            prompt_text_parts.append('[USER]' + prompt.USER_PROMPT_TEMPLATE)
        
        prompt_text = prompt_config['separator'].join(prompt_text_parts)
        
        with open(prompt_path, mode='w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        # 데이터 저장
        result_lf = pl.from_pandas(result_df).lazy()
        concat_lf = pl.concat([lf, result_lf], how='horizontal')
        
        format_config = self.config['output']['format']
        concat_lf.sink_parquet(
            result_path,
            compression=format_config['compression'],
            compression_level=format_config['compression_level'],
            mkdir=True
        )
    
    def process_deduplication(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """중복 제거 및 매핑 테이블 생성"""
        
        dedup_config = self.config['preprocessing']['deduplication']
        
        if not dedup_config['enabled']:
            df['_temp_id'] = range(len(df))
            return df, df
        
        dedup_cols = dedup_config['columns']
        
        # 유니크한 조합만 추출
        unique_df = df[dedup_cols].drop_duplicates().reset_index(drop=True)
        unique_df['_temp_id'] = range(len(unique_df))
        
        # 통계 출력
        if self.config['logging']['progress']['show_deduplication_stats']:
            ratio = len(unique_df) / len(df) * 100
            saved = len(df) - len(unique_df)
            print(f"유니크한 조합: {len(unique_df):,} rows ({ratio:.1f}%)")
            print(f"중복 제거로 {saved:,}개 행 절약")
        
        return unique_df, df
    
    def merge_results(
        self,
        original_df: pd.DataFrame,
        unique_df: pd.DataFrame,
        unique_result_df: pd.DataFrame
    ) -> pd.DataFrame:
        """원본 데이터와 처리 결과 병합"""
        
        dedup_cols = self.config['preprocessing']['deduplication']['columns']
        
        # 유니크 결과에 _temp_id 추가
        unique_result_df['_temp_id'] = unique_df['_temp_id']
        
        # lookup 테이블 생성
        lookup = unique_df[dedup_cols + ['_temp_id']]
        
        # 원본에 _temp_id 매핑
        merged_df = original_df.merge(
            lookup,
            on=dedup_cols,
            how='left'
        )
        
        # 검증
        if self.config['validation']['check_merge_integrity']:
            assert len(merged_df) == len(original_df), "❌ Merge로 행이 증가했습니다!"
        
        # 처리 결과 join
        final_df = merged_df.merge(
            unique_result_df,
            on='_temp_id',
            how='left',
            suffixes=('', '_result')
        )
        
        # 검증
        if self.config['validation']['check_row_count']:
            assert len(final_df) == len(original_df), "❌ 최종 결과 행 수가 원본과 다릅니다!"
        
        return final_df
    
    def get_extraction_columns(self, include_reasoning: bool = False) -> list[str]:
        """추출할 컬럼 목록 반환"""
        
        ext_config = self.config['extraction']
        cols = ext_config['base_fields'].copy()
        
        if include_reasoning:
            suffix = ext_config['reasoning_fields']['suffix']
            reason_cols = [col + suffix for col in cols]
            cols.extend(reason_cols)
        
        return cols
    
    def rename_output_columns(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """출력 컬럼 이름 변경"""
        
        rename_config = self.config['output_columns']
        strategy = rename_config['rename_strategy']
        
        # mdr_report_key + 추출 컬럼만 선택
        result_df = df[['mdr_report_key'] + cols].copy()
        
        # 이름 변경 전략 적용
        if strategy == 'last_segment':
            new_names = ['mdr_report_key'] + [col.split('.')[-1] for col in cols]
            result_df.columns = new_names
        
        return result_df
    
    def execute(self, n_rows: int = None, random: bool = False, reason: bool = False, input_path: str = None):
        """전체 파이프라인 실행"""
        
        # 입력 경로 설정
        input_config = self.config['input']
        input_path = (
            input_path
            if input_path is not None
            else self.data_dir / 
                input_config['source_dir'] / 
                input_config['source_file']
        )
        
        # 출력 경로 설정
        output_config = self.config['output']
        result_dir = self.data_dir / output_config['target_dir']
        
        if output_config['auto_increment']:
            result_dir = increment_path(result_dir, exist_ok=True, mkdir=True)
        else:
            result_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로드
        sampled_lf = self.prepare_data(input_path, n_rows=n_rows, random=random)
        
        # 필요한 컬럼만 수집
        input_cols = self.config['preprocessing']['input_columns']
        sampled_df = sampled_lf.select(pl.col(input_cols)).collect().to_pandas()
        
        print(f"원본 데이터 크기: {len(sampled_df):,} rows")
        
        # 중복 제거
        unique_df, original_df = self.process_deduplication(sampled_df)
        
        # 체크포인트 경로 (base.yaml에서)
        checkpoint_config = self.config['checkpoint']
        checkpoint_dir = self.temp_dir / checkpoint_config['directory']
        
        # 추출할 컬럼
        cols = self.get_extraction_columns(include_reasoning=reason)
        
        # 프롬프트 선택
        prompt_config = self.config['prompt']
        if reason:
            prompt_type = 'sample'
        else:
            prompt_type = prompt_config['type']
        
        prompt = get_prompt(prompt_type)
        
        # LLM 처리
        start_time = time.time()
        
        model_config = self.config['model']
        extractor = MAUDEExtractor(
            model_path=model_config['name'],
            tensor_parallel_size=model_config['vllm']['tensor_parallel_size'],
            gpu_memory_utilization=model_config['vllm']['gpu_memory_utilization'],
            max_model_len=model_config['vllm']['max_model_len'],
            max_num_batched_tokens=model_config['vllm']['max_num_batched_tokens'],
            max_num_seqs=model_config['vllm']['max_num_seqs'],
            max_retries=model_config['retry']['max_retries'],
            enable_prefix_caching=model_config['vllm']['enable_prefix_caching'],
            prompt=prompt
        )
        
        # 유니크 데이터만 처리
        unique_result_df = extractor.process_batch(
            df=unique_df,
            checkpoint_dir=checkpoint_dir if checkpoint_config['enabled'] else None,
            checkpoint_interval=checkpoint_config['interval'],
            checkpoint_prefix=checkpoint_config['prefix']
        )
        
        elapsed = time.time() - start_time
        
        if self.config['logging']['progress']['show_elapsed_time']:
            print(f"Elapsed time: {elapsed:.2f} seconds")
        
        # 결과 병합
        final_result_df = self.merge_results(original_df, unique_df, unique_result_df)
        
        # 결과 컬럼 필터링 (exclude 패턴 제거)
        exclude_patterns = self.config['output_columns']['exclude_patterns']
        result_cols = ['mdr_report_key'] + [
            col for col in final_result_df.columns 
            if col in cols and not any(pattern in col for pattern in exclude_patterns)
        ]
        final_result_df = final_result_df[result_cols]
        
        print(f"✅ 최종 결과 크기: {len(final_result_df):,} rows (원본과 동일)")
        print(final_result_df.columns.tolist())
        
        # 저장
        if not n_rows:
            n_rows = len(final_result_df)
        
        today = datetime.now().date()
        prompt_path = result_dir / f"{output_config['prompt_prefix']}_{n_rows}_{today}.txt"
        result_path = result_dir / f"{output_config['file_prefix']}_{n_rows}_{today}.parquet"
        
        if output_config['auto_increment']:
            sep = output_config['increment_separator']
            prompt_path = increment_path(prompt_path, exist_ok=False, sep=sep)
            result_path = increment_path(result_path, exist_ok=False, sep=sep)
        
        # 컬럼 이름 변경
        rename_result_df = self.rename_output_columns(final_result_df, cols)
        
        # mdr_report_key 제외하고 저장
        save_result_df = rename_result_df.drop(columns=['mdr_report_key'])
        
        self.save_data(sampled_lf, save_result_df, prompt, prompt_path, result_path)
        
        # 메모리 정리
        if self.config['performance']['memory']['clear_cuda_cache']:
            torch.cuda.empty_cache()
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description='LLM text preprocessor.')
    parser.add_argument(
        "--input-path", '-i',
        type=str,
        default=None,
        help="Override input file (default: config value)"
    )

    parser.add_argument(
        '--n-rows', '-n',
        type=int,
        default=None,
        help='Number of rows to process (default: None)'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Random sampling'
    )
    parser.add_argument(
        '--reason',
        action='store_true',
        help='추론도 포함'
    )
    args = parser.parse_args()

    print(f"처리할 행 수: {args.n_rows}")
    
    preprocessor = LLMPreprocessor()
    preprocessor.execute(
        n_rows=args.n_rows,
        random=args.random,
        reason=args.reason,
        input_path=args.input_path
    )


if __name__ == '__main__':
    main()
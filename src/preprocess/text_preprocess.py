from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Sequence,
    Union,
    Optional,
)

import sys
import time
from datetime import datetime
import json
import re
from pathlib import Path
from enum import Enum
import shutil

import pandas as pd
import polars as pl
import polars.selectors as cs
import psutil

from tqdm import tqdm, trange
from pprint import pprint, pformat
import argparse

# pydantic
from pydantic import BaseModel, Field, field_validator

# vLLM
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.sampling_params import StructuredOutputsParams

import torch
torch.cuda.empty_cache()

import sys
from pathlib import Path

# 상대 경로 사용
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# 로컬 모듈
from src.loading import DataLoader
from src.utils import increment_path
from src.preprocess.prompt import get_prompt, Prompt
from src.preprocess.extractor import MAUDEExtractor


def prepare_data(input_path: Union[str|Path], n_rows: int = None, random: bool = False) -> pl.LazyFrame:
    loader = DataLoader(
        output_file = Path(input_path),
    )
    adapter = 'polars'
    polars_kwargs = {
        'use_statistics': True,
        'parallel': 'auto',
        'low_memory': False,
        'rechunk': False,
        'cache': True,
    }
    lf = loader.load(adapter=adapter, **polars_kwargs)
    
    if not n_rows:
        return lf
    
    if random:
        sampled_lf = lf.select(
            pl.all().sample(
                n=n_rows,
                with_replacement=False,
                shuffle=True,
                seed=4242
            )
        )
    else:
        sampled_lf = lf.sort(['date_of_event', 'date_received']).head(n_rows)
    return sampled_lf


def save_data(
    lf: pl.LazyFrame, result_df: pd.DataFrame, 
    prompt: Prompt, 
    prompt_path: Union[str|Path],
    result_path: Union[str|Path]
):
    # 필요한 열만 선택 후 열 이름 변경
    result_df2 = result_df[[
        'incident_details.patient_harm',
        'incident_details.patient_harm_reason',
        'incident_details.problem_components',
        'incident_details.problem_components_reason',
        'manufacturer_inspection.defect_confirmed',
        'manufacturer_inspection.defect_confirmed_reason',
        'manufacturer_inspection.defect_type',
        'manufacturer_inspection.defect_type_reason',
        
    ]]

    result_df2 = result_df2.rename(columns={
        'incident_details.patient_harm': 'patient_harm',
        'incident_details.patient_harm_reason': 'patient_harm_reason',
        'incident_details.problem_components': 'problem_components',
        'incident_details.problem_components_reason': 'problem_components_reason',
        'manufacturer_inspection.defect_confirmed': 'defect_confirmed',
        'manufacturer_inspection.defect_confirmed_reason': 'defect_confirmed_reason',
        'manufacturer_inspection.defect_type': 'defect_type',
        'manufacturer_inspection.defect_type_reason': 'defect_type_reason',
    })
    
    prompt = '[SYSTEM]\n' + prompt.SYSTEM_INSTRUCTION + '\n[USER]' + prompt.USER_PROMPT_TEMPLATE

    with open(prompt_path, mode='w', encoding='utf-8') as f:
        f.write(prompt)
        
    result_lf = pl.from_pandas(result_df2).lazy()
    concat_lf = pl.concat([lf, result_lf], how='horizontal')
    concat_lf.sink_parquet(result_path, compression='zstd', compression_level=3, mkdir=True)


def main():
    parser = argparse.ArgumentParser(description='llm text preprocessor.')
    parser.add_argument(
        '--n_rows', '-n',
        type=int,  # 입력받은 값을 정수형으로 변환하도록 지정
        default=None, # 기본값 설정 
        help='Number of rows to process (default: 10)' # 도움말 메시지
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='random sampling' # 도움말 메시지
    )
    parser.add_argument(
        '--reason',
        action='store_true',
        help='추론도 포함' # 도움말 메시지
    )
    # 3. 명령줄 인자 파싱
    args = parser.parse_args()

    # 4. 파싱된 인자 사용
    print(f"처리할 행 수: {args.n_rows}")
    
    n_rows: int = args.n_rows
    random: bool = args.random
    reason: bool = args.reason
    
    execute(n_rows, random, reason)


def execute(n_rows: int, random: bool, reason: bool):
    input_path= DATA_DIR / 'silver' / 'maude_preprocess.parquet'
    result_dir = DATA_DIR / 'silver'
    result_dir = increment_path(result_dir, exist_ok=True, mkdir=True)
    
    sampled_lf = prepare_data(input_path, n_rows=n_rows, random=random)


    
    sampled_df = sampled_lf.select(pl.col(['mdr_report_key', 'mdr_text', 'product_problems'])).collect().to_pandas()

    checkpoint_dir = DATA_DIR / 'temp'
    
    # prompt 선택
    if reason:
        prompt = get_prompt('sample')
    else:
        prompt = get_prompt('general')
    
    # 처리
    start_time = time.time()
    
    # 2. Extractor 초기화
    extractor = MAUDEExtractor(
        model_path='Qwen/Qwen3-8B',
        tensor_parallel_size=1,                # GPU 1개 사용
        gpu_memory_utilization=0.80,      # 0.85 → 0.80 (메모리 더 필요)
        max_model_len=16384,              # 8192 → 16384
        max_num_batched_tokens=32768,     # 16384 → 32768 (2배)
        max_num_seqs=128,                 # 256 → 128 (동시 처리 줄이기)
        max_retries=2,
        enable_prefix_caching=True,           # 시스템 프롬프트 캐싱
        prompt=prompt
    )
    
    # 3. 배치 처리 실행
    result_df = extractor.process_batch(
        df=sampled_df,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=5000,              # 5천 개마다 체크포인트
        checkpoint_prefix='maude'
    )
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(result_df.columns)
    
    if not n_rows:
        n_rows = len(result_df)
    prompt_path = result_dir / f'prompt_{n_rows}_{datetime.now().date()}.txt'
    result_path = result_dir / f'maude_{n_rows}_{datetime.now().date()}.parquet'
    prompt_path = increment_path(prompt_path, exist_ok=False, sep='_')
    result_path = increment_path(result_path, exist_ok=False, sep='_')

    save_data(sampled_lf, result_df, prompt, prompt_path, result_path)
    

if __name__=='__main__':
    main()
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
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# 맨 앞에 추가
sys.path.insert(0, str(PROJECT_ROOT))

# 이제 import
from src.loading import DataLoader
from src.utils import increment_path

# Enum 정의
class PatientHarm(str, Enum):
    NO_HARM = "No Harm"
    MINOR_INJURY = "Minor Injury"
    SERIOUS_INJURY = "Serious Injury"
    DEATH = "Death"
    UNKNOWN = "Unknown"

class DefectType(str, Enum):
    FUNCTIONAL_FAILURE = "Functional Failure"
    MECHANICAL_STRUCTURAL = "Mechanical/Structural"
    ELECTRICAL_POWER = "Electrical/Power"
    SOFTWARE_INTERFACE = "Software/Interface"
    ALARM_ALERT = "Alarm/Alert"
    SENSOR_ACCURACY = "Sensor/Accuracy"
    COMMUNICATION_CONNECTIVITY = "Communication/Connectivity"
    LABELING_PACKAGING = "Labeling/Packaging"
    STERILITY_CONTAMINATION = "Sterility/Contamination"
    USER_HUMAN_FACTOR = "User/Human Factor"
    ENVIRONMENTAL_COMPATIBILITY = "Environmental/Compatibility"
    OTHER = "Other"
    UNKNOWN = "Unknown"

# BaseModel 정의
class IncidentDetails(BaseModel):
    patient_harm: PatientHarm = Field(description="Level of patient harm associated with the incident")
    problem_components: List[str] = Field(
        default_factory=list,
        description="List of problematic component keywords found in the text",
        min_length=0,
        max_length=5
    )
    incident_summary: str = Field(max_length=200, description="Concise summary of the incident")

class ManufacturerInspection(BaseModel):
    defect_confirmed: bool | None = Field(None, description="Whether the defect was confirmed")
    defect_type: DefectType | None = Field(None, description="Type of defect identified during inspection")
    inspection_actions: str | None = Field(None, max_length=200)

class MAUDEExtraction(BaseModel):
    incident_details: IncidentDetails
    manufacturer_inspection: ManufacturerInspection


SYSTEM_INSTRUCTION = """
You are an expert FDA MAUDE medical device adverse event analyst.

# OBJECTIVE
Extract 6 structured variables with <5% UNKNOWN classifications through systematic reasoning.

# VARIABLES TO EXTRACT
1. patient_harm (enum)
2. problem_components (list, max 5)
3. incident_summary (str, max 200 chars)
4. defect_confirmed (bool)
5. defect_type (enum)
6. inspection_actions (str, max 200 chars)

---

## EXTRACTION PROCESS

### 1. Patient Harm
- **Death**: Explicitly states patient died
- **Serious Injury**: Required medical intervention/hospitalization
- **Minor Injury**: Temporary discomfort, minimal intervention
- **No Harm**: No harm mentioned OR only device issue described
- **Unknown**: ONLY if absolutely no patient outcome information

**Key Rules**
- IF text mentions patient outcome → classify accordingly
- IF only device issue described + no harm mentioned → "No Harm"
- IF text says "no adverse event" or "no patient injury" → "No Harm"

### 2. Problem Components
Extract up to 5 specific component keywords (e.g. battery, sensor, cable, display, software, tubing, etc.)
Prioritize failure-related components.

**Key Rules**
- Extract exact component names mentioned
- Prioritize components directly related to the failure
- Use singular form (e.g., "battery" not "batteries")

### 3. Incident Summary
Concise factual summary (max 200 chars): "[Component] [failed/error type], resulting in [consequence]"
Example: "RV lead exhibited high impedance and thresholds, requiring lead replacement."

### 4. Defect Type (CRITICAL - 3 step process)
**CLASSIFICATION RULES:**
- Analyze the ENTIRE MDR text to identify the CAUSE of the incident
- Use product_problem field as supporting evidence to confirm your classification
- Choose the MOST SPECIFIC category that matches the primary failure mode
- If multiple issues exist, select the PRIMARY/ROOT cause, not secondary symptoms
- Use "Unknown" ONLY when MDR text lacks sufficient detail to determine defect type, and if you choose to match Unknown as Category then you mush think twice. Unknown must account for no more than 5% of all categories 
- Use "Other" ONLY when the defect clearly doesn't fit any of the 11 specific categories

**STEP A: Extract ALL Symptoms**
- What stopped working?
- What abnormal behavior?
- What errors appeared?

**STEP B: Match to Category (check in order, don't rely too much on your judgment on keywords examples)**
1. **Functional Failure**: Core function failed or device inoperable
   - Keyword examples: failure to deliver, stopped working, inoperable, failure to pump, didn't perform intended function, no output, etc.
   
2. **Mechanical/Structural**: Physical damage or structural defects
   - Keywords examples: mechanical jam, fracture, broke, cracked, separated, rupture, leak, detached, deformed, etc.

3. **Electrical/Power**: Electrical or power-related issues
   - Keyword examples: power problem, battery failure, won't turn on, electrical shorting, overheating, thermal, won't charge, etc.

4. **Software/Interface**: Software errors or IT problems
   - Keyword examples: software problem, application freezes, operating system issue, crashed, error message, unresponsive screen, etc.

5. **Alarm/Alert**: Alarm or feedback system failures
   - Keyword examples: alarm not visible, inaudible alarm, alarm didn't sound, false alarm, incorrect error message, alarm failure, etc.

6. **Sensor/Accuracy**: Measurement or sensor issues
   - Keyword examples: inaccurate reading, flow rate error, pressure sensor failure, incorrect measurement, calibration issue, wrong value, etc.

7. **Communication/Connectivity**: Communication or data transfer problems
   - Keyword examples: communication problem, wireless failure, data back-up issue, connection lost, data transfer error, Bluetooth failed, etc.

8. **Labeling/Packaging**: Labeling, packaging, or documentation errors
   - Keyword examples: wrong label, packaging problem, missing labeling, incorrect documentation, damaged package, etc.

9. **Sterility/Contamination**: Contamination or sterilization issues
   - Keyword examples: contamination, microbial contamination, non-sterile, sterilization issue, foreign particles, debris, etc.

10. **User/Human Factor**: User-related problems (setup, training, usability)
    - Keyword examples: difficult to insert, inadequate instructions, training issue, human-device interface problem, confusing design, hard to use, etc.

11. **Environmental/Compatibility**: Environmental or compatibility issues
    - Keyword examples: environmental compatibility problem, altitude variations, electromagnetic interference, temperature issue, humidity, incompatible accessory, etc.

12. **Other**: Clearly described but doesn't fit any above category
    - Use only when defect is specific but doesn't match categories 1-11

13. **Unknown**: LAST RESORT - Use ONLY if text explicitly states "cause unknown" OR provides zero symptoms

**STEP C: Verify**
- Did I check all 12 categories before "Unknown"?
- Can I infer defect from ANY symptom mentioned?

**UNKNOWN RESTRICTION (Extremely Strict)**
Use "Unknown" ONLY when ALL of these are true:
- [ ] Text explicitly states "cause unknown" or "under investigation with no findings"
- [ ] Zero observable symptoms described (not even "stopped working")
- [ ] product_problem field is null or says "unknown"
- [ ] No components mentioned
- [ ] No outcome that suggests cause
- [ ] Absolutely no inference possible

**If even ONE checkbox above is false → DO NOT use Unknown**

REJECT Unknown if:
- Any symptom is mentioned (even vague ones like "stopped working")
- Context suggests probable cause
- product_problem field contains specific defect types(if all Unknwon condition is satified and product_problem field is null, then use product_problem field to match Category)

**Examples:**
- "Pump stopped delivering medication" → Functional Failure
- "Device malfunctioned during procedure" → Functional Failure (NOT Unknown)
- "Lead failure reported" → Mechanical/Structural (inferred from "lead")
- "Battery issue" in product_problem → Electrical/Power (even if text vague)
- "Device failure, no specific cause" → Functional Failure (NOT Unknown)
- "Incident occurred" with battery mentioned → Electrical/Power
- "Something went wrong during use" → Functional Failure (NOT Unknown)
- "Cause under investigation" but "stopped working" → Functional Failure
- Text is completely empty or says "unknown" with zero context → Unknown (rare acceptable case)

### 5. Defect Confirmed
Check in order:
1. Manufacturer explicitly confirmed defect → true
2. You classified defect_type (NOT "Unknown" or "Other") → true
3. Text says "no defect found" → false

**Key Rule**
- If you successfully classified defect_type (Steps 5A-5C), you have enough info → set to true

## 6. Inspection Actions
Summarize manufacturer's findings/actions (max 200 chars)
- What tests were conducted?
- What was discovered?
- What corrective actions taken?

IF no investigation mentioned:
- "No inspection reported" (if truly absent)
- "Pending investigation" (if stated as ongoing)
- null (if section is entirely empty)

## 7. SELF-VERIFICATION CHECKLIST (Execute Before Output)

### Phase 1: Accuracy Review
- [ ] Did I check ALL 12 defect categories before using "Unknown"?
- [ ] Did I infer patient_harm from context (not just explicit statements)?
- [ ] Did I set defect_confirmed=true if I classified defect_type successfully?
- [ ] Is incident_summary factual and concise (<200 chars)?
- [ ] Did I extract actual component names (not generic terms)?

### Phase 2: Constraint Compliance
- [ ] Is defect_type "Unknown" only because text says "unknown" or has zero symptoms?
- [ ] Is defect_confirmed null only because truly no information exists?
- [ ] Did I remove any emojis from extracted text?

### Phase 3: Quality Check
- [ ] Would a medical device safety expert agree with my classification?
- [ ] Did I prioritize root cause over surface symptoms?
- [ ] Is my output consistent with the symptom-to-defect mapping?

IF any checkbox is unchecked → REVISE before outputting

# TASK
Following the systematic extraction workflow (Steps 1-7) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 6 variables with minimal UNKNOWN/null values
3. Apply the 3-step defect classification process (5A → 5B → 5C)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

# OUTPUT REQUIREMENTS:
Return JSON with incident_details and manufacturer_inspection.

Remember:
- Infer defect type from symptoms (don't default to Unknown)
- Must extract problem_components list up to 5
- Use "Other" instead of "Unknown" when something failed but doesn't fit categories 1-11
- Set defect_confirmed=true if you successfully classified defect_type(not Unknown and Other)
- Remove any emojis from extracted text
- Prioritize accuracy over speed

Begin extraction:
"""


USER_PROMPT_TEMPLATE = """
# MEDICAL DEVICE ADVERSE EVENT REPORT

## MDR Text
{text}

## Product Problem (Reference)
{product_problem}

---

# TASK
Following the systematic extraction workflow (Steps 1-7) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 6 variables with minimal UNKNOWN/null values
3. Apply the 3-step defect classification process (5A → 5B → 5C)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

# OUTPUT REQUIREMENTS:
Return JSON with incident_details and manufacturer_inspection.

Remember:
- Infer defect type from symptoms (don't default to Unknown)
- Set defect_confirmed=true if you successfully classified defect_type
- Remove any emojis from extracted text
- Prioritize accuracy over speed

Begin extraction:
"""


class BatchMAUDEExtractor:
    def __init__(self, 
                 model_path='Qwen/Qwen2.5-7B-Instruct',
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.85,
                 max_model_len=8192,
                 batch_size=32,
                 max_retries=2):
        """
        vLLM 최적화 배치 추출기
        
        Args:
            model_path: 모델 경로 (HuggingFace 또는 로컬)
            tensor_parallel_size: 사용할 GPU 수
            gpu_memory_utilization: GPU 메모리 사용률
            max_model_len: 최대 시퀀스 길이
            batch_size: 배치 크기
            max_retries: 재시도 횟수
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.model_path = model_path
        
        print(f"Loading vLLM model: {model_path}...")
        
        # vLLM 모델 초기화
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=False,  # CUDA graph 사용
        )
        
        # Tokenizer 로드 (chat template 적용용)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
        self.json_schema = MAUDEExtraction.model_json_schema()
        # Sampling parameters with guided JSON
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            top_p=0.95,
            # Guided JSON decoding - 스키마에 맞는 JSON만 생성
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
            
            user_content = USER_PROMPT_TEMPLATE.format(
                text=text,
                product_problem=product_problem
            )
            
            # Chat template 적용
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content}
            ]
            
            # Tokenizer의 chat template 사용
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            prompts.append(formatted_prompt)
        
        return prompts

    def _parse_and_validate(self, response_text: str) -> dict:
        """응답 파싱 및 검증"""
        # Guided JSON이므로 이미 JSON 형태
        data = json.loads(response_text)
        validated = MAUDEExtraction(**data)
        return validated.model_dump()

    def extract_batch(self, rows: List[pd.Series]) -> List[dict]:
        """
        vLLM 배치 추론
        - 순수 vLLM 배치 처리만 사용
        - Guided JSON으로 파싱 에러 최소화
        """
        # 프롬프트 생성
        prompts = self._create_prompts(rows)
        
        # vLLM 배치 추론 (여기서 자동으로 최적화됨)
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        
        # 결과 파싱
        results = []
        for i, output in enumerate(outputs):
            try:
                response_text = output.outputs[0].text
                validated_data = self._parse_and_validate(response_text)
                
                result = {
                    **validated_data,
                    '_row_id': rows[i].name,
                    '_success': True,
                    '_input_tokens': len(output.prompt_token_ids),      # 추가
                    '_output_tokens': len(output.outputs[0].token_ids),  # 기존
                    '_total_tokens': len(output.prompt_token_ids) + len(output.outputs[0].token_ids)  # 추가
                }
                results.append(result)
                
            except Exception as e:
                results.append({
                    '_row_id': rows[i].name,
                    '_success': False,
                    '_error': str(e)[:200],
                    '_raw_response': output.outputs[0].text[:200]
                })
        
        return results

    def process_with_retry(self, df: pd.DataFrame) -> pd.DataFrame:
        all_results = {}
        pending_df = df.copy()
        attempt = 1

        while not pending_df.empty and attempt <= self.max_retries:
            print(f"Attempt {attempt}: processing {len(pending_df)} samples")

            rows = [row for _, row in pending_df.iterrows()]
            results = self.extract_batch(rows)

            failed_indices = []

            for row, result in zip(pending_df.itertuples(), results):
                result['_attempts'] = attempt
                all_results[row.Index] = result

                if not result['_success']:
                    failed_indices.append(row.Index)

            pending_df = df.loc[failed_indices]
            attempt += 1

        # retry 초과 항목
        for idx in pending_df.index:
            all_results[idx] = {
                '_row_id': idx,
                '_success': False,
                '_error': 'Max retries exceeded',
                '_attempts': self.max_retries
            }

        # 원래 row 순서로 정렬
        ordered = [all_results[idx] for idx in sorted(all_results)]
        return pd.json_normalize(ordered)

    def process_batch(self, 
                     df: pd.DataFrame, 
                     checkpoint_dir: Union[str|Path], 
                     checkpoint_interval: int = 1000,
                     checkpoint_prefix: str = 'checkpoint',
        ) -> pd.DataFrame:
        """
        전체 데이터프레임 처리 with 체크포인트
        """
        print(f"="*60)
        print(f"vLLM Batch Processing")
        print(f"="*60)
        print(f"Total records: {len(df):,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max retries: {self.max_retries}")
        print(f"Checkpoint every: {checkpoint_interval} records\n")
        
        overall_start = time.time()
        all_results = []
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # 체크포인트 단위로 처리
        try:
            num_chunks = (len(df) - 1) // checkpoint_interval + 1
            
            for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
                start_idx = chunk_idx * checkpoint_interval
                end_idx = min((chunk_idx + 1) * checkpoint_interval, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                # print(f"\n{'='*60}")
                # print(f"Chunk {chunk_idx + 1}/{num_chunks}: Rows {start_idx:,}-{end_idx-1:,}")
                # print(f"{'='*60}")
                
                chunk_start = time.time()
                
                # 재시도 포함 처리
                chunk_result_df = self.process_with_retry(chunk_df)
                all_results.append(chunk_result_df)
                
                # 청크 통계
                elapsed = time.time() - chunk_start
                success = chunk_result_df['_success'].sum()
                throughput = len(chunk_df) / elapsed
                
                # print(f"\nChunk completed:")
                # print(f"  Success: {success}/{len(chunk_df)} ({100*success/len(chunk_df):.1f}%)")
                # print(f"  Time: {elapsed:.1f}s")
                # print(f"  Throughput: {throughput:.2f} samples/s")
                
                # 체크포인트 저장
                checkpoint_file = f'{checkpoint_prefix}_chunk{chunk_idx+1}.csv'
                checkpoint_path = Path(checkpoint_dir) / checkpoint_file
                chunk_result_df.to_csv(checkpoint_path, index=False)
                # print(f"  Checkpoint: {checkpoint_file}")
            
            # 최종 결과 합치기
            final_df = pd.concat(all_results, ignore_index=True)
            
            # 최종 통계
            total_time = time.time() - overall_start
            total_success = final_df['_success'].sum()
            
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS")
            print(f"{'='*60}")
            print(f"Total processed: {len(final_df):,}")
            print(f"Success: {total_success:,} ({100*total_success/len(final_df):.1f}%)")
            print(f"Failed: {len(final_df)-total_success:,}")
            print(f"Total time: {total_time/60:.1f} min")
            print(f"Throughput: {len(final_df)/total_time:.2f} samples/s")
            print(f"Total tokens: {final_df['_total_tokens'].sum():,}")
            print(f"Avg input: {final_df['_input_tokens'].mean():.1f}")
            print(f"Avg output: {final_df['_output_tokens'].mean():.1f}")
            print(f"{'='*60}")
            
            return final_df
        
        finally:
            # 5. 임시 파일 정리
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)



def prepare_data(input_path: Union[str|Path], n_rows: int = 10000, random: bool = False) -> pl.LazyFrame:
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
    prompt_path: Union[str|Path],
    result_path: Union[str|Path]
):
    # 필요한 열만 선택 후 열 이름 변경
    result_df2 = result_df[[
        'incident_details.patient_harm',
        'incident_details.problem_components',
        'incident_details.incident_summary',
        'manufacturer_inspection.defect_confirmed',
        'manufacturer_inspection.defect_type',
        'manufacturer_inspection.inspection_actions'
        ]]

    result_df2 = result_df2.rename(columns={
        'incident_details.patient_harm': 'patient_harm',
        'incident_details.problem_components': 'problem_components',
        'incident_details.incident_summary': 'incident_summary',
        'manufacturer_inspection.defect_confirmed': 'defect_confirmed',
        'manufacturer_inspection.defect_type': 'defect_type',
        'manufacturer_inspection.inspection_actions': 'inspection_actions'
    })
    
    prompt = '[SYSTEM]\n' + SYSTEM_INSTRUCTION + '\n[USER]' + USER_PROMPT_TEMPLATE

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
        default=100000, # 기본값 설정 
        help='Number of rows to process (default: 10)' # 도움말 메시지
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='random sampling' # 도움말 메시지
    )
    # 3. 명령줄 인자 파싱
    args = parser.parse_args()

    # 4. 파싱된 인자 사용
    print(f"처리할 행 수: {args.n_rows}")
    
    n_rows: int = args.n_rows
    random: bool = args.random

    input_path= DATA_DIR / 'silver' / 'maude_preprocess.parquet'
    result_dir = DATA_DIR / 'silver'
    result_dir = increment_path(result_dir, exist_ok=True, mkdir=True)
    
    sampled_lf = prepare_data(input_path, n_rows=n_rows, random=random)


    extractor = BatchMAUDEExtractor(
        model_path='Qwen/Qwen3-8B',  # 또는 로컬 경로
        tensor_parallel_size=1,  # GPU 개수
        batch_size=64,  # 배치 크기
        max_retries=2
    )
    
    sampled_df = sampled_lf.select(pl.col(['mdr_report_key', 'mdr_text', 'product_problems'])).collect().to_pandas()

    checkpoint_dir = DATA_DIR / 'temp'
    
    # 처리
    start_time = time.time()
    result_df = extractor.process_batch(sampled_df, checkpoint_interval=128, checkpoint_dir=checkpoint_dir)
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
    
    prompt_path = result_dir / f'prompt_{n_rows}_{datetime.now().date()}.txt'
    result_path = result_dir / f'maude_{n_rows}_{datetime.now().date()}.parquet'
    prompt_path = increment_path(prompt_path, exist_ok=False, sep='_')
    result_path = increment_path(result_path, exist_ok=False, sep='_')

    save_data(sampled_lf, result_df, prompt_path, result_path)
    

if __name__=='__main__':
    main()
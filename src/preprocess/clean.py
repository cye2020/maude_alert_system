"""
범용 텍스트 전처리 모듈 (LazyFrame 파이프라인용)
다양한 컬럼 타입에 맞춤형 패턴 적용 가능
"""

import re
from country_named_entity_recognition import find_countries
import polars as pl
import zlib
import shutil
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm, trange
import cleantext as cl

from src.utils import process_lazyframe_in_chunks, apply_mapping_to_columns


class TextPreprocessor:
    """범용 텍스트 전처리 클래스 (Stateless)"""
    
    def __init__(self, name: str = "default", remove_countries: bool = False, custom: bool = True):
        """
        Args:
            name: 전처리기 이름
        """
        self.name = name
        self._patterns = {'delete': [], 'remove': []}
        self._compiled = False
        self.remove_countries = remove_countries
        self.custom = custom
    
    def add_pattern(self, pattern: str, action: str, description: str = ""):
        """
        패턴 추가
        
        Args:
            pattern: 정규표현식 패턴
            action: 'DELETE' 또는 'REMOVE'
            description: 패턴 설명 (선택)
        
        Returns:
            self (메서드 체이닝)
        """
        if action not in ['DELETE', 'REMOVE']:
            raise ValueError("action must be 'DELETE' or 'REMOVE'")
        
        key = action.lower()
        self._patterns[key].append({
            'pattern': pattern,
            'regex': None,  # 나중에 컴파일
            'description': description
        })
        self._compiled = False
        
        return self  # 체이닝
    
    def add_patterns(self, patterns: List[Tuple[str, str, str]]):
        """
        패턴 일괄 추가
        
        Args:
            patterns: [(pattern, action, description), ...]
        
        Returns:
            self (메서드 체이닝)
        """
        for pattern, action, desc in patterns:
            self.add_pattern(pattern, action, desc)
        return self
    
    def _compile(self):
        """패턴 컴파일 (lazy)"""
        if self._compiled:
            return
        
        for key in ['delete', 'remove']:
            for p in self._patterns[key]:
                if p['regex'] is None:
                    p['regex'] = re.compile(p['pattern'], re.IGNORECASE)
        
        self._compiled = True
    
    def clean(self, text: Optional[str]) -> Optional[str]:
        """
        단일 텍스트 클린징
        
        Args:
            text: 입력 텍스트
            
        Returns:
            클린징된 텍스트 (삭제 시 None)
        """
        if text is None or text == '':
            return None

        if not self.custom:
            return cl.clean(text)
        
        self._compile()
        
        # 양옆 공백 제거, 대문자로 통일
        text = str(text).strip().upper()
        
        # REMOVE 패턴 적용
        for p in self._patterns['remove']:
            text = p['regex'].sub('', text).strip()
            if not text:
                return None
        
        # DELETE 패턴 체크
        for p in self._patterns['delete']:
            if p['regex'].match(text):
                return None
            
        # 국가명 제거 (NER 기반, 옵션)
        if self.remove_countries:
            text = self._remove_country_names(text)
            if not text:
                return None
        
        return text if text else None

    def _remove_country_names(self, text: str) -> str:
        """
        NER을 사용해 텍스트에서 국가명 제거
        
        Args:
            text: 입력 텍스트
            
        Returns:
            국가명이 제거된 텍스트
            
        Example:
            >>> self._remove_country_names('ALLERGAN (COSTA RICA)')
            'ALLERGAN'
        """
        if not text:
            return text
        
        try:
            # 국가명 찾기
            countries_found = find_countries(text)
            
            if not countries_found:
                return text
            
            match_country = countries_found[0][1][0]
            regex = re.compile(match_country, re.IGNORECASE)
            text = regex.sub('', text).strip()
            if not text:
                return None
            
            return text
            
        except Exception as e:
            # NER 실패 시 원본 텍스트 반환
            print(f"Warning: Failed to remove country names from '{text}': {e}")
            return text
    
    def create_mapping_dict(self, unique_values: pl.Series) -> Dict[str, Optional[str]]:
        """
        Unique 값 목록에서 매핑 딕셔너리 생성
        
        Args:
            unique_values: Unique 값 Series
            
        Returns:
            {원본: 클린징된값} 딕셔너리
        """
        values = unique_values.to_list()

        mapping = {}
        for val in tqdm(values, desc="Cleaning", total=len(values)):
            mapping[val] = self.clean(val)

        return mapping

    def apply_to_lazyframe(
        self,
        lf: pl.LazyFrame,
        columns: str | list[str],
        output_path: Path,
        chunk_size: int = 1_000_000,
        keep_temp: bool = False
    ):
        """
        LazyFrame에 전처리 적용 (단일 또는 여러 컬럼)
        
        Args:
            lf: 입력 LazyFrame
            columns: 처리할 컬럼명 (단일 str 또는 list[str])
            output_path: 출력 경로
            chunk_size: chunk 크기
            keep_temp: 임시 파일 유지 여부
        """
        # 단일 컬럼이면 리스트로 변환
        if isinstance(columns, str):
            columns = [columns]
        
        # 1. 모든 컬럼의 Unique 값 추출 및 통합
        print(f"[{self.name}] Extracting unique values from {len(columns)} column(s)...")
        
        all_unique_values = set()
        for col in columns:
            unique_vals = lf.select(col).unique().collect()[col]
            all_unique_values.update(unique_vals)
        
        all_unique_values = list(all_unique_values)
        print(f"[{self.name}] Creating mapping for {len(all_unique_values):,} unique values...")
        mapping_dict = self.create_mapping_dict(
            pl.Series(all_unique_values, dtype=pl.Utf8)
        )        
        # 통계
        original_count = len([v for v in mapping_dict.values() if v is not None])
        deleted_count = len(all_unique_values) - original_count
        print(f"  - Kept: {original_count:,} ({original_count/len(all_unique_values)*100:.1f}%)")
        print(f"  - Deleted: {deleted_count:,} ({deleted_count/len(all_unique_values)*100:.1f}%)")
        
        # 2. 변환 함수 정의
        def transform(chunk_lf: pl.LazyFrame) -> pl.LazyFrame:
            return apply_mapping_to_columns(chunk_lf, columns, mapping_dict)
        
        root = Path(output_path).parent
        column_str = "_".join(columns)
        column_hash = zlib.adler32(column_str.encode())
        temp_dir_name = root / f'temp_{self.name}_{column_hash}'

        # 3. Chunk 단위 처리 (utils 함수 사용)
        process_lazyframe_in_chunks(
            lf=lf,
            transform_func=transform,
            output_path=output_path,
            chunk_size=chunk_size,
            temp_dir_name=temp_dir_name,
            keep_temp=keep_temp,
            desc=f"[{self.name}] Processing {len(columns)} column(s)"
        )


# ========== 사전 정의된 패턴셋 ==========

class PreprocessorPresets:
    """사전 정의된 전처리 패턴셋"""
    
    @staticmethod
    def _generic_patterns() -> List[Tuple[str, str, str]]:
        """모든 패턴의 기본이 되는 공통 패턴"""
        return [
            # DELETE 패턴 - 범용 무효값
            (r'.*\bUNKNOWN\b.*', 'DELETE', 'UNKNOWN'),
            (r'.*\bUNKOWN\b.*', 'DELETE', 'UNKOWN (오타)'),
            (r'.*\bUNK\b.*', 'DELETE', 'UNK'),
            (r'.*\bUKN\b.*', 'DELETE', 'UKN (오타)'),
            (r'.*\bNULL\b.*', 'DELETE', 'NULL'),
            (r'.*\bNONE\b.*', 'DELETE', 'NONE'),
            (r'.*\bNIL\b.*', 'DELETE', 'NIL'),
            (r'^N/A$', 'DELETE', 'N/A'), # 정확히 NA
            (r'.*\bN\.A\.?\b.*', 'DELETE', 'N.A'),
            (r'.*\bNI\b.*', 'DELETE', 'NI'),
            (r'^NA$', 'DELETE', 'NA'), # 정확히 NA
            (r'.*\bNOT\s+AVAILABLE\b.*', 'DELETE', 'NOT AVAILABLE'),
            (r'.*\bUNAVAILABLE\b.*', 'DELETE', 'UNAVAILABLE'), 
            (r'.*MISSING.*', 'DELETE', 'MISSING'),
            (r'^NO[\s_]?DATA$', 'DELETE', 'NO DATA'),
            (r'^EMPTY$', 'DELETE', 'EMPTY'),
            (r'.*\bNASK\b.*', 'DELETE', 'NASK'),
            (r'.*\bASKU\b.*', 'DELETE', 'ASKU'),
            (r'.*\bTRC\b.*', 'DELETE', 'TRC'),
            (r'.*\bQS\b.*', 'DELETE', 'QS'),
            (r'.*\bMSK\b.*', 'DELETE', 'MSK'),
            (r'^NAV$', 'DELETE', 'NAV'), # 정확히 NA
            (r'.*\bINV\b.*', 'DELETE', 'INV'),
            (r'.*\bOTH\b.*', 'DELETE', 'OTH'),
            (r'.*\bPINF\b.*', 'DELETE', 'PINF'),
            (r'.*\bNINF\b.*', 'DELETE', 'NINF'),
            (r'.*\bUNC\b.*', 'DELETE', 'UNC'),
            (r'.*\bDER\b.*', 'DELETE', 'DER'),
            (r'^0+$', 'DELETE', '모두 0'),
            (r'^\s*$', 'DELETE', '빈 문자열'),
            
            # REMOVE 패턴 - 범용 공백/기호
            (r'^\s+', 'REMOVE', '시작 공백'),
            (r'\s+$', 'REMOVE', '끝 공백'),
        ]
    
    @staticmethod
    def udi_patterns() -> List[Tuple[str, str, str]]:
        """UDI-DI 전용 패턴 = Generic + UDI 특화"""
        patterns = PreprocessorPresets._generic_patterns()
        
        # UDI 특화 DELETE 패턴만 추가
        udi_deletes = [
            (r'^.{2,3}$', 'DELETE', '2-3자'),  # 1자는 generic에 있으므로 2-3자만
            (r'^.{31,}$', 'DELETE', '31자 이상'),
            (r'^N$', 'DELETE', '단일 문자 N'),
            (r'^X+$', 'DELETE', 'X 반복'),
            (r'.*\$\$.*', 'DELETE', '$$'),
            (r'.*[@#%^&*]{2,}.*', 'DELETE', '연속 특수문자'),
        ]
        
        # UDI 특화 REMOVE 패턴
        udi_removes = [
            (r'\(\d{2}\)', 'REMOVE', 'GS1 AI'),
            (r'^\d*\+', 'REMOVE', 'HIBCC 접두어'),
            (r'\s+', 'REMOVE', '공백'),
            (r'[^a-zA-Z0-9]', 'REMOVE', '영숫자 외 모든 문자'),
        ]
        
        return patterns + udi_deletes + udi_removes
    
    @staticmethod
    def company_name_patterns() -> List[Tuple[str, str, str]]:
        """회사명 전용 패턴 = Generic + 회사명 특화"""
        patterns = PreprocessorPresets._generic_patterns()
        
        # 회사명 특화 REMOVE 패턴 - 법인 형태 제거
        company_removes = [
            (r'\b(INC\.?|INCORPORATED)\b', 'REMOVE', 'INC'),
            (r'\b(OPERATIONS)\b', 'REMOVE', 'OPERATIONS'),
            (r'\b(CO\.?|COMPANY)\b', 'REMOVE', 'CO'),
            (r'\b(CORP\.?|CORPORATION)\b', 'REMOVE', 'CORP'),
            (r'\b(LTD\.?|LIMITED)\b', 'REMOVE', 'LTD'),
            (r'\b(LLC\.?)\b', 'REMOVE', 'LLC'),
            (r'\b(L\.L\.C\.?)\b', 'REMOVE', 'L.L.C'),
            (r'\b(PLC\.?)\b', 'REMOVE', 'PLC'),
            (r'\b(S\.A\.?)\b', 'REMOVE', 'S.A'),
            (r'\b(GMBH\.?)\b', 'REMOVE', 'GMBH'),
            (r'\b(AG\.?)\b', 'REMOVE', 'AG'),
            (r'\b(PTY\.?)\b', 'REMOVE', 'PTY'),
            (r'\b(PVT\.?)\b', 'REMOVE', 'PVT'),
            (r'\b(THE)\b', 'REMOVE', 'THE'),
            (r'[^a-zA-Z0-9\s&]', 'REMOVE', '허용된 문자 외 제거'),
        ]
        
        return patterns + company_removes
    
    @staticmethod
    def generic_text_patterns() -> List[Tuple[str, str, str]]:
        """일반 텍스트 클린징 패턴 = Generic + 텍스트 특화"""
        patterns = PreprocessorPresets._generic_patterns()
        
        # 텍스트 특화 DELETE 패턴 (도메인별 특수 코드)
        text_deletes = [
            (r'^.{,5}$', 'DELETE', '5자 이하'), 
            (r'^0+$', 'DELETE', '모두 0'),
            (r'^\s*$', 'DELETE', '빈 문자열'),
        ]
        
        # 텍스트 특화 REMOVE 패턴
        text_removes = [
            (r'[,;]+', 'REMOVE', '구분자'),
            # REMOVE 패턴 - 범용 공백/기호
            (r'^\s+', 'REMOVE', '시작 공백'),
            (r'\s+$', 'REMOVE', '끝 공백'),
        ]
        
        return patterns + text_deletes + text_removes
    
    @staticmethod
    def email_patterns() -> List[Tuple[str, str, str]]:
        """이메일 클린징 패턴 = Generic + 이메일 특화"""
        patterns = PreprocessorPresets._generic_patterns()
        
        # 이메일 특화 DELETE 패턴
        email_deletes = [
            (r'^[^@]+$', 'DELETE', '@ 없음'),
            (r'.*@example\.com$', 'DELETE', 'example.com'),
            (r'.*@test\.com$', 'DELETE', 'test.com'),
        ]
        
        return patterns + email_deletes

# ========== 팩토리 함수 ==========

def create_udi_preprocessor() -> TextPreprocessor:
    """UDI 전처리기 생성"""
    return TextPreprocessor("UDI").add_patterns(
        PreprocessorPresets.udi_patterns()
    )

def create_company_preprocessor() -> TextPreprocessor:
    """회사명 전처리기 생성"""
    return TextPreprocessor("CompanyName", remove_countries=True).add_patterns(
        PreprocessorPresets.company_name_patterns()
    )

def create_number_preprocessor() -> TextPreprocessor:
    """일반 숫자 전처리기 생성"""
    return TextPreprocessor("GenericNumber").add_patterns(
        PreprocessorPresets.generic_text_patterns()
    )
    
def create_generic_preprocessor() -> TextPreprocessor:
    """일반 텍스트 전처리기 생성"""
    return TextPreprocessor("GenericText", custom=False)

def create_email_preprocessor() -> TextPreprocessor:
    """이메일 전처리기 생성"""
    return TextPreprocessor("Email").add_patterns(
        PreprocessorPresets.email_patterns()
    )


# ========== 사용 예시 ==========

if __name__ == "__main__":
    from pathlib import Path
    
    print("="*70)
    print("범용 텍스트 전처리 테스트")
    print("="*70)
    
    # ===== 예시 1: UDI 전처리 =====
    print("\n[예시 1] UDI 전처리")
    udi_preprocessor = create_udi_preprocessor()
    
    udi_tests = ['UNKNOWN', '0+M724CCB541', '00012345678901', 'NULL', '  123  ', '00000000000000']
    for udi in udi_tests:
        cleaned = udi_preprocessor.clean(udi)
        print(f"  {udi:<30} → {cleaned}")
    
    # ===== 예시 2: 회사명 전처리 =====
    print("\n[예시 2] 회사명 전처리")
    company_preprocessor = create_company_preprocessor()
    
    company_tests = [
        'Apple Inc.',
        'Google LLC',
        'Samsung Electronics Co., Ltd.',
        'Microsoft Corporation',
        'The Coca-Cola Company',
        'A & B Company',
        "'POLARSTEM¿'",
        'JOHNSON & JOHNSON SURGICAL VISION',
        'UNKNOWN',
        'HEARTMATE®, MOBILE POWER UNIT, NA'
    ]
    for company in company_tests:
        cleaned = company_preprocessor.clean(company)
        print(f"  {company:<40} → {cleaned}")
    
    # ===== 예시 3: 커스텀 패턴 추가 =====
    print("\n[예시 3] 커스텀 패턴 추가")
    custom_preprocessor = (
        TextPreprocessor("Custom")
        .add_pattern(r'TEST\d+', 'DELETE', '테스트 데이터')
        .add_pattern(r'\bDEMO\b', 'REMOVE', 'DEMO 제거')
        .add_patterns(PreprocessorPresets.generic_text_patterns())
    )
    
    custom_tests = ['TEST123', 'DEMO Product', 'Normal Data', 'UNKNOWN']
    for text in custom_tests:
        cleaned = custom_preprocessor.clean(text)
        print(f"  {text:<30} → {cleaned}")
    
    # ===== 예시 4: LazyFrame 적용 =====
    print("\n[예시 4] LazyFrame 적용")
    
    # 가상 데이터
    data = {
        'id': range(100),
        'udi': ['00012345678901', 'UNKNOWN', '0+M724', None] * 25,
        'company': ['Apple Inc.', 'Google LLC', 'UNKNOWN', 'Samsung Co.'] * 25, 
        'brand': ['\'POLARSTEM¿\'', 'Google LLC', 'N/A', 'HEARTMATE®, MOBILE POWER UNIT, NA'] * 25
    }
    lf = pl.LazyFrame(data)
    
    # UDI 컬럼 전처리
    udi_output = Path('cleaned_udi.parquet')
    udi_preprocessor.apply_to_lazyframe(
        lf=lf,
        columns='udi',
        output_path=udi_output,
        chunk_size=50
    )
    
    # 회사명 컬럼 전처리
    root = Path(__file__).parent
    company_output = root / Path('cleaned_company.parquet')
    
    # 이전 결과를 다시 로드해서 처리
    lf_cleaned = pl.scan_parquet(udi_output)
    company_preprocessor.apply_to_lazyframe(
        lf=lf_cleaned,
        columns=['company', 'brand'],
        output_path=company_output,
        chunk_size=50
    )
    
    # 결과 확인
    print("\n최종 결과:")
    print(pl.scan_parquet(company_output).head(10).collect())
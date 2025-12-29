"""
전처리 모듈

⚠️ preprocess.py는 deprecated되었습니다.
대신 다음 모듈들을 사용하세요:
- src.preprocess.eda: 범용 EDA 함수들
- src.preprocess.domain_eda: MAUDE 도메인 특화 EDA 함수들
- src.preprocess.transforms: 데이터 변환 함수들
- src.preprocess.udi: UDI 처리 함수들
- src.preprocess.mdr: MDR 처리 함수들
- src.utils.polars: Polars 범용 유틸리티
- src.utils.visualization: 시각화 함수들
"""

from src.preprocess.clean import (
    TextPreprocessor,
    create_udi_preprocessor,
    create_company_preprocessor,
    create_generic_preprocessor,
    create_number_preprocessor
)

# 새로운 모듈들 (명시적으로 import 필요)
# from src.preprocess.eda import *
# from src.preprocess.domain_eda import *
# from src.preprocess.transforms import *
# from src.preprocess.udi import *
# from src.preprocess.mdr import *

__all__ = [
    'TextPreprocessor',
    'create_udi_preprocessor',
    'create_company_preprocessor',
    'create_generic_preprocessor',
    'create_number_preprocessor'
]
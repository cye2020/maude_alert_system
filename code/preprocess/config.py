"""
설정 및 상수
"""
from dataclasses import dataclass
from pathlib import Path
# 상대 경로 사용
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
print(PROJECT_ROOT)
@dataclass
class Config:
    """UDI 처리 설정"""
    
    # 퍼지 매칭 임계값
    FUZZY_THRESHOLD: int = 90
    
    # 저준수 제조사 기준
    LOW_COMPLIANCE_THRESHOLD: float = 0.50
    
    # 신뢰도 매핑
    CONFIDENCE_MAP = {
        'original': 'HIGH',
        'extracted': 'HIGH',
        'single_match': 'HIGH',
        'time_inferred': 'MEDIUM',
        'freq_inferred': 'LOW',
        'fallback_oldest': 'LOW',
        'tier3': 'VERY_LOW'
    }
    
    # MAUDE 날짜 우선순위
    MAUDE_DATES = [
        'date_of_event',
        'date_received',
        'date_report',
        'device_0_date_received'
    ]
    
    # UDI DB 날짜
    UDI_DATES = ['publish_date']
    
    FULL_GROUP = [
        'manufacturer',
        'brand',
        'model_number',
        'catalog_number',
    ]
    
    JOIN_COL = 'udi_di'
    
    TEMP_DIR = DATA_DIR / Path("data/_temp")
    CLEANUP_TEMP_ON_SUCCESS = True
    CLEANUP_TEMP_ON_ERROR = False   # 디버깅용
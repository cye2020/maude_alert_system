# ==================================================
# json 파일 평탄화
# ==================================================

# -----------------------------
# 표준 라이브러리
# -----------------------------
from typing import Set, List, Dict, Any


class Flattener:
    """레코드 평탄화 및 정규화"""
    
    def __init__(self, sep: str = '_'):
        self.sep = sep
    
    def clean_empty_arrays(self, obj: Any) -> Any:
        """빈 값 정리"""
        if isinstance(obj, dict):
            return {k: self.clean_empty_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if obj == [""]:
                return None
            return [self.clean_empty_arrays(item) for item in obj]
        elif obj == "":
            return None
        return obj
    
    def flatten_dict(self, nested_dict: Dict, parent_key: str = '') -> Dict:
        """중첩된 딕셔너리를 평탄화"""
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{self.sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    items.extend(self.flatten_dict(item, f"{new_key}_{i}").items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def extract_columns(self, record: Dict) -> Set[str]:
        """레코드에서 컬럼명 추출"""
        cleaned = self.clean_empty_arrays(record)
        flattened = self.flatten_dict(cleaned)
        return set(flattened.keys())
    
    def normalize(self, record: Dict, schema_columns: List[str]) -> Dict:
        """스키마에 맞춰 레코드 정규화"""
        cleaned = self.clean_empty_arrays(record)
        flattened = self.flatten_dict(cleaned)
        
        normalized = {}
        for col in schema_columns:
            val = flattened.get(col, None)
            normalized[col] = str(val) if val is not None else None
        return normalized
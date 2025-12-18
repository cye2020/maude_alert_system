import sys
import uuid
from pathlib import Path

def is_running_in_notebook():
    """
    현재 코드가 Jupyter 노트북 또는 IPython 환경에서 실행 중인지 확인합니다.
    """
    # 'ipykernel'이 sys.modules에 있으면 IPython/Jupyter 환경입니다.
    if 'ipykernel' in sys.modules:
        return True
    
    # __file__이 정의되어 있지 않으면 스크립트가 아닌 대화형 세션일 가능성이 높습니다.
    try:
        __file__
        return False # __file__이 있으면 스크립트 파일입니다.
    except NameError:
        return True # __file__이 없으면 노트북/대화형 세션입니다.


UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

def uuid5_from_str(x: str | None) -> str | None:
    if x is None:
        return None
    return str(uuid.uuid5(UUID_NAMESPACE, x))

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    YOLO 방식 경로 증가 함수
    예: runs/detect/exp -> runs/detect/exp2, exp3 ...
    """
    path = Path(path)

    if path.exists() and not exist_ok:
        base = path.stem
        suffix = path.suffix
        parent = path.parent

        i = 2
        while True:
            new_path = parent / f"{base}{sep}{i}{suffix}"
            if not new_path.exists():
                path = new_path
                break
            i += 1

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path



if __name__=='__main__':
    print(is_running_in_notebook())
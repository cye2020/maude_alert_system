import sys
import uuid
from pathlib import Path
import os

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

def increment_path(path: str | Path, exist_ok: bool = False, sep: str = "", mkdir: bool = False) -> Path:
    """Increment a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to the
    end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the number
    will be appended directly to the end of the path.

    Args:
        path (str | Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is.
        sep (str, optional): Separator to use between the path and the incrementation number.
        mkdir (bool, optional): Create a directory if it does not exist.

    Returns:
        (Path): Incremented path.

    Examples:
        Increment a directory path:
        >>> from pathlib import Path
        >>> path = Path("runs/exp")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp2

        Increment a file path:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp/results2.txt
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path



if __name__=='__main__':
    print(is_running_in_notebook())
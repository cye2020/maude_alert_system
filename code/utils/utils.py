import sys

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

if __name__=='__main__':
    print(is_running_in_notebook())
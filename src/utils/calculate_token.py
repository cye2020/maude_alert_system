import tiktoken
from google import genai
from dotenv import load_dotenv
import os
from typing import Union, List, Optional, Dict, Any
from tqdm import tqdm
import numpy as np

load_dotenv()

# API 키 설정이 필요할 수 있습니다 (환경 변수 권장)
# client = genai.Client(api_key="YOUR_API_KEY")
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

def count_tokens_gemini(model_name: str, text: str) -> int:
    """
    Gemini API를 사용하여 텍스트의 토큰 수를 계산합니다.

    Args:
        model_name: Gemini 모델 이름 (예: "gemini-2.5-flash-light")
        text: 토큰 수를 계산할 텍스트

    Returns:
        int: 토큰 수
    """
    if client is None:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")

    response = client.models.count_tokens(
        model=model_name,
        contents=[{"role": "user", "parts": [{"text": text}]}]
    )
    return response.total_tokens


def count_tokens_openai(model_name: str, text: str) -> int:
    """
    OpenAI 모델의 토크나이저를 사용하여 텍스트의 토큰 수를 계산합니다.

    Args:
        model_name: OpenAI 모델 이름 (예: "gpt-4", "gpt-3.5-turbo")
        text: 토큰 수를 계산할 텍스트

    Returns:
        int: 토큰 수
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)


def count_tokens_huggingface(tokenizer, text: str) -> int:
    """
    HuggingFace 토크나이저를 사용하여 텍스트의 토큰 수를 계산합니다.

    Args:
        tokenizer: HuggingFace AutoTokenizer 인스턴스
        text: 토큰 수를 계산할 텍스트

    Returns:
        int: 토큰 수
    """
    return len(tokenizer.encode(text))


def count_tokens(
    texts: Union[str, List[str]],
    tokenizer_type: str = "huggingface",
    model_name: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 1000,
    show_progress: bool = True,
    return_stats: bool = False,
    max_tokens: Optional[int] = None
) -> Union[int, List[int], Dict[str, Any]]:
    """
    유연한 토큰 계산 함수. 단일 텍스트 또는 텍스트 리스트에 대해 토큰 수를 계산합니다.

    Args:
        texts: 단일 텍스트 또는 텍스트 리스트
        tokenizer_type: 토크나이저 타입 ("huggingface", "openai", "gemini")
        model_name: 모델 이름 (openai, gemini의 경우 필수)
        tokenizer: HuggingFace 토크나이저 인스턴스 (tokenizer_type="huggingface"인 경우 필수)
        batch_size: 배치 크기 (리스트 처리 시)
        show_progress: 진행 상황 표시 여부
        return_stats: 통계 정보 반환 여부 (리스트 처리 시만 유효)
        max_tokens: 최대 토큰 수 (통계 계산 시 사용)

    Returns:
        - 단일 텍스트: int (토큰 수)
        - 텍스트 리스트 + return_stats=False: List[int] (각 텍스트의 토큰 수)
        - 텍스트 리스트 + return_stats=True: Dict (토큰 수 리스트 + 통계)

    Examples:
        >>> # HuggingFace 토크나이저 사용 (단일 텍스트)
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/Qwen3-32B-NVFP4")
        >>> count = count_tokens("Hello world", tokenizer_type="huggingface", tokenizer=tokenizer)

        >>> # HuggingFace 토크나이저 사용 (리스트, 통계 포함)
        >>> texts = ["Hello", "World", "Test"]
        >>> result = count_tokens(
        ...     texts,
        ...     tokenizer_type="huggingface",
        ...     tokenizer=tokenizer,
        ...     return_stats=True,
        ...     max_tokens=1000
        ... )
        >>> print(result['stats'])

        >>> # OpenAI 토크나이저 사용
        >>> count = count_tokens("Hello world", tokenizer_type="openai", model_name="gpt-4")

        >>> # Gemini API 사용
        >>> count = count_tokens("Hello world", tokenizer_type="gemini", model_name="gemini-2.5-flash-light")
    """
    # 단일 텍스트 처리
    if isinstance(texts, str):
        if tokenizer_type == "huggingface":
            if tokenizer is None:
                raise ValueError("tokenizer_type='huggingface'인 경우 tokenizer가 필요합니다.")
            return count_tokens_huggingface(tokenizer, texts)
        elif tokenizer_type == "openai":
            if model_name is None:
                raise ValueError("tokenizer_type='openai'인 경우 model_name이 필요합니다.")
            return count_tokens_openai(model_name, texts)
        elif tokenizer_type == "gemini":
            if model_name is None:
                raise ValueError("tokenizer_type='gemini'인 경우 model_name이 필요합니다.")
            return count_tokens_gemini(model_name, texts)
        else:
            raise ValueError(f"지원하지 않는 tokenizer_type: {tokenizer_type}")

    # 리스트 처리
    token_counts = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(0, len(texts), batch_size)

    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc="Counting tokens")

    for i in iterator:
        batch = texts[i:i+batch_size]

        if tokenizer_type == "huggingface":
            if tokenizer is None:
                raise ValueError("tokenizer_type='huggingface'인 경우 tokenizer가 필요합니다.")
            # 배치 처리 최적화
            batch_counts = [len(ids) for ids in tokenizer(batch)["input_ids"]]
            token_counts.extend(batch_counts)
        elif tokenizer_type == "openai":
            if model_name is None:
                raise ValueError("tokenizer_type='openai'인 경우 model_name이 필요합니다.")
            batch_counts = [count_tokens_openai(model_name, text) for text in batch]
            token_counts.extend(batch_counts)
        elif tokenizer_type == "gemini":
            if model_name is None:
                raise ValueError("tokenizer_type='gemini'인 경우 model_name이 필요합니다.")
            batch_counts = [count_tokens_gemini(model_name, text) for text in batch]
            token_counts.extend(batch_counts)
        else:
            raise ValueError(f"지원하지 않는 tokenizer_type: {tokenizer_type}")

    if not return_stats:
        return token_counts

    # 통계 계산
    token_counts_array = np.array(token_counts)
    stats = {
        "max": int(np.max(token_counts_array)),
        "min": int(np.min(token_counts_array)),
        "mean": float(np.mean(token_counts_array)),
        "median": float(np.median(token_counts_array)),
        "std": float(np.std(token_counts_array)),
        "total": len(token_counts),
    }

    if max_tokens is not None:
        stats["over_limit"] = int(np.sum(token_counts_array > max_tokens))
        stats["over_limit_ratio"] = float(stats["over_limit"] / len(token_counts))

    return {
        "token_counts": token_counts,
        "stats": stats
    }

if __name__ == '__main__':
    text_to_count = "Counting tokens for Gemini models in Python."
    model = "" # 또는 "gpt-3.5-turbo", "text-embedding-ada-002" 등
    token_count = count_tokens_openai(model, text_to_count)

    print(f"'{text_to_count}'의 토큰 수 ({model}): {token_count}")
    
    text_to_count_gemini = "Counting tokens for Gemini models in Python."
    model_gemini = "gemini-2.5-flash-light" 
    token_count_gemini = count_tokens_gemini(model_gemini, text_to_count_gemini)

    print(f"'{text_to_count_gemini}'의 토큰 수 ({model_gemini}): {token_count_gemini}")

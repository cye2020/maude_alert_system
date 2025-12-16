import tiktoken
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# API 키 설정이 필요할 수 있습니다 (환경 변수 권장)
# client = genai.Client(api_key="YOUR_API_KEY") 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

def count_tokens_gemini(model_name: str, text: str) -> int:
    """
    Gemini API를 사용하여 텍스트의 토큰 수를 계산합니다.
    """
    # count_tokens 메소드를 사용합니다.
    response = client.models.count_tokens(
        model=model_name,
        contents=[{"role": "user", "parts": [{"text": text}]}]
    )
    return response.total_tokens


def count_tokens_openai(model_name: str, text: str) -> int:
    """
    OpenAI 모델의 토크나이저를 사용하여 텍스트의 토큰 수를 계산합니다.
    """
    try:
        # 모델 이름에 맞는 인코딩을 가져옵니다.
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # 특정 모델 이름이 tiktoken에 없는 경우 기본값으로 대체할 수 있습니다.
        # 예: gpt-3.5-turbo는 cl100k_base 인코딩을 사용합니다.
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # 텍스트를 토큰 ID 리스트로 인코딩하고 길이를 반환합니다.
    tokens = encoding.encode(text)
    return len(tokens)

if __name__ == '__main__':
    text_to_count = "Counting tokens for Gemini models in Python."
    model = "" # 또는 "gpt-3.5-turbo", "text-embedding-ada-002" 등
    token_count = count_tokens_openai(model, text_to_count)

    print(f"'{text_to_count}'의 토큰 수 ({model}): {token_count}")
    
    text_to_count_gemini = "Counting tokens for Gemini models in Python."
    model_gemini = "gemini-2.5-flash-light" 
    token_count_gemini = count_tokens_gemini(model_gemini, text_to_count_gemini)

    print(f"'{text_to_count_gemini}'의 토큰 수 ({model_gemini}): {token_count_gemini}")

from typing import List, Union
import requests
import zipfile
import io
import orjson
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

SEARCH_URL = 'https://api.fda.gov/download.json'


def search_download_url(start: int, end: int) -> List[str]:
    response = requests.get(SEARCH_URL).json()
    partitions = response['results']['device']['event']['partitions']

    urls = []
    for item in partitions:
        first = item['display_name'].split()[0]
        if first.isdigit():
            year = int(first)
            if start <= year <= end:
                urls.append(item["file"])
    return urls


def download_to_bytes(url: str) -> tuple[bytes, float]:
    start = time.time()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        buf = io.BytesIO()

        for chunk in tqdm(
            r.iter_content(chunk_size=4 * 1024 * 1024),
            total=total // (4 * 1024 * 1024),
            unit="chunk",
            desc=f"다운로드 {url.split('/')[-1]}",
            leave=False
        ):
            buf.write(chunk)

        download_time = time.time() - start
        return buf.getvalue(), download_time


def unzip_and_parse(data: bytes) -> tuple[dict, float]:
    start = time.time()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        json_file = [n for n in z.namelist() if n.endswith(".json")][0]
        result = orjson.loads(z.read(json_file))
    unzip_time = time.time() - start
    return result, unzip_time


def process_url(url: str) -> tuple[dict, float, float]:
    data, download_time = download_to_bytes(url)
    result, unzip_time = unzip_and_parse(data)
    return result, download_time, unzip_time


def collect_json_files_parallel(urls: List[str], max_workers=8) -> dict:
    all_results = []  # 모든 결과를 저장할 리스트
    total_download_time = 0
    total_unzip_time = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_url, url): url for url in urls}

        for future in tqdm(as_completed(futures), total=len(urls), desc="병렬 처리"):
            result, download_time, unzip_time = future.result()
            
            # 'results' 키가 있는 경우 해당 리스트를 확장
            if 'results' in result:
                all_results.extend(result['results'])
            else:
                # 다른 구조인 경우 전체 결과 추가
                all_results.append(result)
            
            total_download_time += download_time
            total_unzip_time += unzip_time

    print(f"\n총 다운로드 시간: {total_download_time:.2f}초")
    print(f"총 압축 해제 시간: {total_unzip_time:.2f}초")
    print(f"총 레코드 수: {len(all_results)}")
    return all_results


def search_and_collect_json(start: Union[str, int], end: Union[str, int], max_workers=4):
    start_time = time.time()
    urls = search_download_url(start, end)
    print(f"찾은 URL 개수: {len(urls)}")
    collection = collect_json_files_parallel(urls, max_workers=max_workers)
    total_time = time.time() - start_time
    print(f"전체 실행 시간: {total_time:.2f}초")
    return collection


if __name__ == '__main__':
    start_time = time.time()
    
    urls = search_download_url(2024, 2024)
    print(f"찾은 URL 개수: {len(urls)}")
    
    collection = collect_json_files_parallel(urls, max_workers=4)
    
    total_time = time.time() - start_time
    print(f"전체 실행 시간: {total_time:.2f}초")
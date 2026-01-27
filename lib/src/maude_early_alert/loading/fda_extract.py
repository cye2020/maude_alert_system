# ======================
# 표준 라이브러리
# ======================
import re
from typing import Dict, List

# ======================
# 서드파티 라이브러리
# ======================
import requests
import structlog


class FDAExtractor:
    """FDA Open API에서 device 메타데이터를 조회하고 파일 목록을 추출"""

    SUPPORTED_CATEGORIES = ('udi', 'event')

    @staticmethod
    def _filter_event(url: str, start: int = None, end: int = None) -> bool:
        """event 카테고리: URL에서 연도를 추출하여 범위 필터링"""
        match = re.search(r'/(\d{4})q\d+/', url)
        if not match:
            return False
        year = int(match.group(1))
        if start and year < start:
            return False
        if end and year > end:
            return False
        return True

    def __init__(self, session: requests.Session = None):
        """Args:
            session: HTTP 세션 (미지정 시 기본 Session 사용)
        """
        self.url = 'https://api.fda.gov/download.json'
        self.session = session or requests.Session()
        self.logger = structlog.get_logger(__name__)
        self.metadata = None

    def fetch_metadata(self) -> Dict:
        """FDA API 메타데이터 조회 (캐싱됨)

        Raises:
            requests.HTTPError: API 요청 실패 시
        """
        if self.metadata is None:
            response = self.session.get(self.url)
            response.raise_for_status()
            self.metadata = response.json()
        return self.metadata

    def extract(
        self, category: str,
        start: int = None, end: int = None
    ) -> List[Dict[str, str]]:
        """카테고리별 파일 정보(url, display_name) 추출

        Args:
            category: 데이터 카테고리 ('udi' 또는 'event')
            start: 시작 연도 (event만 적용)
            end: 종료 연도 (event만 적용)

        Raises:
            ValueError: 지원하지 않는 카테고리
        """
        if category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        metadata = self.fetch_metadata()
        device_data = metadata['results']['device']

        if category not in device_data or 'partitions' not in device_data[category]:
            return []

        # 카테고리별 필터 함수 탐색 (convention: _filter_{category})
        filter_fn = getattr(self, f'_filter_{category}', None)

        files = []
        for partition in device_data[category]['partitions']:
            url = partition.get('file', '')

            if f'device/{category}/' not in url:
                continue

            if filter_fn and not filter_fn(url, start, end):
                continue

            files.append({
                'url': url,
                'display_name': partition.get('display_name'),
            })

        return files


if __name__ == '__main__':
    extractor = FDAExtractor()

    import pendulum
    files = extractor.extract('event', start=2020, end=pendulum.now().year)
    print(f"총 {len(files)}개 파일 추출")
    for f in files:
        print(f"  {f['display_name']}: {f['url']}")

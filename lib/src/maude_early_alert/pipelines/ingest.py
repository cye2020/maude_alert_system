from typing import List
import requests
import structlog
import pendulum

from maude_early_alert.loaders.s3_load import S3Loader
from maude_early_alert.loaders.fda_extract import FDAExtractor
from maude_early_alert.pipelines.config import get_config

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class IngestPipeline:
    def __init__(self, logical_date: pendulum.DateTime):
        self.cfg = get_config().bronze
        self.logical_date = logical_date

    def extract(self, session: requests.Session) -> List[str]:
        url = self.cfg.get_extract_url()
        period = self.cfg.get_extract_period()
        categories = self.cfg.get_extract_categories()

        end = self.logical_date.year
        start = end - period + 1

        data_list = []

        extractor = FDAExtractor(url, session)

        for category in categories:
            extracted_list = extractor.extract(category, start, end)

            data_list.extend(extracted_list)

        return data_list

    def s3_load(self, files: List[str], client, session: requests.Session):
        if not self.cfg.get_s3_enabled():
            logger.warning('S3 로드 비활성화 상태, 건너뜀')
            return

        bucket_name = self.cfg.get_s3_bucket_name()

        s3_loader = S3Loader(bucket_name=bucket_name, client=client, session=session)

        ym = self.logical_date.strftime('%Y%m')
        for file in files:
            s3_key = s3_loader.s3_key_generate(file, ym)
            result = s3_loader.load(s3_key, file)
            logger.info('Load Result', s3_key=s3_key, result=result)


if __name__ == '__main__':
    from maude_early_alert.logging_config import configure_logging

    configure_logging(level='DEBUG', log_file='temp.log')

    logical_date = pendulum.now()

    pipeline = IngestPipeline(logical_date)

    data_urls = []
    with requests.Session() as session:
        data_urls = pipeline.extract(session)
        logger.debug('추출 성공', data_urls=data_urls)

    import boto3

    client = boto3.client('s3')
    with requests.Session() as session:
        pipeline.s3_load(data_urls, client, session)
    client.close()

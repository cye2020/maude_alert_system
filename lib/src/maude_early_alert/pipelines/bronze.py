import requests
import structlog
import pendulum

from maude_early_alert.pipelines.config import get_config
from maude_early_alert.loaders.fda_extract import FDAExtractor

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class BronzePipeline:
    def __init__(self, database: str, schema: str):
        self.cfg = get_config().bronze
        self.database = database
        self.schema = schema
    
    def extract(self, session: requests.Session):
        url = self.cfg.get_extract_url()
        period = self.cfg.get_extract_period()
        categories = self.cfg.get_extract_categories()
        
        end = pendulum.now().year
        start = end - period + 1
        
        data_list = []
        
        extractor = FDAExtractor(url, session)
        
        for category in categories:
            extracted_list = extractor.extract(category, start, end)
            
            data_list.extend(extracted_list)
        
        return data_list
    
    def load():
        pass


if __name__ == '__main__':
    from maude_early_alert.logging_config import configure_logging
    
    configure_logging(level='DEBUG', log_file='temp.log')
    
    database = 'MAUDE'
    schema = 'BRONZE'
    pipeline = BronzePipeline(database, schema)
    
    with requests.Session() as session:
        data_urls = pipeline.extract(session)
        logger.debug('추출 성공', data_urls=data_urls)
    
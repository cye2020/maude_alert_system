from airflow.sdk import Asset
from maude_early_alert.utils.config_loader import load_config

_storage = load_config('storage')

_s3_bucket = _storage['s3']['bucket_name']
_sf_load_db = _storage['snowflake']['load']['database']
_sf_load_schema = _storage['snowflake']['load']['schema']
_sf_load_tables = list(_storage['snowflake']['load']['tables'].keys())

_sf_transform_db = _storage['snowflake']['transform']['database']
_sf_transform_schema = _storage['snowflake']['transform']['schema']
_sf_transform_tables = list(_storage['snowflake']['transform']['tables'].keys())

# ingest_dag (s3_load) → bronze_dag 트리거
MAUDE_S3_ASSET = Asset(f's3://{_s3_bucket}')

# bronze_dag (load_all) → silver_dag 트리거
MAUDE_BRONZE_ASSETS = [
    Asset(f'snowflake://{_sf_load_db}/{_sf_load_schema}/{table}')
    for table in _sf_load_tables
]

# silver_dag (preprocess) → llm_dag 트리거
MAUDE_SILVER_ASSETS = [
    Asset(f'snowflake://{_sf_transform_db}/{_sf_transform_schema}/{table}')
    for table in _sf_transform_tables
]

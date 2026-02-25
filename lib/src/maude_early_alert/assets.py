from airflow.sdk import Asset
from maude_early_alert.utils.config_loader import load_config

_storage = load_config('storage')

_s3_bucket = _storage['s3']['bucket_name']
_sf_database = _storage['snowflake']['load']['database']
_sf_schema = _storage['snowflake']['load']['schema']
_sf_tables = list(_storage['snowflake']['load']['tables'].keys())

# ingest_dag (s3_load) → bronze_dag 트리거
MAUDE_S3_ASSET = Asset(f's3://{_s3_bucket}')

# bronze_dag (load_all) → silver_dag 트리거
MAUDE_BRONZE_ASSETS = [
    Asset(f'snowflake://{_sf_database}/{_sf_schema}/{table}')
    for table in _sf_tables
]

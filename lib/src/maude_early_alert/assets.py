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

# llm_dag (join_extraction) → cluster_dag 트리거
_llm_cfg = load_config('preprocess/llm_extraction')
_llm_category = _llm_cfg['source']['category'].upper()
_llm_join_suffix = _llm_cfg['extracted']['join_suffix'].upper()
MAUDE_LLM_ASSET = Asset(
    f'snowflake://{_sf_transform_db}/{_sf_transform_schema}/{_llm_category}{_llm_join_suffix}'
)

# clustering_dag (clustering) → gold_dag 트리거
_clustering_cfg = load_config('preprocess/clustering')
_clustering_category = _clustering_cfg['source']['category'].upper()
_clustering_output_suffix = _clustering_cfg['output']['suffix'].upper()
MAUDE_CLUSTERED_ASSET = Asset(
    f'snowflake://{_sf_transform_db}/{_sf_transform_schema}/{_clustering_category}{_clustering_output_suffix}'
)

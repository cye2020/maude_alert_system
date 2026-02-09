CREATE OR REPLACE FUNCTION fuzzy_match_score(source VARCHAR, target VARCHAR)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
PACKAGES = ('rapidfuzz')
HANDLER = 'calculate_score'
AS
$$
from rapidfuzz import fuzz

def calculate_score(source, target):
    if source is None or target is None:
        return 0.0
    return fuzz.ratio(source.upper(), target.upper()) / 100.0
$$
;

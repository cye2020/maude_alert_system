CREATE OR REPLACE FUNCTION remove_country_names_udf(input_text VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
PACKAGES = ('country-named-entity-recognition')
HANDLER = 'remove_countries'
AS
$$
import re
from country_named_entity_recognition import find_countries

def remove_countries(input_text):
    if not input_text or str(input_text).strip() == '':
        return None
    try:
        countries_found = find_countries(input_text)
        if not countries_found:
            return input_text
        match_country = countries_found[0][1][0]
        regex = re.compile(match_country, re.IGNORECASE)
        result = regex.sub('', input_text).strip()
        return result if result else None
    except Exception:
        return input_text
$$;
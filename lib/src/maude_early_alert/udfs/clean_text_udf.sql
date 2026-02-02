CREATE OR REPLACE FUNCTION clean_text_udf(input_text VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
PACKAGES = ('clean-text')
HANDLER = 'clean_text'
AS
$$
import cleantext as cl

def clean_text(input_text):
    if input_text is None or str(input_text).strip() == '':
        return None
    result = cl.clean(input_text, lower=False)
    return result if result else None
$$
;
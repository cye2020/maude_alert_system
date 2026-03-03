# logging_config.py
import logging
import structlog

_SUPPRESS = [
    'snowflake.connector',
    'urllib3',
    'botocore',
    'boto3',
]

def configure_logging(level: str = 'INFO', log_file: str | None = None,):
    log_level = getattr(logging, level.upper())
    logging.basicConfig(filename=log_file, level=log_level)

    for name in _SUPPRESS:
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

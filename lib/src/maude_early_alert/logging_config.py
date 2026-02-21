# logging_config.py
import logging
import structlog

def configure_logging(level: str = 'INFO', log_file: str | None = None,):
    log_level = getattr(logging, level.upper())
    logging.basicConfig(filename=log_file, level=log_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

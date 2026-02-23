# order_dag.py
from maude_early_alert.logging_config import configure_logging
import structlog
from structlog.contextvars import bind_contextvars

configure_logging(level='INFO', log_file='ingest.log')

bind_contextvars(
    dag_id="order_dag",
    run_id="{{ run_id }}",
)

logger = structlog.get_logger()
logger.info("dag_started")

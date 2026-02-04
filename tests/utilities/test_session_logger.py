import ast
import datetime
import logging

import pytest
import structlog


@pytest.fixture
def test_client_factory():
    pass


def run_logs_msg(msg, logger, caplog):
    logger.debug(f"{msg}, debug.")
    logger.info(f"{msg}, info.")
    logger.warning(f"{msg}, warning.")
    return [ast.literal_eval(message)["event"] for message in caplog.messages]


def test_setup_logger(test_client_factory, caplog):
    from samgis_core.utilities import session_logger
    
    try:
        session_logger.setup_logging(json_logs=False, log_level="INFO")
        test_logger = structlog.stdlib.get_logger(__name__)
        events_info = run_logs_msg("msg_info", test_logger, caplog)
        if events_info != ["msg_info, info.", "msg_info, warning."]:
            raise ValueError(f"wrong log events (INFO): {type(events_info)} => {events_info} #")
        message = ast.literal_eval(caplog.messages[0])
        utcnow = datetime.datetime.now(datetime.timezone.utc)
        timestamp_fake = f"{utcnow.isoformat()}Z"
        message["timestamp"] = timestamp_fake
        logging.warning("test_setup_logger message:")
        logging.warning(msg=message)
        if message != {
                'event': 'msg_info, info.', 'func_name': 'run_logs_msg', 'level': 'info', 'lineno': 16,
                'logger': 'tests.utilities.test_session_logger', 'timestamp': timestamp_fake
            }:
            raise ValueError(f"wrong log message: {type(message)} => {message} #")

        caplog.clear()

        session_logger.setup_logging(json_logs=False, log_level="DEBUG")
        events_debug = run_logs_msg("msg_debug", test_logger, caplog)
        logging.warning("events_debug:")
        logging.warning(events_debug)
        if events_debug != ["msg_debug, debug.", "msg_debug, info.", "msg_debug, warning."]:
            raise ValueError(f"wrong log events (DEBUG): {type(events_debug)} => {events_debug} #")
    except ValueError as ve:
        logging.error("This test will fail if executed with a custom logging level!")
        raise ve


if __name__ == '__main__':
    pytest.main()

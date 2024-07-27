import ast
import datetime

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

    session_logger.setup_logging(json_logs=False, log_level="INFO")
    test_logger = structlog.stdlib.get_logger(__name__)

    events_info = run_logs_msg("msg_info", test_logger, caplog)
    assert events_info == ["msg_info, info.", "msg_info, warning."]
    message = ast.literal_eval(caplog.messages[0])
    utcnow = datetime.datetime.utcnow()
    timestamp_fake = f"{utcnow.isoformat()}Z"
    message["timestamp"] = timestamp_fake
    assert message == {
        'event': 'msg_info, info.', 'func_name': 'run_logs_msg', 'level': 'info', 'lineno': 15,
        'logger': 'tests.utilities.test_session_logger', 'timestamp': timestamp_fake
    }

    caplog.clear()

    session_logger.setup_logging(json_logs=False, log_level="DEBUG")
    events_info = run_logs_msg("msg_debug", test_logger, caplog)
    assert events_info == ["msg_debug, debug.", "msg_debug, info.", "msg_debug, warning."]


if __name__ == '__main__':
    pytest.main()

from samgis_core import PROJECT_ROOT_FOLDER, app_logger

TEST_ROOT_FOLDER = PROJECT_ROOT_FOLDER / "tests"
TEST_EVENTS_FOLDER = TEST_ROOT_FOLDER / "events"
LOCAL_URL_TILE = "http://localhost:8000/lambda_handler/{z}/{x}/{y}.png"

test_logger = app_logger

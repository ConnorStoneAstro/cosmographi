import datetime
import json
from pathlib import Path
import time
from typing import Dict

import pytest


_TIMING_RESULTS: Dict[str, float] = {}


@pytest.fixture
def mark_time(request):

    start = time.process_time()
    key = f"{str(request.module.__name__)}.{request.function.__name__}"

    yield

    _TIMING_RESULTS[key] = time.process_time() - start


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish():

    if _TIMING_RESULTS:
        timestamp = datetime.datetime.strftime(
            datetime.datetime.now(datetime.timezone.utc), "%Y-%m-%d %H:%M:%S"
        )
        with Path("test_timings.json").open("w") as f:
            json.dump(
                {
                    "test": list(_TIMING_RESULTS.keys()),
                    timestamp: list(_TIMING_RESULTS.values()),
                },
                f,
            )

    yield

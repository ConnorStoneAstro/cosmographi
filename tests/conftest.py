import datetime
import json
from pathlib import Path
from os import getenv
import time
from typing import Dict

import pytest


_TIMING_RESULTS: Dict[str, float] = {}

_TIMING_EXEMPT = []


@pytest.fixture(autouse=True)
def mark_time(request):

    start = time.process_time()
    key = f"{str(request.module.__name__)}.{request.function.__name__}"

    yield

    # only run in GH action environments
    if getenv("GITHUB_ACTION") is not None and key not in _TIMING_EXEMPT:
        _TIMING_RESULTS[key] = round(time.process_time() - start, 3)


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

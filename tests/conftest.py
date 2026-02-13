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
        with Path("test_timings2.json").open("w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "results": {
                        "test": list(_TIMING_RESULTS.keys()),
                        "time": list(_TIMING_RESULTS.values()),
                    },
                },
                f,
            )

    yield


# check_times(max_runs=50)
# each column header is a datetime string
# for the ones that are 20% more, add an alert
# that means storing them as artifacts (comma-separated string)
# then, if the artifact exists, you can use this action to add a comment to the PR: peter-evans/create-pull-request@v6
# This should certainly reference the commit hash as well

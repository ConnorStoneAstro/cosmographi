import json

import numpy as np
import pandas as pd

from cosmographi.scripts.compare_times import compare_test_times


def test_compare_times(tmp_path):

    tmpdir = tmp_path / "compare"
    tmpdir.mkdir()

    # we include an NaN, which is the value pandas would report for a test that's been
    # outer_join'd to the cached set but was not originally in it (i.e., test_foo was added on 2-14)
    cached_times = {
        "test": ["test_foo", "test_bar", "test_baz", "test_bad"],
        "2026-02-13 15:59:24": [np.NaN, 1, 1, 7],
        "2026-02-14 15:59:24": [1, 1, 1, 7],
        "2026-02-15 15:59:24": [2, 1.5, 1.3, 7],
    }

    # we have 2 outliers here, "test_foo" and "test_bar"
    latest_times = {
        "test": ["test_foo", "test_bar", "test_baz", "test_good"],
        "2026-02-16 15:59:24": [2, 1.3, 1.2, 1],
    }

    cached_path = tmpdir / "cached.json"
    latest_path = tmpdir / "latest.json"
    cache_save_path = tmpdir / "new_cache.json"
    outliers_save_path = tmpdir / "save.json"

    with open(cached_path, "w") as f:
        json.dump(cached_times, f)

    with open(latest_path, "w") as f:
        json.dump(latest_times, f)

    compare_test_times(cached_path, latest_path, outliers_save_path, cache_save_path)

    new_df = pd.read_json(cache_save_path)

    outliers = pd.read_json(outliers_save_path)

    # assert that we have 2 outliers
    assert len(outliers) == 2

    # assert that our total run count discards NaNs
    assert outliers[outliers["test"] == "test_foo"]["total_runs"].item() == 4

    # assert that our new cached times have 5 tests (4 existing, 1 new)
    assert len(new_df) == 5

    # assert that our new cached times have 4 dates (3 existing, 1 new),
    # plus the test name column
    assert len(new_df.columns) == 5

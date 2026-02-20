import json

import numpy as np
import pandas as pd

from cosmographi.scripts.compare_times import compare_test_times


def test_compare_times(tmp_path):

    tmpdir = tmp_path / "compare"
    tmpdir.mkdir()

    # we include an NaN, which is the value pandas would report for a test that's been
    # outer_join'd to the cached set but was not originally in it (i.e., test_foo was added on 2-14)
    foo_test = np.array([np.NaN, 1, 2])
    bar_test = np.array([1, 1, 1.5])
    baz_test = np.array([1, 1, 1.3])
    deleted_test = np.array([7, 7, 7])
    second_test = np.array([np.NaN, np.NaN, 1])

    tests = np.vstack([foo_test, bar_test, baz_test, deleted_test, second_test])

    # pandas uses 1 ddof, numpy 0 by default
    threshes = (np.nanstd(tests, ddof=1, axis=1) * 3) + np.nanmedian(tests, axis=1)

    cached_times = {
        "test": ["test_foo", "test_bar", "test_baz", "test_bad", "test_second"],
        "2026-02-13 15:59:24": list(tests.T[0]),
        "2026-02-14 15:59:24": list(tests.T[1]),
        "2026-02-15 15:59:24": list(tests.T[2]),
    }

    # we have 2 outliers here, "test_foo" and "test_bar"
    latest_times = {
        "test": ["test_foo", "test_bar", "test_baz", "test_new", "test_second"],
        "2026-02-16 15:59:24": [
            threshes[0] + 0.1,
            threshes[1] + 0.5,
            threshes[2] - 0.01,
            100,  # doen't matter, first run
            1.001,  # doesn't matter, thresh is na if only 1 in cache
        ],
    }

    cached_path = tmpdir / "cached.json"
    latest_path = tmpdir / "latest.json"
    cache_save_path = tmpdir / "new_cache.json"
    outliers_save_path = tmpdir / "outliers.json"
    flakes_save_path = tmpdir / "flakes.json"

    with open(cached_path, "w") as f:
        json.dump(cached_times, f)

    with open(latest_path, "w") as f:
        json.dump(latest_times, f)

    compare_test_times(
        cached_path, latest_path, outliers_save_path, flakes_save_path, cache_save_path
    )

    new_df = pd.read_json(cache_save_path)

    outliers = pd.read_json(outliers_save_path)

    # assert that we have 2 outliers
    assert len(outliers) == 2

    # assert that our total run count discards NaNs
    foo_outlier = outliers[outliers["test"] == "test_foo"]
    assert foo_outlier["total_runs"].item() == 4
    assert foo_outlier["median"].item() == np.nanmedian(foo_test).round(3)
    assert foo_outlier["std"].item() == np.nanstd(foo_test, ddof=1).round(3)

    # assert that our new cached times have 6 tests (5 existing, 1 new)
    assert len(new_df) == 6

    # assert that our new cached times have 4 dates (3 existing, 1 new),
    # plus the test name column
    assert len(new_df.columns) == 5

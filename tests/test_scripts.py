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
        "datetime": "2026-01-01",
        "benchmarks": [
            {"fullname": "test_foo", "stats": {"median": threshes[0] + 0.1}},
            {"fullname": "test_bar", "stats": {"median": threshes[1] + 0.5}},
            {"fullname": "test_baz", "stats": {"median": threshes[2] - 0.1}},
            {"fullname": "test_new", "stats": {"median": 100}},  # doesn't matter, first run
            {
                "fullname": "test_second",
                "stats": {"median": 100},
            },  # doesn't matter, thresh is no if only 1 in cache
        ],
    }

    cached_path = tmpdir / "cached.json"
    latest_path = tmpdir / "latest.json"
    cache_save_path = tmpdir / "new_cache.json"
    outliers_save_path = tmpdir / "outliers.md"
    flakes_save_path = tmpdir / "flakes.json"

    with open(cached_path, "w") as f:
        json.dump(cached_times, f)

    with open(latest_path, "w") as f:
        json.dump(latest_times, f)

    compare_test_times(
        cached_benchmarks_path=cached_path,
        new_benchmarks_path=latest_path,
        outliers_save_path=outliers_save_path,
        flakes_save_path=flakes_save_path,
        cache_save_path=cache_save_path,
    )

    new_df = pd.read_json(cache_save_path)

    outliers = pd.read_table(outliers_save_path, delimiter="|", skiprows=[1], header=0)
    # handle md table weirdness
    outliers.columns = [c.strip() for c in outliers.columns]
    outliers["test"] = [c.strip() for c in outliers["test"]]

    # assert that we have 2 outliers
    assert len(outliers) == 2

    # assert that our total run count discards NaNs
    foo_outlier = outliers[outliers["test"] == "test_foo"]
    assert foo_outlier["total_runs"].item() == 4
    assert np.isclose(foo_outlier["median"].item(), np.nanmedian(foo_test))
    assert np.isclose(foo_outlier["std"].item(), np.nanstd(foo_test, ddof=1))

    # assert that our new cached times have 6 tests (5 existing, 1 new)
    assert len(new_df) == 6

    # assert that our new cached times have 4 dates (3 existing, 1 new),
    # plus the test name column
    assert len(new_df.columns) == 5

    empty_cache_path = tmpdir / "empty.json"

    # test we can pass in an empty path and get a fresh cache
    compare_test_times(
        cached_benchmarks_path=empty_cache_path,
        new_benchmarks_path=latest_path,
        outliers_save_path=outliers_save_path,
        flakes_save_path=flakes_save_path,
        cache_save_path=empty_cache_path,
        save_latest_if_no_cache=True,
    )

    fresh_cache = pd.read_json(empty_cache_path)
    # we expect only test and timestamps cols
    assert len(fresh_cache.columns) == 2
    # 5 tests in the latest 'run'
    assert len(fresh_cache) == 5

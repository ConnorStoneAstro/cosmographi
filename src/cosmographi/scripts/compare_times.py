import json

import pandas as pd


def compare_test_times(
    cached_timings_path: str,
    new_timings_path: str,
    outliers_save_path: str,
    flakes_save_path: str,
    cache_save_path: str,
    max_cols: int = 50,
) -> None:
    """Compare the latest test times with a collection of previous test times
    and save a list of "outliers", currently defined as those that are more than
    25% slower than the median for that test. All inputs and outputs are JSON.
    Outliers will include the median, the test time, the date, and the total number
    of instances of that particular test in the cached set.

    Parameters
    ----------
    cached_timings_path : str
        The path to the JSON with the previous timings
    new_timings_path : str
        The path to the JSON with the new timings
    outliers_save_path : str
        The path where the outliers should be saved, if they exist
    flakes_save_path : str
        The path where the flakes should be saved, if they exist
    cache_save_path : str
        The path where the new timimgs (including current) should be saved
    max_cols: int, optional
        The maximum number of timing observations to cache, defaults to 50
    """
    with open(cached_timings_path, "r") as f:
        cached_timings = json.load(f)

    with open(new_timings_path, "r") as f:
        new_timings = json.load(f)

    df = pd.DataFrame(cached_timings)

    df.set_index("test", drop=True, inplace=True)

    median_stds = (
        df.std(axis=1)
        .round(3)
        .to_frame(name="std")
        .merge(
            df.median(axis=1).round(3).to_frame(name="median"), left_index=True, right_index=True
        )
    )

    df_new = pd.DataFrame(new_timings)

    df_new.set_index("test", drop=True, inplace=True)

    # join by test
    compare = pd.merge(df_new, median_stds, how="inner", left_index=True, right_index=True)

    outliers = compare[
        compare.loc[:, "median"] + (compare.loc[:, "std"] * 3) < compare.iloc[:, 0]
    ].reset_index(drop=False)

    new_cached = (
        pd.merge(df, df_new, how="outer", left_index=True, right_index=True)
        .iloc[:, -max_cols:]
        .reset_index(drop=False)
    )

    new_cached.to_json(cache_save_path)

    if len(outliers):
        outliers["total_runs"] = new_cached.count(axis=1)
        outliers.to_json(outliers_save_path, orient="records")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Compare cached test times against the latest.")

    parser.add_argument(
        "cached_timings_path",
        type=str,
        help="Path to the cached timings JSON file.",
    )

    parser.add_argument(
        "new_timings_path",
        type=str,
        help="Path to the new timings JSON file.",
    )

    parser.add_argument(
        "outliers_save_path",
        type=str,
        help="Path where the outliers should be saved.",
    )

    parser.add_argument(
        "flakes_save_path",
        type=str,
        help="Path where the flakds should be saved.",
    )

    parser.add_argument(
        "cache_save_path",
        type=str,
        help="Path where the new timings should be saved.",
    )

    parser.add_argument(
        "--max_cols",
        type=int,
        default=50,
        help="The maximum number of columns in the cached frame.",
    )

    args = parser.parse_args()

    compare_test_times(**vars(args))

import json

import pandas as pd


def parse_benchmark_file(file_path):
    rows = []
    with open(file_path, "r") as f:
        data = json.load(f)
        timestamp = data["datetime"]
        for benchmark in data["benchmarks"]:
            rows.append(
                dict(
                    test=benchmark["fullname"],
                    duration=benchmark["stats"]["median"],
                )
            )
    return rows, timestamp


def compare_test_times(
    cached_benchmarks_path: str,
    new_benchmarks_path: str,
    outliers_save_path: str,
    flakes_save_path: str,
    cache_save_path: str,
    save_latest_if_no_cache: bool = False,
    max_cols: int = 50,
) -> None:
    """Compare the latest test times with a collection of previous test times
    and save a list of "outliers", currently defined as those that are more than
    3 stds slower than the median for that test. All inputs and outputs are JSON,
    except for the outliers, which is markdown.
    Outliers will include the median, std, the test time, the date, and the total number
    of instances of that particular test in the cached set.

    Note that if the ``cached_benchmarks_path`` does not exist, the latest (formatted)
    benchmarks will be saved at ``cache_save_path`` as long as it is passed.

    Parameters
    ----------
    cached_benchmarks_path : str
        The path to the JSON with the previous benchmarks
    new_benchmarks_path : str
        The path to the JSON with the new benchmarks
    outliers_save_path : str
        The path where the outliers should be saved, if they exist
    flakes_save_path : str
        The path where the flakes should be saved, if they exist
    cache_save_path : str
        The path where the new timimgs (including current) should be saved
    save_latest_if_no_cache: bool, defaults to False
        Whether to save the latest benchmarks as the cache if the cache file is not found
    max_cols: int, defaults to 50
        The maximum number of benchmark observations to cache, defaults to 50
    """

    with open(new_benchmarks_path, "r") as f:
        new_benchmarks = json.load(f)

    new_benchmarks, timestamp = parse_benchmark_file(new_benchmarks_path)

    df_new = pd.DataFrame(new_benchmarks).set_index("test", drop=True)

    df_new.columns = [timestamp]

    try:
        with open(cached_benchmarks_path, "r") as f:
            cached_benchmarks = json.load(f)
    except FileNotFoundError:
        if save_latest_if_no_cache:
            df_new = df_new.reset_index(drop=False)
            df_new.to_json(cache_save_path, index=False, orient="records")
            return
        else:
            raise

    df = pd.DataFrame(cached_benchmarks)

    df.set_index("test", drop=True, inplace=True)

    median_stds = (
        df.std(axis=1)
        .round(8)
        .to_frame(name="std")
        .merge(
            df.median(axis=1).round(8).to_frame(name="median"), left_index=True, right_index=True
        )
    )

    # join by test
    compare = pd.merge(df_new, median_stds, how="inner", left_index=True, right_index=True)

    outliers = compare[
        compare.loc[:, "median"] + (compare.loc[:, "std"] * 3) < compare[timestamp]
    ].reset_index(drop=False)

    new_cached = (
        pd.merge(df, df_new, how="outer", left_index=True, right_index=True)
        .iloc[:, -max_cols:]
        .reset_index(drop=False)
    )

    new_cached.to_json(cache_save_path, index=False, orient="records")

    if len(outliers):
        outliers["total_runs"] = new_cached.count(axis=1)
        outliers.rename({timestamp: "duration"}, axis=1, inplace=True)
        outliers.to_markdown(outliers_save_path, index=False, tablefmt="github")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Compare cached test times against the latest.")

    parser.add_argument(
        "cached_benchmarks_path",
        type=str,
        help="Path to the cached benchmarks JSON file.",
    )

    parser.add_argument(
        "new_benchmarks_path",
        type=str,
        help="Path to the new benchmarks JSON file.",
    )

    parser.add_argument(
        "outliers_save_path",
        type=str,
        help="Path where the outliers should be saved.",
    )

    parser.add_argument(
        "flakes_save_path",
        type=str,
        help="Path where the flakes should be saved.",
    )

    parser.add_argument(
        "cache_save_path",
        type=str,
        help="Path where the flakes should be saved.",
    )

    parser.add_argument(
        "--max_cols",
        type=int,
        help="The maximum number of test runs to save.",
        default=50,
    )

    parser.add_argument(
        "--save_latest_if_no_cache",
        default=False,
        action="store_true",
        help="Whether to save the latest benchmarks as the cache if the cache file is not found.",
    )

    args = parser.parse_args()

    compare_test_times(**vars(args))

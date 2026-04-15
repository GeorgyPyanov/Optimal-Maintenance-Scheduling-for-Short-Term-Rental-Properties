import pandas as pd
import pytest


@pytest.mark.integration
def test_result_dataset_core_ranges_are_valid():
    # Integration check for schema and hard numeric boundaries in production dataset.
    df = pd.read_csv("src/result_dataset.csv")
    required = {
        "id",
        "latitude",
        "longitude",
        "best_maintenance_month",
        "availability_ratio",
        "aco_priority",
    }
    assert required.issubset(df.columns)
    assert df["id"].is_unique
    assert df["best_maintenance_month"].between(1, 12).all()
    assert df["availability_ratio"].between(0, 1).all()
    assert (df["aco_priority"] >= 0).all()


@pytest.mark.integration
def test_result_dataset_month_distribution_has_expected_april_peak():
    # Regression guard for current preprocessing output distribution.
    df = pd.read_csv("src/result_dataset.csv")
    top_month = int(df["best_maintenance_month"].value_counts().idxmax())

    # Regression guard for current preprocessing output: April dominates.
    assert top_month == 4

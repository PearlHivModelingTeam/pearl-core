"""
Tests for population.py
"""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import fixture

from pearl.parameters import Parameters
from pearl.population import PearlPopulation


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def expected_population():
    return pd.read_parquet(Path("tests/pearl/assets/population_test/population.parquet"))


@fixture
def test_parameters(param_file_path):
    return Parameters(
        path=param_file_path,
        output_folder=None,
        replication=42,
        group_name="msm_black_male",
        new_dx="base",
        final_year=2015,
        mortality_model="by_sex_race_risk",
        mortality_threshold_flag=1,
        idu_threshold="2x",
        seed=42,
    )


def test_pearl_populations(test_parameters, expected_population):
    result_population = PearlPopulation(test_parameters)(pd.DataFrame([]))
    assert_frame_equal(result_population, expected_population)

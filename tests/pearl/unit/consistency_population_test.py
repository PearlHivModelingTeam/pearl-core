"""
Tests for population.py to ensure consistency between new and old pearl
"""

from pathlib import Path

import numpy as np
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
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_population_test/before_run_population.parquet")
    ).drop(columns=["index"])
    population = population.reset_index()
    population["id"] = np.array(range(population.index.size))
    population = population.set_index(["id"])
    return population


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

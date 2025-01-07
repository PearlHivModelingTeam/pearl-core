# test module for base_model.py
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import fixture

from pearl.base_model import BasePearl
from pearl.parameters import Parameters


@fixture
def param_file_path():
    return Path("tests/pearl/assets/parameters.h5")


@fixture
def config():
    config = {
        "new_dx": "base",
        "final_year": 2012,
        "mortality_model": "by_sex_race_risk",
        "mortality_threshold_flag": 1,
        "idu_threshold": "2x",
        "verbose": 0,
    }
    return config


@fixture
def expected_population():
    population = pd.read_parquet(
        Path("tests/pearl/assets/base_model_test/final_state.parquet")
    ).drop(columns=["index", "group", "replication"])

    population = population.reset_index()
    population["id"] = np.array(range(population.index.size))
    population = population.set_index(["id"])

    return population


@fixture
def starting_population():
    population = pd.read_parquet(
        Path("tests/pearl/assets/base_model_test/starting_population.parquet")
    ).drop(columns=["index"])

    population = population.reset_index()
    population["id"] = np.array(range(population.index.size))
    population = population.set_index(["id"])

    return population


@fixture
def parameters(param_file_path, config):
    return Parameters(
        path=param_file_path,
        output_folder=None,
        replication=1,
        group_name="msm_black_male",
        new_dx=config["new_dx"],
        start_year=2009,
        final_year=config["final_year"],
        mortality_model=config["mortality_model"],
        final_state=True,
        mortality_threshold_flag=config["mortality_threshold_flag"],
        idu_threshold=config["idu_threshold"],
        seed=42,
    )


@fixture
def test_model(parameters):
    return BasePearl(parameters)


def test_starting_population(test_model, starting_population):
    assert_frame_equal(test_model.population, starting_population)


def test_base_pearl(test_model, expected_population):
    test_model.run()

    assert_frame_equal(test_model.population, expected_population)

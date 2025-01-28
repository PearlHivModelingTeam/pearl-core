# test module to ensure that the behavior in the new pearl engine matches original
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import fixture

from pearl.events import (
    AddNewUser,
    ComorbidityIncidence,
    DecreaseCD4Count,
    IncreaseCD4Count,
    IncrementYear,
    KillInCare,
    KillOutCare,
    LoseToFollowUp,
    Reengage,
    append_new,
    update_mm,
)
from pearl.parameters import Parameters


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
def increment_years_parameters(config: dict[str, Any]):
    return Parameters(
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
def before_increment_year():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/start.parquet")
    ).drop(columns=["index"])

    population = population.reset_index()
    population["id"] = np.array(range(population.index.size))
    population = population.set_index(["id"])

    return population


@fixture
def after_increment_year():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_increment_years.parquet")
    ).drop(columns=["index"])

    return population


def test_IncrementYears(  # noqa: N802
    increment_years_parameters: Parameters,
    before_increment_year: pd.DataFrame,
    after_increment_year: pd.DataFrame,
):
    increment_year = IncrementYear(increment_years_parameters)

    population_after_increment_year = increment_year(before_increment_year)

    assert_frame_equal(population_after_increment_year, after_increment_year)

    assert increment_years_parameters.year == increment_years_parameters.start_year + 1


@fixture
def random_state():
    return np.random.RandomState(seed=42)


@fixture
def parameters(config: dict[str, Any]):
    parameters = Parameters(
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

    parameters.year += 1
    return parameters


@fixture
def after_comorbidity_incidence():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_comorbidity_incidence.parquet")
    ).drop(columns=["index"])

    return population


def test_comorbidity_incidence(
    parameters, after_increment_year, after_comorbidity_incidence, random_state
):
    # reset the random state for testing
    parameters.random_state = random_state
    comorbidity_incidence = ComorbidityIncidence(parameters)

    population_after_comorbidity_incidence = comorbidity_incidence(after_increment_year)

    assert_frame_equal(population_after_comorbidity_incidence, after_comorbidity_incidence)


@fixture
def after_update_mm():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_update_mm.parquet")
    ).drop(columns=["index"])

    return population


def test_update_mm(parameters, after_comorbidity_incidence, after_update_mm, random_state):
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_update_mm = update_mm(after_comorbidity_incidence)

    assert_frame_equal(population_after_update_mm, after_update_mm)


@fixture
def after_increase_cd4_count():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_increase_cd4_count.parquet")
    ).drop(columns=["index"])

    return population


def test_IncreaseCD4Count(parameters, after_update_mm, after_increase_cd4_count, random_state):  # noqa: N802
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_increase_cd4_count = IncreaseCD4Count(parameters)(after_update_mm)

    assert_frame_equal(population_after_increase_cd4_count, after_increase_cd4_count)


@fixture
def after_add_new_user():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_add_new_user.parquet")
    ).drop(columns=["index"])

    return population


def test_AddNewUser(parameters, after_increase_cd4_count, after_add_new_user, random_state):  # noqa: N802
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_add_new_user = AddNewUser(parameters)(after_increase_cd4_count)

    assert_frame_equal(population_after_add_new_user, after_add_new_user)


@fixture
def after_kill_in_care():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_kill_in_care.parquet")
    ).drop(columns=["index"])

    return population


def test_KillInCare(parameters, after_add_new_user, after_kill_in_care, random_state):  # noqa: N802
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_add_new_user = KillInCare(parameters)(after_add_new_user)

    assert_frame_equal(population_after_add_new_user, after_kill_in_care)


@fixture
def after_lose_to_follow_up():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_lose_to_follow_up.parquet")
    ).drop(columns=["index"])

    return population


def test_LoseToFollowUp(parameters, after_kill_in_care, after_lose_to_follow_up, random_state):  # noqa: N802
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_kill_in_care = LoseToFollowUp(parameters)(after_kill_in_care)

    assert_frame_equal(population_after_kill_in_care, after_lose_to_follow_up)


@fixture
def after_decrease_cd4_count():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_decrease_cd4_count.parquet")
    ).drop(columns=["index"])

    return population


def test_DecreaseCD4Count(  # noqa: N802
    parameters, after_lose_to_follow_up, after_decrease_cd4_count, random_state
):
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_decrease_cd4_count = DecreaseCD4Count(parameters)(after_lose_to_follow_up)

    assert_frame_equal(population_after_decrease_cd4_count, after_decrease_cd4_count)


@fixture
def after_kill_out_care():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_kill_out_care.parquet")
    ).drop(columns=["index"])

    return population


def test_KillOutCare(  # noqa: N802
    parameters, after_decrease_cd4_count, after_kill_out_care, random_state
):
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_kill_out_care = KillOutCare(parameters)(after_decrease_cd4_count)

    assert_frame_equal(population_after_kill_out_care, after_kill_out_care)


@fixture
def after_reengage():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_reengage.parquet")
    ).drop(columns=["index"])

    return population


def test_Reengage(  # noqa: N802
    parameters, after_kill_out_care, after_reengage, random_state
):
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_after_reengage = Reengage(parameters)(after_kill_out_care)

    assert_frame_equal(population_after_after_reengage, after_reengage)


@fixture
def after_append_new():
    population = pd.read_parquet(
        Path("tests/pearl/assets/consistency_events_test/after_append_new.parquet")
    ).drop(columns=["index"])

    return population


def test_append_new(parameters, after_reengage, after_append_new, random_state):
    # reset the random state for testing
    parameters.random_state = random_state

    population_after_after_append_new = append_new(after_reengage)

    assert_frame_equal(population_after_after_append_new, after_append_new)

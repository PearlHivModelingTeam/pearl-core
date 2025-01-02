"""Module for events"""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pearl.calculations import calculate_prob
from pearl.definitions import (
    ART_NAIVE,
    ART_NONUSER,
    ART_USER,
    DEAD_ART_NONUSER,
    DEAD_ART_USER,
    DYING_ART_NONUSER,
    DYING_ART_USER,
    LTFU,
    REENGAGED,
    SMEARING,
    STAGE1,
    STAGE2,
    STAGE3,
)
from pearl.engine import Event
from pearl.interpolate import restricted_cubic_spline_var, restricted_quadratic_spline_var
from pearl.multimorbidity import create_comorbidity_pop_matrix
from pearl.parameters import Parameters


def append_new(population: pd.DataFrame) -> pd.DataFrame:
    """Move agents from the temporary, statuses to the main statuses at the end of the year."""
    reengaged = population["status"] == REENGAGED
    ltfu = population["status"] == LTFU
    dying_art_user = population["status"] == DYING_ART_USER
    dying_art_nonuser = population["status"] == DYING_ART_NONUSER

    population.loc[reengaged, "status"] = ART_USER
    population.loc[ltfu, "status"] = ART_NONUSER
    population.loc[dying_art_user, "status"] = DEAD_ART_USER
    population.loc[dying_art_nonuser, "status"] = DEAD_ART_NONUSER

    return population


def create_mortality_out_care_pop_matrix(pop: pd.DataFrame, parameters: Parameters) -> Any:
    """
    Return the population matrix as a numpy array for calculating mortality out of care.
    This log odds of mortality are a linear function of calendar year and age and sqrt cd4 count
    modeled as restricted cubic splines. If using comorbidities, log odds of mortality are a
    linear  function of calendar year, age category, sqrt cd4 count, delta bmi and post art bmi
    modeled as  restricted cubic splines, and presence of each individual comorbidity modeled as
    binary variables.

    Parameters
    ----------
    pop : pd.DataFrame
        Population Dataframe containing age_cat, anx, post_art_bmi, ckd, dm, dpr, esld, hcv,
        ht, intercept, lipid, malig, mi, smoking, time_varying_sqrtcd4n, init_sqrtcd4n, and year
        columns.
    parameters : Parameters
        Parameters object with mortality_out_care_post_art_bmi attribute.

    Returns
    -------
    NDArray[Any]
        A numpy representation of the population for feedining into Pearl.calculate_prob to
        calculate death probability for out care mortality modeling.
    """

    pop["post_art_bmi_"] = restricted_cubic_spline_var(
        pop["post_art_bmi"].to_numpy(), parameters.mortality_out_care_post_art_bmi.to_numpy(), 1
    )
    pop["post_art_bmi__"] = restricted_cubic_spline_var(
        pop["post_art_bmi"].to_numpy(), parameters.mortality_out_care_post_art_bmi.to_numpy(), 2
    )
    return pop[
        [
            "age_cat",
            "anx",
            "post_art_bmi",
            "post_art_bmi_",
            "post_art_bmi__",
            "ckd",
            "dm",
            "dpr",
            "esld",
            "hcv",
            "ht",
            "intercept",
            "lipid",
            "malig",
            "mi",
            "smoking",
            "time_varying_sqrtcd4n",
            "year",
        ]
    ].to_numpy(dtype=float)


def calculate_cd4_decrease(
    pop: pd.DataFrame, parameters: Parameters, smearing: Optional[float] = SMEARING
) -> NDArray[Any]:
    """
    Calculate out of care cd4 count via a linear function of years out of care and sqrt cd4
    count at exit from care.

    Parameters
    ----------
    pop : pd.DataFrame
        Population Dataframe containing intercept, ltfu_year, sqrtcd4n_exit, and year columns.
    parameters : Parameters
        Parameters object with cd4_decrease attribute.
    smearing : Optional[float], optional
        Smearing value, by default SMEARING value set in Pearl.definitions.

    Returns
    -------
    NDArray[Any]
        numpy array representing the decreased cd4 count values.
    """

    coeffs = parameters.cd4_decrease.to_numpy(dtype=float)

    # Calculate the time_out variable and perform the matrix multiplication
    pop["time_out"] = pop["year"] - pop["ltfu_year"]
    pop_matrix = pop[["intercept", "time_out", "sqrtcd4n_exit"]].to_numpy(dtype=float)
    diff = np.matmul(pop_matrix, coeffs)

    new_cd4 = np.sqrt((pop["sqrtcd4n_exit"].to_numpy(dtype=float) ** 2) * np.exp(diff) * smearing)
    new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
    return np.array(new_cd4)


def create_ltfu_pop_matrix(pop: pd.DataFrame, knots: pd.DataFrame) -> Any:
    """
    Create and return the population matrix as a numpy array for use in calculating probability
    of loss to follow up.

    Parameters
    ----------
    pop : pd.DataFrame
        The population DataFrame that we wish to calculate loss to follow up on.
    knots : pd.DataFrame
        Quadratic spline knot values which are stored in Parameters.

    Returns
    -------
    NDArray[Any]
        numpy array for passing into Pearl.calculate_prob
    """
    # Create all needed intermediate variables
    knots = knots.to_numpy()
    pop["age_"] = restricted_quadratic_spline_var(pop["age"].to_numpy(), knots, 1)
    pop["age__"] = restricted_quadratic_spline_var(pop["age"].to_numpy(), knots, 2)
    pop["age___"] = restricted_quadratic_spline_var(pop["age"].to_numpy(), knots, 3)
    pop["haart_period"] = (pop["h1yy"].values > 2010).astype(int)
    return pop[
        [
            "intercept",
            "age",
            "age_",
            "age__",
            "age___",
            "year",
            "init_sqrtcd4n",
            "haart_period",
        ]
    ].to_numpy()


def create_mortality_in_care_pop_matrix(pop: pd.DataFrame, parameters: Parameters) -> Any:
    """
    Return the population matrix as a numpy array for calculating mortality in care. This log
    odds of mortality are a linear function of calendar year, ART init year category modeled as two
    binary variables, and age and sqrt initial cd4 count modeled as restricted cubic splines. Log
    odds of mortality are a linear function of calendar year, age category, initial cd4 count,
    delta bmi and post art bmi modeled as restricted cubic splines, and presence of each individual
    comorbidity modeled as binary variables.

    Parameters
    ----------
    pop : pd.DataFrame
        Population Dataframe containing age_cat, anx, post_art_bmi, ckd, dm, dpr, esld, h1yy, hcv,
        ht, intercept, lipid, malig, mi, smoking, init_sqrtcd4n, and year columns.
    parameters : Parameters
        Parameters object with mortality_in_care_post_art_bmi attribute.

    Returns
    -------
    NDArray[Any]
        A numpy representation of the population for feedining into Pearl.calculate_prob to
        calculate death probability for in care mortality modeling.
    """

    pop["post_art_bmi_"] = restricted_cubic_spline_var(
        pop["post_art_bmi"].to_numpy(), parameters.mortality_in_care_post_art_bmi.to_numpy(), 1
    )
    pop["post_art_bmi__"] = restricted_cubic_spline_var(
        pop["post_art_bmi"].to_numpy(), parameters.mortality_in_care_post_art_bmi.to_numpy(), 2
    )
    return pop[
        [
            "age_cat",
            "anx",
            "post_art_bmi",
            "post_art_bmi_",
            "post_art_bmi__",
            "ckd",
            "dm",
            "dpr",
            "esld",
            "h1yy",
            "hcv",
            "ht",
            "intercept",
            "lipid",
            "malig",
            "mi",
            "smoking",
            "init_sqrtcd4n",
            "year",
        ]
    ].to_numpy(dtype=float)


def update_mm(population: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and update the multimorbidity, defined as the number of stage 2 and 3
    comorbidities in each agent.
    """
    population["mm"] = population[STAGE2 + STAGE3].sum(axis=1)

    return population


def calculate_cd4_increase(pop: pd.DataFrame, parameters: Parameters) -> NDArray[Any]:
    """
    Return new cd4 count of the given population as calculated via a linear function of time
    since art initiation modeled as a spline, initial cd4 count category, age category and
    cross terms.

    Parameters
    ----------
    pop : pd.DataFrame
        Population Dataframe containing age_cat, intercept, last_h1yy, last_init_sqrtcd4n, and year
        columns.
    parameters : Parameters
        Parameters object with cd4_increase_knots and cd4_increase attributes.

    Returns
    -------
    NDArray[Any]
        numpy array representing the increased cd4 count values.
    """
    knots = parameters.cd4_increase_knots
    coeffs = parameters.cd4_increase.to_numpy(dtype=float)

    # Calculate spline variables
    pop["time_from_h1yy"] = pop["year"] - pop["last_h1yy"]
    pop["time_from_h1yy_"] = restricted_quadratic_spline_var(
        pop["time_from_h1yy"].to_numpy(), knots.to_numpy(), 1
    )
    pop["time_from_h1yy__"] = restricted_quadratic_spline_var(
        pop["time_from_h1yy"].to_numpy(), knots.to_numpy(), 2
    )
    pop["time_from_h1yy___"] = restricted_quadratic_spline_var(
        pop["time_from_h1yy"].to_numpy(), knots.to_numpy(), 3
    )

    # Calculate CD4 Category Variables
    pop["cd4_cat_349"] = (
        pop["last_init_sqrtcd4n"].ge(np.sqrt(200.0)) & pop["last_init_sqrtcd4n"].lt(np.sqrt(350.0))
    ).astype(int)
    pop["cd4_cat_499"] = (
        pop["last_init_sqrtcd4n"].ge(np.sqrt(350.0)) & pop["last_init_sqrtcd4n"].lt(np.sqrt(500.0))
    ).astype(int)
    pop["cd4_cat_500"] = pop["last_init_sqrtcd4n"].ge(np.sqrt(500.0)).astype(int)

    # Create cross term variables
    pop["timecd4cat349_"] = pop["time_from_h1yy_"] * pop["cd4_cat_349"]
    pop["timecd4cat499_"] = pop["time_from_h1yy_"] * pop["cd4_cat_499"]
    pop["timecd4cat500_"] = pop["time_from_h1yy_"] * pop["cd4_cat_500"]
    pop["timecd4cat349__"] = pop["time_from_h1yy__"] * pop["cd4_cat_349"]
    pop["timecd4cat499__"] = pop["time_from_h1yy__"] * pop["cd4_cat_499"]
    pop["timecd4cat500__"] = pop["time_from_h1yy__"] * pop["cd4_cat_500"]
    pop["timecd4cat349___"] = pop["time_from_h1yy___"] * pop["cd4_cat_349"]
    pop["timecd4cat499___"] = pop["time_from_h1yy___"] * pop["cd4_cat_499"]
    pop["timecd4cat500___"] = pop["time_from_h1yy___"] * pop["cd4_cat_500"]

    # Create numpy matrix
    pop_matrix = pop[
        [
            "intercept",
            "time_from_h1yy",
            "time_from_h1yy_",
            "time_from_h1yy__",
            "time_from_h1yy___",
            "cd4_cat_349",
            "cd4_cat_499",
            "cd4_cat_500",
            "age_cat",
            "timecd4cat349_",
            "timecd4cat499_",
            "timecd4cat500_",
            "timecd4cat349__",
            "timecd4cat499__",
            "timecd4cat500__",
            "timecd4cat349___",
            "timecd4cat499___",
            "timecd4cat500___",
        ]
    ].to_numpy(dtype=float)

    # Perform matrix multiplication
    new_cd4 = np.matmul(pop_matrix, coeffs)

    new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
    return np.array(new_cd4)


class AddNewUser(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

    def __call__(self, population):
        """Add newly initiating ART users."""
        new_user = (population["status"] == ART_NAIVE) & (
            population["h1yy"] == self.parameters.year
        )
        population.loc[new_user, "status"] = ART_USER

        return population


class IncreaseCD4Count(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

    def __call__(self, population):
        """Calculate and set new CD4 count for ART using population."""
        in_care = population["status"] == ART_USER

        new_sqrt_cd4 = calculate_cd4_increase(population.loc[in_care].copy(), self.parameters)

        population.loc[in_care, "time_varying_sqrtcd4n"] = new_sqrt_cd4

        return population


class IncrementYear(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Increment calendar year for all agents, increment age and age_cat for those alive in the
        model, and increment the number of years spent out of care for the ART non-using
        population.
        """
        self.parameters.year += 1
        alive_and_initiated = population["status"].isin([ART_USER, ART_NONUSER])
        out_care = population["status"] == ART_NONUSER
        population["year"] = np.array(self.parameters.year, dtype="int16")
        population.loc[alive_and_initiated, "age"] += np.array(1, dtype="int8")
        population["age_cat"] = np.floor(population["age"] / 10).astype("int8")
        population.loc[population["age_cat"] < 2, "age_cat"] = np.array(2, dtype="int8")
        population.loc[population["age_cat"] > 7, "age_cat"] = np.array(7, dtype="int8")
        population.loc[out_care, "years_out"] += np.array(1, dtype="int8")

        return population


class ComorbidityIncidence(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.coeff = self.parameters.comorbidity_coeff_dict
        self.sa_variables = parameters.sa_variables
        self.sa_scalars = parameters.sa_scalars

    def __call__(self, population):
        """Calculate probability of incidence of all comorbidities and then draw to determine which
        agents experience incidence. Record incidence data stratified by care status and age
        category.
        """
        in_care = population["status"] == ART_USER
        out_care = population["status"] == ART_NONUSER

        # Iterate over all comorbidities
        for condition in STAGE1 + STAGE2 + STAGE3:
            # Calculate probability
            coeff_matrix = self.coeff[condition].to_numpy(dtype=float)
            pop_matrix = create_comorbidity_pop_matrix(
                population.copy(), condition=condition, parameters=self.parameters
            )

            prob = calculate_prob(
                pop_matrix,
                coeff_matrix,
            )

            if self.self.sa_variables and f"{condition}_incidence" in self.sa_variables:
                prob = np.clip(
                    a=prob * self.sa_scalars[f"{condition}_incidence"],
                    a_min=0,
                    a_max=1,
                )

            # Draw for incidence
            rand = prob > self.random_state.rand(len(population.index))
            old = population[condition]
            new = rand & (in_care | out_care) & ~old  # new incident comorbidities
            population[condition] = (old | new).astype("bool")

            # Update time of incident comorbidity
            population[f"t_{condition}"] = np.array(
                population[f"t_{condition}"] + new * self.parameters.year, dtype="int16"
            )

            return population


class KillInCare(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.coeff = parameters.mortality_in_care_co
        self.mortality_flag = parameters.mortality_threshold_flag
        self.mortality_threshold = parameters.mortality_threshold

    def __call__(self, population):
        """Calculate probability of mortality for in care population. Optionally, use the general
        population mortality threshold to increase age category grouped probability of mortality to
        have the same mean as the general population. Draw random numbers to determine
        who will die.
        """

        # Calculate death probability
        in_care = population["status"] == ART_USER
        pop = population.copy()
        coeff_matrix = self.coeff.to_numpy(dtype=float)

        pop_matrix = create_mortality_in_care_pop_matrix(pop.copy(), parameters=self.parameters)

        pop["death_prob"] = calculate_prob(
            pop_matrix,
            coeff_matrix,
        )

        # Increase mortality to general population threshold
        if self.mortality_threshold_flag:
            pop["mortality_age_group"] = pd.cut(
                pop["age"],
                bins=[0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85],
                right=True,
                labels=np.arange(14),
            )
            mean_mortality = pd.DataFrame(
                pop.loc[in_care]
                .groupby(["mortality_age_group"], observed=False)["death_prob"]
                .mean()
            )
            mean_mortality["p"] = self.mortality_threshold["p"] - mean_mortality["death_prob"]
            mean_mortality.loc[mean_mortality["p"] <= 0, "p"] = 0
            for mortality_age_group in np.arange(14):
                excess_mortality = mean_mortality.loc[mortality_age_group, "p"]
                pop.loc[
                    in_care & (pop["mortality_age_group"] == mortality_age_group),
                    "death_prob",
                ] += excess_mortality

        # Draw for mortality
        died = (
            (pop["death_prob"] > self.random_state.rand(len(population.index)))
            | (population["age"] > 85)
        ) & in_care
        population.loc[died, "status"] = DYING_ART_USER
        population.loc[died, "year_died"] = np.array(self.year, dtype="int16")

        return population


class LoseToFollowUp(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.coeff = parameters.loss_to_follow_up

    def __call__(self, population):
        """Calculate probability of in care agents leaving care. Draw random number to decide who
        leaves care. For those leaving care, draw the number of years to spend out of care from a
        normalized, truncated Poisson distribution.
        """
        # Calculate probability and draw
        in_care = population["status"] == ART_USER
        pop = population.copy()
        coeff_matrix = self.coeff.to_numpy(dtype=float)
        pop_matrix = create_ltfu_pop_matrix(pop.copy(), self.parameters.ltfu_knots)
        pop["ltfu_prob"] = calculate_prob(
            pop_matrix,
            coeff_matrix,
        )

        lost = (pop["ltfu_prob"] > self.random_state.rand(len(population.index))) & in_care

        p = self.parameters.years_out_of_care["probability"]

        years_out_of_care = self.random_state.choice(
            a=self.parameters.years_out_of_care["years"],
            size=len(population.loc[lost]),
            p=p,
        )

        # Set variables for lost population
        population.loc[lost, "return_year"] = (self.parameters.year + years_out_of_care).astype(
            "int16"
        )
        population.loc[lost, "status"] = LTFU
        population.loc[lost, "sqrtcd4n_exit"] = population.loc[lost, "time_varying_sqrtcd4n"]
        population.loc[lost, "ltfu_year"] = self.parameters.year
        population.loc[lost, "n_lost"] += 1

        return population


class DecreaseCD4Count(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

    def __call__(self, population):
        """Calculate and set new CD4 count for ART non-using population."""
        out_care = self.population["status"] == ART_NONUSER
        new_sqrt_cd4 = calculate_cd4_decrease(
            self.population.loc[out_care].copy(), self.parameters
        )

        self.population.loc[out_care, "time_varying_sqrtcd4n"] = new_sqrt_cd4


class KillOutCare(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.coeff = parameters.mortality_out_care_co
        self.mortality_threshold_flag = self.parameters.mortality_threshold_flag
        self.mortality_threshold = parameters.mortality_threshold

    def __call__(self, population):
        """Calculate probability of mortality for out of care population. Optionally, use the
        general population mortality threshold to increase age category grouped probability of
        mortality to have the same mean as the general population. Draw random
        numbers to determine who will die.
        """

        # Calculate death probability
        out_care = population["status"] == ART_NONUSER
        pop = population.copy()
        coeff_matrix = self.coeff.to_numpy(dtype=float)

        pop_matrix = create_mortality_out_care_pop_matrix(pop.copy(), parameters=self.parameters)

        pop["death_prob"] = calculate_prob(pop_matrix, coeff_matrix)

        # Increase mortality to general population threshold
        if self.mortality_threshold_flag:
            pop["mortality_age_group"] = pd.cut(
                pop["age"],
                bins=[0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85],
                right=True,
                labels=np.arange(14),
            )
            mean_mortality = pd.DataFrame(
                pop.loc[out_care]
                .groupby(["mortality_age_group"], observed=False)["death_prob"]
                .mean()
            )
            mean_mortality["p"] = self.mortality_threshold["p"] - mean_mortality["death_prob"]
            mean_mortality.loc[mean_mortality["p"] <= 0, "p"] = 0
            for mortality_age_group in np.arange(14):
                excess_mortality = mean_mortality.loc[mortality_age_group, "p"]
                pop.loc[
                    out_care & (pop["mortality_age_group"] == mortality_age_group),
                    "death_prob",
                ] += excess_mortality

        # Draw for mortality
        died = (
            (pop["death_prob"] > self.random_state.rand(len(population.index)))
            | (population["age"] > 85)
        ) & out_care
        population.loc[died, "status"] = DYING_ART_NONUSER
        population.loc[died, "year_died"] = np.array(self.parameters.year, dtype="int16")
        population.loc[died, "return_year"] = 0

        return population


class Reengage(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

    def __call__(self, population):
        """Move out of care population scheduled to reenter care."""
        out_care = population["status"] == ART_NONUSER
        reengaged = (self.parameters.year == population["return_year"]) & out_care
        population.loc[reengaged, "status"] = REENGAGED

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        population.loc[reengaged, "last_init_sqrtcd4n"] = population.loc[
            reengaged, "time_varying_sqrtcd4n"
        ]
        population.loc[reengaged, "last_h1yy"] = self.parameters.year
        population.loc[reengaged, "return_year"] = 0
        population.loc[reengaged, "years_out"] = 0

        return population

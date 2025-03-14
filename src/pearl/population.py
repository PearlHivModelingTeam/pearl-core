from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import override

from pearl.definitions import (
    ART_NAIVE,
    ART_NONUSER,
    ART_USER,
    DELAYED,
    POPULATION_TYPE_DICT,
    STAGE0,
    STAGE1,
    STAGE2,
    STAGE3,
)
from pearl.engine import Event, EventGrouping
from pearl.interpolate import restricted_cubic_spline_var, restricted_quadratic_spline_var
from pearl.parameters import Parameters
from pearl.sample import draw_from_trunc_norm


def add_id(population: pd.DataFrame) -> pd.DataFrame:
    """Add an id column to the population DataFrame.

    Parameters
    ----------
    population : pd.DataFrame
        Population Dataframe.

    Returns
    -------
    pd.DataFrame
        Population DataFrame with id column added.
    """
    population["id"] = np.array(range(population.index.size))
    population = population.set_index(["age_cat", "id"]).sort_index()
    return population


def add_age_categories(population: pd.DataFrame) -> pd.DataFrame:
    """
    Add an age_cat column corresponding to the decade age of the agent, truncated at a maximum
    age category of 7.

    Parameters
    ----------
    population : pd.DataFrame
        Population with an age column.

    Returns
    -------
    pd.DataFrame
        Population with age_cat column added.
    """
    population["age"] = np.floor(population["age"])
    population["age_cat"] = np.floor(population["age"] / 10)
    population.loc[population["age_cat"] > 7, "age_cat"] = 7
    population.loc[population["age_cat"] < 2, "age_cat"] = 2
    return population


# TODO combine these add_default_columns


def add_default_columns(population: pd.DataFrame) -> pd.DataFrame:
    """
    Add default values for columns necessary for simulation.

    Parameters
    ----------
    population : pd.DataFrame
        Population DataFrame to add default columns to.

    Returns
    -------
    pd.DataFrame
        Population with added default columns
    """
    # Add final columns used for calculations and output
    population["last_h1yy"] = population["h1yy"]
    population["last_init_sqrtcd4n"] = population["init_sqrtcd4n"]
    population["init_age"] = population["age"] - (2009 - population["h1yy"])
    population["n_lost"] = np.array(0, dtype="int32")
    population["years_out"] = np.array(0, dtype="int16")
    population["year_died"] = np.nan
    population["sqrtcd4n_exit"] = 0
    population["ltfu_year"] = np.array(0, dtype="int16")
    population["return_year"] = np.array(0, dtype="int16")
    population["intercept"] = 1.0
    population["year"] = np.array(2009, dtype="int16")

    return population


def add_default_columns_new(population: pd.DataFrame) -> pd.DataFrame:
    # Calculate time varying cd4 count and other needed variables
    population["last_h1yy"] = population["h1yy"]
    population["time_varying_sqrtcd4n"] = population["init_sqrtcd4n"]
    population["last_init_sqrtcd4n"] = population["init_sqrtcd4n"]
    population["init_age"] = population["age"]
    population["n_lost"] = 0
    population["years_out"] = 0
    population["year_died"] = np.nan
    population["sqrtcd4n_exit"] = 0
    population["ltfu_year"] = 0
    population["return_year"] = 0
    population["intercept"] = 1.0
    population["year"] = 2009
    return population


def delta_bmi(population: pd.DataFrame) -> pd.DataFrame:
    """Calculate the change in BMI for each agent.

    Parameters
    ----------
    population : pd.DataFrame
        Population Dataframe.

    Returns
    -------
    pd.DataFrame
        Population DataFrame with delta_bmi column added.
    """
    population["delta_bmi"] = population["post_art_bmi"] - population["pre_art_bmi"]
    return population


def add_multimorbidity(population: pd.DataFrame) -> pd.DataFrame:
    """Calculate the multimorbidity for each agent.

    Parameters
    ----------
    population : pd.DataFrame
        Population Dataframe.

    Returns
    -------
    pd.DataFrame
        Population DataFrame with mm column added.
    """
    population["mm"] = np.array(population[STAGE2 + STAGE3].sum(axis=1), dtype="int8")
    return population


def sort_alphabetically(population: pd.DataFrame) -> pd.DataFrame:
    """Sort columns alphabetically.

    Parameters
    ----------
    population : pd.DataFrame
        Population Dataframe.

    Returns
    -------
    pd.DataFrame
        Population DataFrame with columns sorted alphabetically.
    """
    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)
    return population


def cast_type(population: pd.DataFrame) -> pd.DataFrame:
    """Cast population columns to save memory.

    Parameters
    ----------
    population : pd.DataFrame
        Population Dataframe.

    Returns
    -------
    pd.DataFrame
        Type cast population Dataframe.
    """
    population = population.astype(POPULATION_TYPE_DICT)
    return population


class Status(Event):
    """Assign a status to the populaton."""

    def __init__(self, parameters: Parameters, status: int) -> None:
        """Store parameters and status.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        status : int
            Status defined in pearl.definitions
        """
        super().__init__(parameters)
        self.status = status

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Assign status to the population.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population DataFrame with status column added.
        """
        population["status"] = self.status
        return population


class SimulateAges(Event):
    """Simulate ages for the given popeulation size and conditions."""

    def __init__(
        self, parameters: Parameters, population_size: int, h1yy: Optional[bool] = None
    ) -> None:
        """

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        population_size : int
            Size of the population to simulate.
        h1yy : Optional[bool], optional
            Whether or not to simulate ages by h1yy, by default None
        """
        super().__init__(parameters)
        self.population_size = population_size
        self.h1yy = h1yy
        if self.h1yy:
            self.coeffs = self.parameters.age_by_h1yy.loc[h1yy]
        else:
            self.coeffs = self.parameters.age_in_2009

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Simulate ages.

        Parameters
        ----------
        population : pd.DataFrame
            This dataframe is ignored. It currently just serves to maintain the Dataframe in
            Dataframe out API.

        Returns
        -------
        pd.DataFrame
            Population of simulated ages.
        """
        # Draw population size of each normal from the binomial distribution
        pop_size_1 = self.random_state.binomial(
            self.population_size, self.coeffs.loc["lambda1", "estimate"]
        )
        pop_size_2 = self.population_size - pop_size_1

        # Draw ages from truncated normal
        ages_1 = draw_from_trunc_norm(
            18,
            85,
            self.coeffs.loc["mu1", "estimate"],
            self.coeffs.loc["sigma1", "estimate"],
            pop_size_1,
            self.random_state,
        )
        ages_2 = draw_from_trunc_norm(
            18,
            85,
            self.coeffs.loc["mu2", "estimate"],
            self.coeffs.loc["sigma2", "estimate"],
            pop_size_2,
            self.random_state,
        )
        ages = np.concatenate((ages_1, ages_2))
        assert ages.min() > 18
        assert ages.max() < 85
        population["age"] = np.array(ages)
        return population


class H1yy(Event):
    """Assign diagnosis date (H1yy) to the population."""

    def __init__(self, parameters: Parameters):
        """Store parameters and coefficients.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        super().__init__(parameters)
        self.coeffs = self.parameters.h1yy_by_age_2009

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Assign H1yy to the population.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population DataFrame with h1yy column added.
        """
        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        for age_cat, grouped in population.groupby("age_cat"):
            h1yy_data = self.coeffs.loc[age_cat].reset_index()
            population.loc[age_cat, "h1yy"] = self.random_state.choice(
                h1yy_data["h1yy"], size=len(grouped), p=h1yy_data["pct"]
            )

        # Reindex for group operation
        population["h1yy"] = population["h1yy"].astype(int)
        population = population.reset_index().set_index(["h1yy", "id"]).sort_index()

        return population


# TODO combine these sqrtCd4n classes into a single one


class SqrtCd4nInit(Event):
    """Assign initial sqrtCD4 counts to the population."""

    def __init__(self, parameters: Parameters):
        """Store parameters and coefficients.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        super().__init__(parameters)
        self.coeffs = self.parameters.cd4n_by_h1yy_2009

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Assign initial CD4 counts to the population.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with init_sqrtcd4n column added.
        """
        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        for h1yy, group in population.groupby(level=0):
            mu = self.coeffs.loc[(h1yy, "mu"), "estimate"]
            sigma = self.coeffs.loc[(h1yy, "sigma"), "estimate"]
            size = group.shape[0]
            sqrt_cd4n = draw_from_trunc_norm(
                0, np.sqrt(2000.0), mu, sigma, size, self.random_state
            )
            population.loc[(h1yy,), "init_sqrtcd4n"] = sqrt_cd4n
        population = population.reset_index().set_index("id").sort_index()

        return population


class SqrtCd4nNew(Event):
    """Assign sqrtCD4 counts to new agents."""

    def __init__(self, parameters: Parameters) -> None:
        """Store Parameters.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        super().__init__(parameters)

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Assign sqrtCD4 counts to new agents.

        Parameters
        ----------
        population : pd.DataFrame
            Population Datafame.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with sqrtcd4n column added.
        """
        population = population.reset_index()
        unique_h1yy = population["h1yy"].unique()
        population["init_sqrtcd4n"] = 0.0
        for h1yy in unique_h1yy:
            mu = self.parameters.cd4n_by_h1yy.loc[(h1yy, "mu"), "estimate"]
            sigma = self.parameters.cd4n_by_h1yy.loc[(h1yy, "sigma"), "estimate"]
            size = len(population[population["h1yy"] == h1yy]["init_sqrtcd4n"])
            sqrt_cd4n = draw_from_trunc_norm(
                0, np.sqrt(2000.0), mu, sigma, size, self.random_state
            )
            population.loc[population["h1yy"] == h1yy, "init_sqrtcd4n"] = sqrt_cd4n

        population = population.reset_index().set_index("id").sort_index()
        return population


class Cd4Increase(Event):
    """Calculate the increase in CD4 count for the population."""

    def __init__(self, parameters: Parameters):
        """Store parameters, knot coefficients, and coefficients.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        super().__init__(parameters)

        self.knots = self.parameters.cd4_increase_knots
        self.coeffs = self.parameters.cd4_increase.to_numpy(dtype=float)

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Increase CD4 count for the population.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with increased CD4 count.
        """
        pop = population.copy()
        # Calculate spline variables
        pop["time_from_h1yy"] = pop["year"] - pop["last_h1yy"]
        pop["time_from_h1yy_"] = restricted_quadratic_spline_var(
            pop["time_from_h1yy"].to_numpy(), self.knots.to_numpy(), 1
        )
        pop["time_from_h1yy__"] = restricted_quadratic_spline_var(
            pop["time_from_h1yy"].to_numpy(), self.knots.to_numpy(), 2
        )
        pop["time_from_h1yy___"] = restricted_quadratic_spline_var(
            pop["time_from_h1yy"].to_numpy(), self.knots.to_numpy(), 3
        )

        # Calculate CD4 Category Variables
        pop["cd4_cat_349"] = (
            pop["last_init_sqrtcd4n"].ge(np.sqrt(200.0))
            & pop["last_init_sqrtcd4n"].lt(np.sqrt(350.0))
        ).astype(int)
        pop["cd4_cat_499"] = (
            pop["last_init_sqrtcd4n"].ge(np.sqrt(350.0))
            & pop["last_init_sqrtcd4n"].lt(np.sqrt(500.0))
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
        new_cd4 = np.matmul(pop_matrix, self.coeffs)

        new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
        population["time_varying_sqrtcd4n"] = np.array(new_cd4)
        return population


class PreArtBMI(Event):
    """Calculate pre-ART BMI for the population."""

    def __init__(self, parameters: Parameters) -> None:
        """Store parameters and coefficients.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        super().__init__(parameters)
        self.coeffs = self.parameters.pre_art_bmi.to_numpy(dtype=float)
        self.t_age = self.parameters.pre_art_bmi_age_knots.to_numpy(dtype=float)
        self.t_h1yy = parameters.pre_art_bmi_h1yy_knots.to_numpy(dtype=float)
        self.rse = self.parameters.pre_art_bmi_rse
        self.model = self.parameters.pre_art_bmi_model

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Calculate pre-art BMI and addd pre_art_bmi column

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe

        Returns
        -------
        pd.DataFrame
            Population Dataframe with pre_art_bmi column added.
        """
        pop = population.copy()
        pre_art_bmi = np.nan
        if self.model == 6:
            pop["age_"] = restricted_cubic_spline_var(pop["init_age"].to_numpy(), self.t_age, 1)
            pop["age__"] = restricted_cubic_spline_var(pop["init_age"].to_numpy(), self.t_age, 2)
            h1yy = pop["h1yy"].values
            pop["h1yy_"] = restricted_cubic_spline_var(h1yy, self.t_h1yy, 1)
            pop["h1yy__"] = restricted_cubic_spline_var(h1yy, self.t_h1yy, 2)
            pop_matrix = pop[
                ["init_age", "age_", "age__", "h1yy", "h1yy_", "h1yy__", "intercept"]
            ].to_numpy(dtype=float)
            log_pre_art_bmi = np.matmul(pop_matrix, self.coeffs)

        elif self.model == 5:
            pop["age_"] = restricted_cubic_spline_var(pop["init_age"].to_numpy(), self.t_age, 1)
            pop["age__"] = restricted_cubic_spline_var(pop["init_age"].to_numpy(), self.t_age, 2)
            pop_matrix = pop[["init_age", "age_", "age__", "h1yy", "intercept"]].to_numpy(
                dtype=float
            )
            log_pre_art_bmi = np.matmul(pop_matrix, self.coeffs)

        elif self.model == 3:
            pop_matrix = pop[["init_age", "h1yy", "intercept"]].to_numpy(dtype=float)
            log_pre_art_bmi = np.matmul(pop_matrix, self.coeffs)

        elif self.model == 2:
            pop["age_"] = (pop["init_age"] >= 30) & (pop["init_age"] < 40)
            pop["age__"] = (pop["init_age"] >= 40) & (pop["init_age"] < 50)
            pop["age___"] = (pop["init_age"] >= 50) & (pop["init_age"] < 60)
            pop["age____"] = pop["init_age"] >= 60
            h1yy = pop["h1yy"].values
            pop["h1yy_"] = restricted_cubic_spline_var(h1yy, self.t_h1yy, 1)
            pop["h1yy__"] = restricted_cubic_spline_var(h1yy, self.t_h1yy, 2)
            pop_matrix = pop[
                [
                    "age_",
                    "age__",
                    "age___",
                    "age____",
                    "h1yy",
                    "h1yy_",
                    "h1yy__",
                    "intercept",
                ]
            ].to_numpy(dtype=float)
            log_pre_art_bmi = np.matmul(pop_matrix, self.coeffs)

        elif self.model == 1:
            pop["age_"] = (pop["init_age"] >= 30) & (pop["init_age"] < 40)
            pop["age__"] = (pop["init_age"] >= 40) & (pop["init_age"] < 50)
            pop["age___"] = (pop["init_age"] >= 50) & (pop["init_age"] < 60)
            pop["age____"] = pop["init_age"] >= 60
            pop_matrix = pop[["age_", "age__", "age___", "age____", "h1yy", "intercept"]].to_numpy(
                dtype=float
            )
            log_pre_art_bmi = np.matmul(pop_matrix, self.coeffs)

        log_pre_art_bmi = log_pre_art_bmi.T[0]

        log_pre_art_bmi = draw_from_trunc_norm(
            np.log10(10),
            np.log10(65),
            log_pre_art_bmi,
            np.sqrt(self.rse),
            len(log_pre_art_bmi),
            self.random_state,
        )

        pre_art_bmi = 10.0**log_pre_art_bmi

        if self.parameters.sa_variables and "pre_art_bmi" in self.parameters.sa_variables:
            pre_art_bmi *= self.parameters.sa_scalars["pre_art_bmi"]

        population["pre_art_bmi"] = np.array(pre_art_bmi)
        return population


class PostArtBMI(Event):
    """Calculate Post-ART BMI for the population."""

    def __init__(self, parameters: Parameters) -> None:
        """Store parameters and coefficients."""
        super().__init__(parameters)
        self.coeffs = self.parameters.post_art_bmi.to_numpy(dtype=float)
        self.t_age = self.parameters.post_art_bmi_age_knots.to_numpy(dtype=float)
        self.t_pre_sqrt = self.parameters.post_art_bmi_pre_art_bmi_knots.to_numpy(dtype=float)
        self.t_sqrtcd4 = self.parameters.post_art_bmi_cd4_knots.to_numpy(dtype=float)
        self.t_sqrtcd4_post = self.parameters.post_art_bmi_cd4_post_knots.to_numpy(dtype=float)
        self.rse = self.parameters.post_art_bmi_rse

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Calculate post-ART BMI for the population and add post_art_bmi column.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with post_art_bmi column added.
        """
        pop = population.copy()
        # Calculate spline variables

        pop["age_"] = restricted_cubic_spline_var(pop["init_age"].to_numpy(), self.t_age, 1)
        pop["age__"] = restricted_cubic_spline_var(pop["init_age"].to_numpy(), self.t_age, 2)
        pop["pre_sqrt"] = pop["pre_art_bmi"] ** 0.5
        pop["pre_sqrt_"] = restricted_cubic_spline_var(
            pop["pre_sqrt"].to_numpy(), self.t_pre_sqrt, 1
        )
        pop["pre_sqrt__"] = restricted_cubic_spline_var(
            pop["pre_sqrt"].to_numpy(), self.t_pre_sqrt, 2
        )
        pop["sqrtcd4"] = pop["init_sqrtcd4n"]
        pop["sqrtcd4_"] = restricted_cubic_spline_var(pop["sqrtcd4"].to_numpy(), self.t_sqrtcd4, 1)
        pop["sqrtcd4__"] = restricted_cubic_spline_var(
            pop["sqrtcd4"].to_numpy(), self.t_sqrtcd4, 2
        )

        # Calculate cd4 count 2 years after art initiation and its spline terms
        pop_future = pop.copy().assign(age=pop["init_age"] + 2)
        pop_future["year"] = pop["h1yy"] + 2
        pop_future["age_cat"] = np.floor(pop_future["age"] / 10)
        pop_future.loc[pop_future["age_cat"] < 2, "age_cat"] = 2
        pop_future.loc[pop_future["age_cat"] > 7, "age_cat"] = 7
        # TODO fix
        pop["sqrtcd4_post"] = Cd4Increase(self.parameters)(pop_future)["time_varying_sqrtcd4n"]

        pop["sqrtcd4_post_"] = restricted_cubic_spline_var(
            pop["sqrtcd4_post"].to_numpy(), self.t_sqrtcd4_post, 1
        )
        pop["sqrtcd4_post__"] = restricted_cubic_spline_var(
            pop["sqrtcd4_post"].to_numpy(), self.t_sqrtcd4_post, 2
        )

        # Create the population matrix and perform the matrix multiplication
        pop_matrix = pop[
            [
                "init_age",
                "age_",
                "age__",
                "h1yy",
                "intercept",
                "pre_sqrt",
                "pre_sqrt_",
                "pre_sqrt__",
                "sqrtcd4",
                "sqrtcd4_",
                "sqrtcd4__",
                "sqrtcd4_post",
                "sqrtcd4_post_",
                "sqrtcd4_post__",
            ]
        ].to_numpy(dtype=float)
        sqrt_post_art_bmi = np.matmul(pop_matrix, self.coeffs)
        sqrt_post_art_bmi = sqrt_post_art_bmi.T[0]

        sqrt_post_art_bmi = draw_from_trunc_norm(
            np.sqrt(10),
            np.sqrt(65),
            sqrt_post_art_bmi,
            np.sqrt(self.rse),
            len(sqrt_post_art_bmi),
            self.random_state,
        )
        post_art_bmi = sqrt_post_art_bmi**2.0

        if self.parameters.sa_variables and "post_art_bmi" in self.parameters.sa_variables:
            post_art_bmi *= self.parameters.sa_scalars["post_art_bmi"]

        population["post_art_bmi"] = np.array(post_art_bmi)
        return population


class BasePopulation(Event):
    """Base population object."""

    def __init__(self, parameters: Parameters, population_size: int):
        """Store parameters, the population size, and the events to be applied.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        population_size : int
            Size of population to simulate.
        """
        super().__init__(parameters)
        self.population_size = population_size

        self.events = EventGrouping(
            [
                SimulateAges(self.parameters, self.population_size),
                add_age_categories,
                add_id,
                H1yy(self.parameters),
                SqrtCd4nInit(self.parameters),
                add_default_columns,
                Cd4Increase(self.parameters),
            ]
        )

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Return the base population.

        Parameters
        ----------
        population : pd.DataFrame
            This dataframe is ignored. It currently just serves to maintain the Dataframe in
            Dataframe out API.

        Returns
        -------
        pd.DataFrame
            Population Dataframe for the base population.
        """
        return self.events(population)


class Bmi(Event):
    """Calculate all BMI related variables."""

    def __init__(self, parameters: Parameters):
        """Store parameters and PreArtBMI, PostArtBMI, and delta_bmi events.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        super().__init__(parameters)
        self.events = EventGrouping(
            [PreArtBMI(self.parameters), PostArtBMI(self.parameters), delta_bmi]
        )

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Run all events.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with BMI related variables added.
        """
        return self.events(population)


class Comorbidity(Event):
    """Assign comorbidities for a random subset of the population based on each agents
    characteristics.
    """

    def __init__(self, parameters: Parameters, comorbidity: str, user: bool, new_init: bool):
        """Store parameters, comorbidity, whether or not the population is ART users, and whether
        or not the population is new initiators.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        comorbidity : str
            Comorbidity to assign.
        user : bool
            Whether or not the population is ART users.
        new_init : bool
            Whether or not the population is new initiators.
        """
        super().__init__(parameters)
        self.comorbidity = comorbidity
        self.new_init = new_init
        self.probability = (
            self.parameters.prev_inits_dict[self.comorbidity].values
            if new_init
            else self.parameters.prev_users_dict[self.comorbidity].values
        )
        self.user = user

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Assign comorbidity

        Parameters
        ----------
        population : pd.DataFrame
           Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with comorbidity assigned.
        """
        population[self.comorbidity] = (
            self.random_state.rand(len(population.index)) < self.probability
        ).astype(int)
        if self.user:
            population[f"t_{self.comorbidity}"] = np.array(0, dtype="int8")
        else:
            population[f"t_{self.comorbidity}"] = population[self.comorbidity]
        return population


class ApplyComorbidities(Event):
    """Apply all comorbidities sequentially"""

    def __init__(self, paramaters: Parameters, user: bool, new_init: bool) -> None:
        """_summary_

        Parameters
        ----------
        paramaters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        user : bool
            Whether or not the population is ART users.
        new_init : bool
            Whether or not the population is new initiators.
        """
        super().__init__(paramaters)
        self.user = user
        self.new_init = new_init

        self.events = EventGrouping(
            [
                Comorbidity(self.parameters, comorbidity, self.user, self.new_init)
                for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
            ]
        )

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Run the sequence of events.

        Parameters
        ----------
        population : pd.DataFrame
           Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with comorbidities assigned.
        """
        return self.events(population)


class Ltfu(Event):
    """Lost to follow up event."""

    def __init__(self, parameters: Parameters, population_size: int) -> None:
        """Store parameters and population size.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        population_size : int
            Size of the population.
        """
        super().__init__(parameters)
        self.population_size = population_size
        self.coeffs = self.parameters.years_out_of_care["years"]
        self.probability = self.parameters.years_out_of_care["probability"]

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Lose a subset of population to follow up.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe

        Returns
        -------
        pd.DataFrame
            Population Dataframe adjusted after loss to follow up.
        """
        years_out_of_care = self.random_state.choice(
            a=self.coeffs,
            size=self.population_size,
            p=self.probability,
        )

        population["sqrtcd4n_exit"] = population["time_varying_sqrtcd4n"]
        population["ltfu_year"] = 2009
        population["return_year"] = 2009 + years_out_of_care
        population["n_lost"] += 1

        return population


class YearsOutCare(Event):
    """Calculate years out of care for delayed start agents."""

    def __init__(self, parameters: Parameters) -> None:
        """Store parameters.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        """
        super().__init__(parameters)

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Generate number of years for delayed initiators to wait before beginning care and modify
        their start year accordingly

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with  years out of care adjustment.
        """

        delayed = population["status"] == DELAYED
        years_out_of_care = self.random_state.choice(
            a=self.parameters.years_out_of_care["years"],
            size=len(population.loc[delayed]),
            p=self.parameters.years_out_of_care["probability"],
        )
        population.loc[delayed, "h1yy"] = population.loc[delayed, "h1yy"] + years_out_of_care
        population.loc[delayed, "status"] = ART_NAIVE
        population = population[population["h1yy"] <= self.parameters.final_year].copy()
        return population


class NewAges(Event):
    """Simulate ages for new initiators."""

    def __init__(self, parameters: Parameters):
        """Store parameters.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        """
        super().__init__(parameters)

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Simulate ages by h1yy for new initiators.

        Parameters
        ----------
        population : pd.DataFrame
            Unused population Dataframe to keep the Dataframe in Dataframe out API.

        Returns
        -------
        pd.DataFrame
            Population Dataframe with simulated ages.
        """
        for h1yy in self.parameters.age_by_h1yy.index.levels[0]:
            grouped_pop = pd.DataFrame()
            n_initiators = self.parameters.n_new_agents.loc[h1yy, "art_initiators"]
            n_delayed = self.parameters.n_new_agents.loc[h1yy, "art_delayed"]
            grouped_pop["age"] = SimulateAges(self.parameters, n_initiators + n_delayed, h1yy)(
                pd.DataFrame([])
            )
            grouped_pop["h1yy"] = h1yy
            grouped_pop["status"] = ART_NAIVE
            delayed = self.random_state.choice(
                a=len(grouped_pop.index), size=n_delayed, replace=False
            )
            grouped_pop.loc[delayed, "status"] = DELAYED
            population = pd.concat([population, grouped_pop])
        return population


class UserPopInit(Event):
    """Population generator for ART users."""

    def __init__(self, parameters: Parameters, population_size: int):
        """Store parameters, population size, and events to be applied.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        population_size : int
            Size of population to simulate.
        """
        super().__init__(parameters)
        self.population_size = population_size

        self.events = EventGrouping(
            [
                BasePopulation(self.parameters, self.population_size),
                Status(self.parameters, ART_USER),
                ApplyComorbidities(self.parameters, user=True, new_init=False),
                add_multimorbidity,
                Bmi(self.parameters),
                sort_alphabetically,
                cast_type,
            ]
        )

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Generate population.

        Parameters
        ----------
        population : pd.DataFrame
            Unused population Dataframe to keep the Dataframe in Dataframe out API.

        Returns
        -------
        pd.DataFrame
            ART user population.
        """
        return self.events(population)


class NonUserPopInit(Event):
    """Population generator for ART non-users."""

    def __init__(self, parameters: Parameters, population_size: int) -> None:
        """Store parameters, population size, and events to be applied.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        population_size : int
            Size of population to simulate.
        """
        super().__init__(parameters)
        self.population_size = population_size

        self.events = EventGrouping(
            [
                BasePopulation(self.parameters, self.population_size),
                Ltfu(self.parameters, self.population_size),
                Status(self.parameters, ART_NONUSER),
                ApplyComorbidities(self.parameters, user=False, new_init=False),
                add_multimorbidity,
                Bmi(self.parameters),
                sort_alphabetically,
                cast_type,
            ]
        )

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Generate population.

        Parameters
        ----------
        population : pd.DataFrame
            Unused population Dataframe to keep the Dataframe in Dataframe out API.

        Returns
        -------
        pd.DataFrame
            ART non-user population.
        """
        return self.events(population)


class NewPopulation(Event):
    def __init__(self, parameters: Parameters) -> None:
        """Store parameters and events to be applied.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        """
        super().__init__(parameters)

        self.events = EventGrouping(
            [
                NewAges(self.parameters),
                YearsOutCare(self.parameters),
                add_age_categories,
                add_id,
                SqrtCd4nNew(self.parameters),
                add_default_columns_new,
                ApplyComorbidities(self.parameters, user=False, new_init=True),
                add_multimorbidity,
                Bmi(self.parameters),
                sort_alphabetically,
                cast_type,
            ]
        )

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Generate population.

        Parameters
        ----------
        population : pd.DataFrame
            Unused population Dataframe to keep the Dataframe in Dataframe out API.

        Returns
        -------
        pd.DataFrame
            New ART user population.
        """
        return self.events(population)


class PearlPopulation(Event):
    """Base PEARL population generator"""

    def __init__(self, parameters: Parameters):
        """Store parameters, as well as user, non-user, and new population generators.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        """
        super().__init__(parameters)

        self.user_pop = UserPopInit(self.parameters, self.parameters.n_initial_users)
        self.non_user_pop = NonUserPopInit(self.parameters, self.parameters.n_initial_nonusers)
        self.new_pop = NewPopulation(self.parameters)

    @override
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Return the full PEARL base population.

        Parameters
        ----------
        population : pd.DataFrame
            Unused population Dataframe to keep the Dataframe in Dataframe out API.

        Returns
        -------
        pd.DataFrame
            Base PEARL population.
        """
        user_pop = self.user_pop(pd.DataFrame([]))
        non_user_pop = self.non_user_pop(pd.DataFrame([]))
        new_pop = self.new_pop(pd.DataFrame([]))

        population = (
            pd.concat(
                [
                    user_pop,
                    non_user_pop,
                    new_pop,
                ]
            )
            .fillna(0)
            .drop(columns=["index"])
        )
        population = population.reset_index()
        population["id"] = np.array(range(population.index.size))
        population = population.set_index(["id"])
        return population

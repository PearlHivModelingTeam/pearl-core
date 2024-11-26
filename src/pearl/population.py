import numpy as np
import pandas as pd

from pearl.definitions import (
    ART_NONUSER,
    ART_USER,
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
    population["id"] = np.array(range(population.index.size))
    population = population.set_index(["age_cat", "id"]).sort_index()

    return population


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


def delta_bmi(population):
    population["delta_bmi"] = population["post_art_bmi"] - population["pre_art_bmi"]


def add_multimorbidity(population: pd.DataFrame) -> pd.DataFrame:
    population["mm"] = np.array(population[STAGE2 + STAGE3].sum(axis=1), dtype="int8")
    return population


def sort_alphabetically(population: pd.DataFrame) -> pd.DataFrame:
    # Sort columns alphabetically
    population = population.reindex(sorted(population), axis=1)
    return population


def cast_type(population: pd.DataFrame) -> pd.DataFrame:
    population = population.astype(POPULATION_TYPE_DICT)
    return population


class Status(Event):
    def __init__(self, parameters, status: str):
        super().__init__(parameters)
        self.status = status

    def __call__(self, population):
        population["status"] = self.status
        return population


class SimulateAges(Event):
    def __init__(self, parameters: Parameters, population_size: int) -> None:
        super().__init__(parameters)
        self.population_size = population_size
        self.coeffs = self.parameters.age_in_2009

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
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
    def __init__(self, parameters):
        super().__init__(parameters)
        self.coeffs = self.parameters.h1yy_by_age_2009

    def __call__(self, population):
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


class SqrtCd4nInit(Event):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.coeffs = self.parameters.cd4n_by_h1yy_2009

    def __call__(self, population):
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


class Cd4Increase(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.knots = self.parameters.cd4_increase_knots
        self.coeffs = self.parameters.cd4_increase.to_numpy(dtype=float)

    def __call__(self, population):
        # Calculate spline variables
        population["time_from_h1yy"] = population["year"] - population["last_h1yy"]
        population["time_from_h1yy_"] = restricted_quadratic_spline_var(
            population["time_from_h1yy"].to_numpy(), self.knots.to_numpy(), 1
        )
        population["time_from_h1yy__"] = restricted_quadratic_spline_var(
            population["time_from_h1yy"].to_numpy(), self.knots.to_numpy(), 2
        )
        population["time_from_h1yy___"] = restricted_quadratic_spline_var(
            population["time_from_h1yy"].to_numpy(), self.knots.to_numpy(), 3
        )

        # Calculate CD4 Category Variables
        population["cd4_cat_349"] = (
            population["last_init_sqrtcd4n"].ge(np.sqrt(200.0))
            & population["last_init_sqrtcd4n"].lt(np.sqrt(350.0))
        ).astype(int)
        population["cd4_cat_499"] = (
            population["last_init_sqrtcd4n"].ge(np.sqrt(350.0))
            & population["last_init_sqrtcd4n"].lt(np.sqrt(500.0))
        ).astype(int)
        population["cd4_cat_500"] = population["last_init_sqrtcd4n"].ge(np.sqrt(500.0)).astype(int)

        # Create cross term variables
        population["timecd4cat349_"] = population["time_from_h1yy_"] * population["cd4_cat_349"]
        population["timecd4cat499_"] = population["time_from_h1yy_"] * population["cd4_cat_499"]
        population["timecd4cat500_"] = population["time_from_h1yy_"] * population["cd4_cat_500"]
        population["timecd4cat349__"] = population["time_from_h1yy__"] * population["cd4_cat_349"]
        population["timecd4cat499__"] = population["time_from_h1yy__"] * population["cd4_cat_499"]
        population["timecd4cat500__"] = population["time_from_h1yy__"] * population["cd4_cat_500"]
        population["timecd4cat349___"] = (
            population["time_from_h1yy___"] * population["cd4_cat_349"]
        )
        population["timecd4cat499___"] = (
            population["time_from_h1yy___"] * population["cd4_cat_499"]
        )
        population["timecd4cat500___"] = (
            population["time_from_h1yy___"] * population["cd4_cat_500"]
        )

        # Create numpy matrix
        pop_matrix = population[
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
    def __init__(self, parameters):
        super().__init__(parameters)
        self.coeffs = self.parameters.pre_art_bmi.to_numpy(dtype=float)
        self.t_age = self.parameters.pre_art_bmi_age_knots.to_numpy(dtype=float)
        self.t_h1yy = parameters.pre_art_bmi_h1yy_knots.to_numpy(dtype=float)
        self.rse = self.parameters.pre_art_bmi_rse
        self.model = self.parameters.pre_art_bmi_model

    def __call__(self, population):
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
    def __init__(self, parameters):
        super().__init__(parameters)
        self.coeffs = self.parameters.post_art_bmi.to_numpy(dtype=float)
        self.t_age = self.parameters.post_art_bmi_age_knots.to_numpy(dtype=float)
        self.t_pre_sqrt = self.parameters.post_art_bmi_pre_art_bmi_knots.to_numpy(dtype=float)
        self.t_sqrtcd4 = self.parameters.post_art_bmi_cd4_knots.to_numpy(dtype=float)
        self.t_sqrtcd4_post = self.parameters.post_art_bmi_cd4_post_knots.to_numpy(dtype=float)
        self.rse = self.parameters.post_art_bmi_rse

    def __call__(self, population):
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
        pop["sqrtcd4_post"] = Cd4Increase(self.parameters)(pop_future)

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
    def __init__(self, parameters: Parameters, population_size: int):
        super().__init__(parameters)
        self.population_size = population_size

        self.events = EventGrouping(
            [
                SimulateAges(self.parameters, self.population_size),
                add_age_categories,
                H1yy(self.parameters),
                SqrtCd4nInit(self.parameters),
                add_default_columns,
                Cd4Increase(self.parameters),
            ]
        )

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        return self.events(population)


class Bmi(Event):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.events = [PreArtBMI(self.parameters), PostArtBMI(self.parameters), delta_bmi]

    def __call__(self, population):
        return self.events(population)


class Comorbidity(Event):
    def __init__(self, parameters, comorbidity: str, user: bool, new_init: bool):
        super().__init__(parameters)
        self.comorbidity = comorbidity
        self.new_init = new_init
        self.probability = (
            self.parameters.prev_users_dict[self.comorbidity].values
            if new_init
            else self.parameters.prev_inits_dict[self.comorbidity].values
        )
        self.user = user

    def __call__(self, population):
        population[self.condition] = (
            self.random_state.rand(len(population.index)) < self.probability
        ).astype(int)
        if self.user:
            population[f"t_{self.condition}"] = np.array(0, dtype="int8")
        else:
            population[f"t_{self.condition}"] = population[self.condition]
        return population


class ApplyComorbidities(Event):
    def __init__(self, events, user: bool):
        super().__init__(events)
        self.user = user

        self.events = EventGrouping(
            [
                Comorbidity(self.parameters, comorbidity, self.user)
                for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
            ]
        )

    def __call__(self, population):
        return self.events(population)


class Ltfu(Event):
    def __init__(self, parameters, population_size: int):
        super().__init__(parameters)
        self.population_size = population_size
        self.coeffs = self.parameters.years_out_of_care["years"]
        self.probability = self.parameters.years_out_of_care["probability"]

    def __call__(self, population):
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


class UserPopInit(Event):
    def __init__(self, parameters: Parameters, population_size: int):
        super().__init__(parameters)
        self.population_size = population_size

        self.events = EventGrouping(
            [
                BasePopulation(self.parameters, self.population_size),
                Status(self.parameters, ART_USER),
                Bmi(self.parameters),
                ApplyComorbidities(self.parameters, user=True),
                add_multimorbidity,
                sort_alphabetically,
                cast_type,
            ]
        )

    def __call__(self, population):
        return self.events(population)


class NonUserPopInit(Event):
    def __init__(self, parameters, population_size: int):
        super().__init__(parameters)
        self.population_size = population_size

        self.events = EventGrouping(
            [
                BasePopulation(self.parameters, self.population_size),
                Ltfu(self.parameters, self.population_size),
                Status(self.parameters, ART_NONUSER),
                Bmi(self.parameters),
                ApplyComorbidities(self.parameters, user=True),
                add_multimorbidity,
                sort_alphabetically,
                cast_type,
            ]
        )

    def __call__(self, population):
        return super().__call__(population)


class NewPopulation(Event):
    def __init__(self, parameters):
        super().__init__(parameters)

    def __call__(self, population):
        return super().__call__(population)

"""
Parameters class that stores all parameters needed to run the PEARL model.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from pearl.definitions import ROOT_DIR, STAGE0, STAGE1, STAGE2, STAGE3


class Parameters:
    """This class holds all the parameters needed for PEARL to run."""

    def __init__(
        self,
        output_folder: Path,
        replication: int,
        group_name: str,
        new_dx: str,
        start_year: int,
        final_year: int,
        mortality_model: str,
        mortality_threshold_flag: bool,
        idu_threshold: str,
        seed: int,
        history: Optional[List[str]] = None,
        final_state: bool = False,
        ignore_columns: Optional[List[str]] = None,
        bmi_intervention_scenario: int = 0,
        bmi_intervention_start_year: int = 2020,
        bmi_intervention_end_year: int = 2030,
        bmi_intervention_coverage: float = 1.0,
        bmi_intervention_effectiveness: float = 1.0,
        sa_variables: Optional[list[str]] = None,
    ):
        """
        Takes the path to the parameters.h5 file, the path to the folder containing rerun data
        if the run is a rerun, the output folder, the group name, a flag indicating if the
        simulation is for aim 2, a flag indicating whether to record detailed comorbidity
        information, the type of new_dx parameter to use, the final year of the model, the
        mortality model to use, whether to use a mortality threshold, verbosity, the sensitivity
        analysis dict, the classic sensitivity analysis dict, and the aim 2 sensitivity
        analysis dict.

        Parameters
        ----------
        output_folder : Path
            Folder to write simulation outputs to.
        replication : int
            replication number
        group_name : str
            Subpopulation name from [msm_white_male, msm_black_male, msm_hisp_male, idu_white_male,
            idu_black_male, idu_hisp_male, idu_white_female, idu_black_female, idu_hisp_female,
            het_white_male, het_black_male, het_hisp_male, het_white_female, het_black_female,
            het_hisp_female].
        new_dx : str
            new diagnosis model from [base, ehe].
        start_year : int
            Start year of simulation. Default is 2009.
        final_year : int
            Final year of simulation. The simulation will run from 2009 until the final year.
        mortality_model : str
            Which mortality model to run from [by_sex_race_risk, by_sex_race, by_sex, overall]
        mortality_threshold_flag : bool
            To use the mortality threshold or not.
        idu_threshold : str
            IDU threshold from [2x, 5x, 10x]
        seed : int
            Value for random number generation seeding.
        history: bool
            Whether or not to store history
        final_state: bool
            Whether or not to only store final state
        ignore_columns: list[str]
            List of columns to ignore when storing history
        bmi_intervention_scenario : int, optional
            BMI intervention to apply from [0 for no intervention, or 1, 2, 3], by default 0
        bmi_intervention_start_year : int, optional
            Year to start BMI intervention, by default 2020
        bmi_intervention_end_year : int, optional
            Year to end BMI intervention, by default 2030
        bmi_intervention_coverage : float, optional
            Probability of eligible population that receives BMI intervention between 0 and 1
            , by default 1.0
        bmi_intervention_effectiveness : float, optional
            Efficacy of BMI intervention for those that do receive it between 0 and 1
            , by default 1.0
        sa_variables : list[str]
            variables for sensitivity analysis
        Raises
        ------
        ValueError
            Raises value error if inputs are outside of the described acceptable values.
        """

        # check to ensure a proper group_name is provided
        if group_name not in [
            "msm_white_male",
            "msm_black_male",
            "msm_hisp_male",
            "idu_white_male",
            "idu_black_male",
            "idu_hisp_male",
            "idu_white_female",
            "idu_black_female",
            "idu_hisp_female",
            "het_white_male",
            "het_black_male",
            "het_hisp_male",
            "het_white_female",
            "het_black_female",
            "het_hisp_female",
        ]:
            raise ValueError("group_name not supported")

        # Save inputs as class attributes
        self.parameters_path = ROOT_DIR / "parameter_weights/parameters.h5"
        """Path to the parameters file for the PEARL model."""
        self.output_folder = output_folder
        """File path to the folder where PEARL outputs will be saved."""
        self.replication = replication
        """Replication number for the model run."""
        self.group_name = group_name
        """Group name for the model run."""
        self.new_dx_val = new_dx
        """Diagnosis model to use for the model run."""
        self.start_year = start_year
        """Start year of the model run."""
        self.final_year = final_year
        """Final year of the model run."""
        self.year = start_year
        """Current year of the model run, initialized to start_year."""
        self.mortality_model = mortality_model
        """Mortality model to use for the model run."""
        self.mortality_threshold_flag = mortality_threshold_flag
        """Mortality threshold flag for the model run."""
        self.idu_threshold = idu_threshold
        """IDU threshold for the model run."""
        self.seed = seed
        """Random seed for the model run."""
        self.random_state = np.random.RandomState(seed=seed)
        """Random state object for the model run, initialized with seed using 
        np.random.RandomState."""
        self.init_random_state = np.random.RandomState(seed=replication)
        """Random state object for parameter initialization, initialized with replication number 
        using np.random.RandomState."""
        self.history = history
        """Whether or not to store history."""
        self.final_state = final_state
        """Whether or not to only store final state."""
        self.ignore_columns = ignore_columns
        """Columns to ignore when storing history."""
        self.bmi_intervention_scenario = bmi_intervention_scenario
        """BMI intervention scenario to apply for the model run."""
        self.bmi_intervention_start_year = bmi_intervention_start_year
        """BMI intervention start year for the model run."""
        self.bmi_intervention_end_year = bmi_intervention_end_year
        """Bmi intervention end year for the model run."""
        self.bmi_intervention_coverage = bmi_intervention_coverage
        """BMI intervention coverage for the model run."""
        self.bmi_intervention_effectiveness = bmi_intervention_effectiveness
        """BMI intervention effectiveness for the model run."""
        self.sa_variables = sa_variables
        """Sensitivity analysis variables for the model run."""

        # 2009 population
        self.on_art_2009 = pd.read_hdf(self.parameters_path, "on_art_2009").loc[group_name]
        """Parameter for number of people on ART in 2009 for given group."""
        self.age_in_2009 = pd.read_hdf(self.parameters_path, "age_in_2009").loc[group_name]
        """Parameter for age distribution of people in 2009 for given group."""
        self.h1yy_by_age_2009 = pd.read_hdf(self.parameters_path, "h1yy_by_age_2009").loc[
            group_name
        ]
        """Paramters for year of HIV diagnosis by age in 2009 for given group."""
        self.cd4n_by_h1yy_2009 = pd.read_hdf(self.parameters_path, "cd4n_by_h1yy_2009").loc[
            group_name
        ]
        """Parameters for CD4 count by year of HIV diagnosis in 2009 for given group."""

        # New initiator statistics
        self.linkage_to_care = pd.read_hdf(self.parameters_path, "linkage_to_care").loc[group_name]
        """Parameter for linkage to care for given group."""
        self.age_by_h1yy = pd.read_hdf(self.parameters_path, "age_by_h1yy").loc[group_name]
        """Parameter for age by year of HIV diagnosis for given group."""
        self.cd4n_by_h1yy = pd.read_hdf(self.parameters_path, "cd4n_by_h1yy").loc[group_name]
        """Parameter for CD4 count by year of HIV diagnosis for given group."""
        # Choose new ART initiator model
        self.new_dx: pd.DataFrame
        """Parameter for new ART initiators. Chosen based on new_dx input, either "base" or "ehe" 
        for given group."""
        if new_dx == "base":
            self.new_dx = pd.read_hdf(self.parameters_path, "new_dx").loc[group_name]
        elif new_dx == "ehe":
            self.new_dx = pd.read_hdf(self.parameters_path, "new_dx_ehe").loc[group_name]
        else:
            raise ValueError("Invalid new diagnosis file specified")
        # Choose mortality model
        mortality_model_str: str
        """Mortality model for the run."""
        if mortality_model == "by_sex_race_risk":
            mortality_model_str = ""
        else:
            mortality_model_str = "_" + mortality_model

        if (mortality_model != "by_sex_race_risk") and (
            mortality_model != "by_sex_race_risk_2015" and (idu_threshold != "2x")
        ):
            raise ValueError(
                "Alternative mortality models with idu threshold changes is not implemented"
            )

        # Mortality In Care
        self.mortality_in_care = pd.read_hdf(
            self.parameters_path, f"mortality_in_care{mortality_model_str}"
        ).loc[group_name]
        """Parameter for mortality in care for given group. Chosen based on mortality_model 
        input."""
        self.mortality_in_care_age = pd.read_hdf(
            self.parameters_path, f"mortality_in_care_age{mortality_model_str}"
        ).loc[group_name]
        """Parameter for mortality in care by age for given group. Chosen based on mortality_model 
        input."""
        self.mortality_in_care_sqrtcd4 = pd.read_hdf(
            self.parameters_path, f"mortality_in_care_sqrtcd4{mortality_model_str}"
        ).loc[group_name]
        """Parameter for mortality in care by sqrt CD4 count for given group. 
        Chosen based on mortality_model input."""
        self.mortality_in_care_vcov = pd.read_hdf(
            self.parameters_path, "mortality_in_care_vcov"
        ).loc[group_name]
        """Parameter for variance-covariance matrix for mortality in care for given group."""

        # Mortality Out Of Care
        self.mortality_out_care = pd.read_hdf(
            self.parameters_path, f"mortality_out_care{mortality_model_str}"
        ).loc[group_name]
        """Parameter for mortality out of care for given group. Chosen based on mortality_model 
        input."""
        self.mortality_out_care_age = pd.read_hdf(
            self.parameters_path, f"mortality_out_care_age{mortality_model_str}"
        ).loc[group_name]
        """Parameter for mortality out of care by age for given group. Chosen based on 
        mortality_model input."""
        self.mortality_out_care_tv_sqrtcd4 = pd.read_hdf(
            self.parameters_path, f"mortality_out_care_tv_sqrtcd4{mortality_model_str}"
        ).loc[group_name]
        """Parameter for mortality out of care by time-varying sqrt CD4 count for given group. 
        Chosen based on mortality_model input."""
        self.mortality_out_care_vcov = pd.read_hdf(
            self.parameters_path, "mortality_out_care_vcov"
        ).loc[group_name]
        """Parameter for variance-covariance matrix for mortality out of care for given group."""

        # Mortality Threshold
        self.mortality_threshold: pd.DataFrame
        """Parameter for mortality threshold for given group. Chosen based on idu_threshold input. 
        If idu_threshold is not 2x, then the mortality threshold is dependent on the idu_threshold. 
        If idu_threshold is 2x, then the mortality threshold is dependent on the mortality_model."""
        if idu_threshold != "2x":
            self.mortality_threshold = pd.read_hdf(
                self.parameters_path, f"mortality_threshold_idu_{idu_threshold}"
            ).loc[group_name]
        else:
            self.mortality_threshold = pd.read_hdf(
                self.parameters_path, f"mortality_threshold{mortality_model_str}"
            ).loc[group_name]

        # Loss To Follow Up
        self.loss_to_follow_up = pd.read_hdf(self.parameters_path, "loss_to_follow_up").loc[
            group_name
        ]
        """Parameter for loss to follow up for given group."""
        self.ltfu_knots = pd.read_hdf(self.parameters_path, "ltfu_knots").loc[group_name]
        """Parameter for loss to follow up knots for given group."""
        self.loss_to_follow_up_vcov = pd.read_hdf(
            self.parameters_path, "loss_to_follow_up_vcov"
        ).loc[group_name]
        """Parameter for variance-covariance matrix for loss to follow up for given group."""

        # Cd4 Increase
        self.cd4_increase = pd.read_hdf(self.parameters_path, "cd4_increase").loc[group_name]
        """Parameter for CD4 increase for given group."""
        self.cd4_increase_vcov = pd.read_hdf(self.parameters_path, "cd4_increase_vcov").loc[
            group_name
        ]
        """Parameter for variance-covariance matrix for CD4 increase for given group."""
        self.cd4_increase_knots_age = pd.read_hdf(
            self.parameters_path, "cd4_increase_knots_age"
        ).loc[group_name]
        """Parameter for CD4 increase knots by age for given group."""
        self.cd4_increase_knots_cd4_init = pd.read_hdf(
            self.parameters_path, "cd4_increase_knots_cd4_init"
        ).loc[group_name]
        """Parameter for CD4 increase knots by initial CD4 count for given group."""
        self.cd4_increase_knots_time_from_h1yy = pd.read_hdf(
            self.parameters_path, "cd4_increase_knots_time_from_h1yy"
        ).loc[group_name]
        """Parameter for CD4 increase knots by time from HIV diagnosis for given group."""

        # Cd4 Decrease
        self.cd4_decrease = pd.read_hdf(self.parameters_path, "cd4_decrease").loc["all"]
        """Parameter for CD4 decrease for all groups."""
        self.cd4_decrease_vcov = pd.read_hdf(self.parameters_path, "cd4_decrease_vcov")
        """Parameter for variance-covariance matrix for CD4 decrease for all groups."""

        # Years out of Care
        self.years_out_of_care = pd.read_hdf(self.parameters_path, "years_out_of_care")
        """Parameter for years out of care for all groups."""

        # BMI
        self.pre_art_bmi = pd.read_hdf(self.parameters_path, "pre_art_bmi").loc[group_name]
        """Parameter for pre-ART BMI for given group."""
        self.pre_art_bmi_model = (
            pd.read_hdf(self.parameters_path, "pre_art_bmi_model").loc[group_name].values[0]
        )
        """Parameter for pre-ART BMI model for given group."""
        self.pre_art_bmi_age_knots = pd.read_hdf(
            self.parameters_path, "pre_art_bmi_age_knots"
        ).loc[group_name]
        """Parameter for pre-ART BMI age knots for given group."""
        self.pre_art_bmi_h1yy_knots = pd.read_hdf(
            self.parameters_path, "pre_art_bmi_h1yy_knots"
        ).loc[group_name]
        """Parameter for pre-ART BMI year of HIV diagnosis knots for given group."""
        self.pre_art_bmi_rse = (
            pd.read_hdf(self.parameters_path, "pre_art_bmi_rse").loc[group_name].values[0]
        )
        """Parameter for pre-ART BMI residual standard error for given group."""
        self.post_art_bmi = pd.read_hdf(self.parameters_path, "post_art_bmi").loc[group_name]
        """Parameter for post-ART BMI for given group."""
        self.post_art_bmi_age_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_age_knots"
        ).loc[group_name]
        """Parameter for post-ART BMI age knots for given group."""
        self.post_art_bmi_pre_art_bmi_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_pre_art_bmi_knots"
        ).loc[group_name]
        """Parameter for post-ART BMI pre-ART BMI knots for given group."""
        self.post_art_bmi_cd4_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_cd4_knots"
        ).loc[group_name]
        """Parameter for post-ART BMI CD4 count knots for given group."""
        self.post_art_bmi_cd4_post_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_cd4_post_knots"
        ).loc[group_name]
        """Parameter for post-ART BMI CD4 count after ART initiation knots for given group."""
        self.post_art_bmi_rse = (
            pd.read_hdf(self.parameters_path, "post_art_bmi_rse").loc[group_name].values[0]
        )
        """Parameter for post-ART BMI residual standard error for given group."""

        # BMI Intervention parameters
        if bmi_intervention_scenario not in [0, 1, 2, 3]:
            raise ValueError("bmi_intervention_scenario values only supported for 0, 1, 2, and 3")
        self.bmi_intervention_scenario = bmi_intervention_scenario
        """Parameter for BMI intervention scenario to apply for given group. Must be 0, 1, 2, or 3.
        0 corresponds to no intervention, 1 corresponds to a lifestyle intervention for those with 
        BMI over 25, 2 corresponds to a lifestyle intervention for those with BMI over 30, 
        and 3 corresponds to a pharmacological intervention for those with BMI over 30."""
        self.bmi_intervention_start_year = bmi_intervention_start_year
        """Parameter for BMI intervention start year for given group."""
        self.bmi_intervention_end_year = bmi_intervention_end_year
        """Parameter for BMI intervention end year for given group."""
        if bmi_intervention_coverage < 0 or bmi_intervention_coverage > 1:
            raise ValueError("bmi_intervention_coverage must be between 0 and 1 inclusive")
        self.bmi_intervention_coverage = bmi_intervention_coverage
        """Parameter for BMI intervention coverage for given group. Must be between 0 and 1 
        inclusive. Represents the proportion of eligible population that receives the BMI 
        intervention."""
        if bmi_intervention_effectiveness < 0 or bmi_intervention_effectiveness > 1:
            raise ValueError("bmi_intervention_effectiveness must be between 0 and 1 inclusive")
        self.bmi_intervention_effectiveness = bmi_intervention_effectiveness
        """Parameter for BMI intervention effectiveness for given group. Must be between 0 and 1 
        inclusive. Represents the proportion of eligible population that receives the BMI 
        intervention."""

        # Comorbidities
        self.prev_users_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_prev_users").loc[
                group_name
            ]
            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
        }
        """Parameter for prevalence of comorbidity among users for given group. Dictionary with 
        keys for each comorbidity and values as the prevalence of that comorbidity among users 
        for the given group."""
        self.prev_inits_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_prev_inits").loc[
                group_name
            ]
            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
        }
        """Parameter for prevalence of comorbidity among new initiators for given group. Dictionary 
        with keys for each comorbidity and values as the prevalence of that comorbidity among new 
        initiators for the given group."""
        self.comorbidity_coeff_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_coeff").loc[group_name]
            for comorbidity in STAGE1 + STAGE2 + STAGE3
        }
        """Parameter for coefficient for comorbidity in the CD4 decrease model for given group. 
        Dictionary with keys for each comorbidity and values as the coefficient for that 
        comorbidity in the CD4 decrease model for the given group."""
        self.delta_bmi_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_delta_bmi").loc[
                group_name
            ]
            for comorbidity in STAGE2 + STAGE3
        }
        """Parameter for change in BMI associated with comorbidity for given group. Dictionary with 
        keys for each comorbidity and values as the change in BMI associated with that comorbidity 
        for the given group."""
        self.post_art_bmi_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_post_art_bmi").loc[
                group_name
            ]
            for comorbidity in STAGE2 + STAGE3
        }
        """Parameter for post-ART BMI associated with comorbidity for given group. Dictionary with 
        keys for each comorbidity and values as the post-ART BMI associated with that comorbidity 
        for the given group."""

        # Aim 2 Mortality
        self.mortality_in_care_co = pd.read_hdf(self.parameters_path, "mortality_in_care_co").loc[
            group_name
        ]
        """Parameter for mortality in care for given group. Coefficients for the mortality in care 
        model for the given group."""
        self.mortality_in_care_post_art_bmi = pd.read_hdf(
            self.parameters_path, "mortality_in_care_post_art_bmi"
        ).loc[group_name]
        """Parameter for mortality in care for given group. Coefficients for the post-ART BMI variable
        in the mortality in care model for the given group."""
        self.mortality_out_care_co = pd.read_hdf(
            self.parameters_path, "mortality_out_care_co"
        ).loc[group_name]
        """Parameter for mortality out of care for given group. Coefficients for the mortality out 
        of care model for the given group."""
        self.mortality_out_care_post_art_bmi = pd.read_hdf(
            self.parameters_path, "mortality_out_care_post_art_bmi"
        ).loc[group_name]
        """Parameter for mortality out of care for given group. Coefficients for the post-ART BMI variable
        in the mortality out of care model for the given group."""

        # Year and age ranges
        self.AGES = np.arange(18, 87)
        """Parameter for age range of agents in the model. Minimum age is 18 and maximum age is 
        86."""
        self.AGE_CATS = np.arange(2, 8)
        """Parameter for age categories for agents in the model. Age categories are defined as 
        18-29, 30-39, 40-49, 50-59, 60-69, and 70-79."""
        self.SIMULATION_YEARS = np.arange(2010, final_year + 1)
        """Parameter for years of the simulation. Simulation runs from 2010 to final_year."""
        self.ALL_YEARS = np.arange(2000, final_year + 1)
        """Parameter for all years in the model. Range from 2000 to final_year."""
        self.INITIAL_YEARS = np.arange(2000, 2010)
        """Parameter for initial years of the model. Range from 2000 to 2009."""
        self.CD4_BINS = np.arange(2001)
        """Parameter for CD4 count bins for the model. Range from 0 to 2000."""

        # Sensitivity Analysis
        self.sa_variables = sa_variables
        """Parameter for sensitivity analysis variables. List of variables to include in 
        sensitivity analysis."""
        self.sa_scalars = {}
        """Parameter for sensitivity analysis scalars. Dictionary with keys for each variable 
        included in sensitivity analysis and values as the scalar to multiply that variable by for 
        the sensitivity analysis."""

        if self.sa_variables:
            for comorbidity in self.prev_users_dict:
                if f"{comorbidity}_prevalence_prev" in self.sa_variables:
                    self.sa_scalars[f"{comorbidity}_prevalence_prev"] = (
                        self.init_random_state.uniform(0.8, 1.2)
                    )
                    self.prev_users_dict[comorbidity] *= self.sa_scalars[
                        f"{comorbidity}_prevalence_prev"
                    ]

            for comorbidity in self.prev_inits_dict:
                if f"{comorbidity}_prevalence" in self.sa_variables:
                    self.sa_scalars[f"{comorbidity}_prevalence"] = self.init_random_state.uniform(
                        0.8, 1.2
                    )
                    self.prev_inits_dict[comorbidity] *= self.sa_scalars[
                        f"{comorbidity}_prevalence"
                    ]

            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3:
                if f"{comorbidity}_incidence" in self.sa_variables:
                    self.sa_scalars[f"{comorbidity}_incidence"] = self.init_random_state.uniform(
                        0.8, 1.2
                    )

            if "pre_art_bmi" in self.sa_variables:
                self.sa_scalars["pre_art_bmi"] = self.init_random_state.uniform(0.8, 1.2)

            if "post_art_bmi" in self.sa_variables:
                self.sa_scalars["post_art_bmi"] = self.init_random_state.uniform(0.8, 1.2)

            if "art_initiators" in self.sa_variables:
                self.sa_scalars["art_initiators"] = self.init_random_state.uniform(0.8, 1.2)

        # Draw a random value between predicted and 2018 predicted value for years greater than
        # 2018
        # TODO refactor this and save it in parameters
        self.age_by_h1yy["estimate"] = (
            self.random_state.rand(len(self.age_by_h1yy.index))
            * (self.age_by_h1yy["high_value"] - self.age_by_h1yy["low_value"])
        ) + self.age_by_h1yy["low_value"]

        self.cd4n_by_h1yy["estimate"] = (
            self.random_state.rand(len(self.cd4n_by_h1yy.index))
            * (self.cd4n_by_h1yy["high_value"] - self.cd4n_by_h1yy["low_value"])
        ) + self.cd4n_by_h1yy["low_value"]

        self.n_initial_users = self.on_art_2009.iloc[0]
        """Parameter for number of ART users in 2009 for given group, taken from on_art_2009 
        parameter."""

        self.n_initial_nonusers: int
        """Parameter for number of ART non-users in 2009 for given group. Calculated based on the 
        number of new ART initiators each year and the assumption that those not initiating ART in 
        the first few years of the model are the initial ART non-users."""

        self.n_new_agents: int
        """Parameter for number of new agents entering the model each year. Calculated based on the 
        number of new ART initiators each year and the number of new ART non-users each year."""

        # Simulate number of new art initiators and initial nonusers
        self.n_initial_nonusers, self.n_new_agents = self.simulate_new_dx()

        self.save_parameters()

    def save_parameters(self) -> None:
        """
        Save all parameters as a dataframe.
        """

        param_dict = {
            "replication": self.replication,
            "group": self.group_name,
            "new_dx": self.new_dx_val,
            "final_year": self.final_year,
            "mortality_model": self.mortality_model,
            "mortality_threshold_flag": self.mortality_threshold_flag,
            "idu_threshold": self.idu_threshold,
            "seed": self.seed,
            "bmi_intervention_scenario": self.bmi_intervention_scenario,
            "bmi_intervention_start_year": self.bmi_intervention_start_year,
            "bmi_intervention_end_year": self.bmi_intervention_end_year,
            "bmi_intervention_coverage": self.bmi_intervention_coverage,
            "bmi_intervention_effectiveness": self.bmi_intervention_effectiveness,
        }

        for scalar in self.sa_scalars:
            param_dict[scalar] = self.sa_scalars[scalar]

        self.param_dataframe = pd.DataFrame(param_dict, index=[0])

        if self.output_folder:
            self.param_dataframe.to_parquet(
                self.output_folder / "parameters.parquet", compression="zstd"
            )

    def simulate_new_dx(self) -> Tuple[int, pd.DataFrame]:
        """
        Return the number of ART non-users in 2009 as an integer and the number of agents entering
        the model each year as art users and non-users as a dataframe. Draw number of new diagnoses
        from a uniform distribution between upper and lower bounds. Calculate number of new art
        initiators by assuming a certain number link in the first year as estimated by a linear
        regression on CDC data, capped at 95%. We assume that 40% of the remaining population links
        to care over the next 3 years. We assume that 70% of those linking to care begin ART,
        rising to 85% in 2011 and 97% afterwards. We take the number of people not initiating ART
        2006 - 2009 in this calculation to be the out of care population size in 2009 for our
        simulation.

        Parameters
        ----------
        parameters : Parameters
            Parameter object with new_dx and linkage_to_care attributes.
        random_state : np.random.RandomState
            Random State object for random number sampling.

        Returns
        -------
        Tuple[int, pd.DataFrame]
            (number of ART non-users in 2009 as an integer, number of agents entering the model
            each year as art users and non-users as a dataframe)
        """
        new_dx = self.new_dx.copy()
        linkage_to_care = self.linkage_to_care

        # Draw new dx from a uniform distribution between upper and lower for 2016-final_year
        new_dx["n_dx"] = (
            new_dx["lower"] + (new_dx["upper"] - new_dx["lower"]) * self.random_state.uniform()
        )

        # Only a proportion of new diagnoses link to care and 40% of the remaining link
        # in the next 3 years
        new_dx["unlinked"] = new_dx["n_dx"] * (1 - linkage_to_care["link_prob"])
        new_dx["gardner_per_year"] = new_dx["unlinked"] * 0.4 / 3.0
        new_dx["year0"] = new_dx["n_dx"] * linkage_to_care["link_prob"]
        new_dx["year1"] = new_dx["gardner_per_year"].shift(1, fill_value=0)
        new_dx["year2"] = new_dx["gardner_per_year"].shift(2, fill_value=0)
        new_dx["year3"] = new_dx["gardner_per_year"].shift(3, fill_value=0)
        new_dx["total_linked"] = (
            new_dx["year0"] + new_dx["year1"] + new_dx["year2"] + new_dx["year3"]
        )

        # Proportion of those linked to care start ART
        new_dx["art_initiators"] = (new_dx["total_linked"] * linkage_to_care["art_prob"]).astype(
            int
        )
        new_dx["art_delayed"] = (
            new_dx["total_linked"] * (1 - linkage_to_care["art_prob"])
        ).astype(int)

        # TODO make the start and end dates here parametric
        # Count those not starting art 2006 - 2009 as initial ART nonusers
        n_initial_nonusers = new_dx.loc[np.arange(2006, 2010), "art_delayed"].sum()

        # Compile list of number of new agents to be introduced in the model
        new_agents = new_dx.loc[
            np.arange(2010, new_dx.index.max() + 1), ["art_initiators", "art_delayed"]
        ]

        if self.sa_variables and "art_initiators" in self.sa_variables:
            new_agents["art_initiators"] *= self.sa_scalars["art_initiators"]
            new_agents["art_delayed"] *= self.sa_scalars["art_initiators"]

            new_agents = new_agents.astype({"art_initiators": int, "art_delayed": int})

        return n_initial_nonusers, new_agents

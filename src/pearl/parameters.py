"""
Parameters class
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
        self.output_folder = output_folder
        self.replication = replication
        self.group_name = group_name
        self.new_dx_val = new_dx
        self.start_year = start_year
        self.final_year = final_year
        self.year = start_year
        self.mortality_model = mortality_model
        self.mortality_threshold_flag = mortality_threshold_flag
        self.idu_threshold = idu_threshold
        self.seed = seed
        self.random_state = np.random.RandomState(seed=seed)
        self.init_random_state = np.random.RandomState(seed=replication)
        self.ignore_columns = ignore_columns
        self.history = history
        self.final_state = final_state
        self.bmi_intervention_scenario = bmi_intervention_scenario
        self.bmi_intervention_start_year = bmi_intervention_start_year
        self.bmi_intervention_end_year = bmi_intervention_end_year
        self.bmi_intervention_coverage = bmi_intervention_coverage
        self.bmi_intervention_effectiveness = bmi_intervention_effectiveness
        self.sa_variables = sa_variables

        # 2009 population
        self.on_art_2009 = pd.read_hdf(self.parameters_path, "on_art_2009").loc[group_name]
        self.age_in_2009 = pd.read_hdf(self.parameters_path, "age_in_2009").loc[group_name]
        self.h1yy_by_age_2009 = pd.read_hdf(self.parameters_path, "h1yy_by_age_2009").loc[
            group_name
        ]
        self.cd4n_by_h1yy_2009 = pd.read_hdf(self.parameters_path, "cd4n_by_h1yy_2009").loc[
            group_name
        ]

        # New initiator statistics
        self.linkage_to_care = pd.read_hdf(self.parameters_path, "linkage_to_care").loc[group_name]
        self.age_by_h1yy = pd.read_hdf(self.parameters_path, "age_by_h1yy").loc[group_name]
        self.cd4n_by_h1yy = pd.read_hdf(self.parameters_path, "cd4n_by_h1yy").loc[group_name]
        # Choose new ART initiator model
        if new_dx == "base":
            self.new_dx = pd.read_hdf(self.parameters_path, "new_dx").loc[group_name]
        elif new_dx == "ehe":
            self.new_dx = pd.read_hdf(self.parameters_path, "new_dx_ehe").loc[group_name]
        else:
            raise ValueError("Invalid new diagnosis file specified")
        # Choose mortality model
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
        self.mortality_in_care_age = pd.read_hdf(
            self.parameters_path, f"mortality_in_care_age{mortality_model_str}"
        ).loc[group_name]
        self.mortality_in_care_sqrtcd4 = pd.read_hdf(
            self.parameters_path, f"mortality_in_care_sqrtcd4{mortality_model_str}"
        ).loc[group_name]
        self.mortality_in_care_vcov = pd.read_hdf(
            self.parameters_path, "mortality_in_care_vcov"
        ).loc[group_name]

        # Mortality Out Of Care
        self.mortality_out_care = pd.read_hdf(
            self.parameters_path, f"mortality_out_care{mortality_model_str}"
        ).loc[group_name]
        self.mortality_out_care_age = pd.read_hdf(
            self.parameters_path, f"mortality_out_care_age{mortality_model_str}"
        ).loc[group_name]
        self.mortality_out_care_tv_sqrtcd4 = pd.read_hdf(
            self.parameters_path, f"mortality_out_care_tv_sqrtcd4{mortality_model_str}"
        ).loc[group_name]
        self.mortality_out_care_vcov = pd.read_hdf(
            self.parameters_path, "mortality_out_care_vcov"
        ).loc[group_name]

        # Mortality Threshold
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
        self.ltfu_knots = pd.read_hdf(self.parameters_path, "ltfu_knots").loc[group_name]
        self.loss_to_follow_up_vcov = pd.read_hdf(
            self.parameters_path, "loss_to_follow_up_vcov"
        ).loc[group_name]

        # Cd4 Increase
        self.cd4_increase = pd.read_hdf(self.parameters_path, "cd4_increase").loc[group_name]
        self.cd4_increase_knots = pd.read_hdf(self.parameters_path, "cd4_increase_knots").loc[
            group_name
        ]
        self.cd4_increase_vcov = pd.read_hdf(self.parameters_path, "cd4_increase_vcov").loc[
            group_name
        ]

        # Cd4 Decrease
        self.cd4_decrease = pd.read_hdf(self.parameters_path, "cd4_decrease").loc["all"]
        self.cd4_decrease_vcov = pd.read_hdf(self.parameters_path, "cd4_decrease_vcov")

        # Years out of Care
        self.years_out_of_care = pd.read_hdf(self.parameters_path, "years_out_of_care")

        # BMI
        self.pre_art_bmi = pd.read_hdf(self.parameters_path, "pre_art_bmi").loc[group_name]
        self.pre_art_bmi_model = (
            pd.read_hdf(self.parameters_path, "pre_art_bmi_model").loc[group_name].values[0]
        )
        self.pre_art_bmi_age_knots = pd.read_hdf(
            self.parameters_path, "pre_art_bmi_age_knots"
        ).loc[group_name]
        self.pre_art_bmi_h1yy_knots = pd.read_hdf(
            self.parameters_path, "pre_art_bmi_h1yy_knots"
        ).loc[group_name]
        self.pre_art_bmi_rse = (
            pd.read_hdf(self.parameters_path, "pre_art_bmi_rse").loc[group_name].values[0]
        )
        self.post_art_bmi = pd.read_hdf(self.parameters_path, "post_art_bmi").loc[group_name]
        self.post_art_bmi_age_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_age_knots"
        ).loc[group_name]
        self.post_art_bmi_pre_art_bmi_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_pre_art_bmi_knots"
        ).loc[group_name]
        self.post_art_bmi_cd4_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_cd4_knots"
        ).loc[group_name]
        self.post_art_bmi_cd4_post_knots = pd.read_hdf(
            self.parameters_path, "post_art_bmi_cd4_post_knots"
        ).loc[group_name]
        self.post_art_bmi_rse = (
            pd.read_hdf(self.parameters_path, "post_art_bmi_rse").loc[group_name].values[0]
        )

        # BMI Intervention parameters
        if bmi_intervention_scenario not in [0, 1, 2, 3]:
            raise ValueError("bmi_intervention_scenario values only supported for 0, 1, 2, and 3")
        self.bmi_intervention_scenario = bmi_intervention_scenario
        self.bmi_intervention_start_year = bmi_intervention_start_year
        self.bmi_intervention_end_year = bmi_intervention_end_year
        if bmi_intervention_coverage < 0 or bmi_intervention_coverage > 1:
            raise ValueError("bmi_intervention_coverage must be between 0 and 1 inclusive")
        self.bmi_intervention_coverage = bmi_intervention_coverage
        if bmi_intervention_effectiveness < 0 or bmi_intervention_effectiveness > 1:
            raise ValueError("bmi_intervention_effectiveness must be between 0 and 1 inclusive")
        self.bmi_intervention_effectiveness = bmi_intervention_effectiveness

        # Comorbidities
        self.prev_users_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_prev_users").loc[
                group_name
            ]
            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
        }
        self.prev_inits_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_prev_inits").loc[
                group_name
            ]
            for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3
        }
        self.comorbidity_coeff_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_coeff").loc[group_name]
            for comorbidity in STAGE1 + STAGE2 + STAGE3
        }
        self.delta_bmi_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_delta_bmi").loc[
                group_name
            ]
            for comorbidity in STAGE2 + STAGE3
        }
        self.post_art_bmi_dict = {
            comorbidity: pd.read_hdf(self.parameters_path, f"{comorbidity}_post_art_bmi").loc[
                group_name
            ]
            for comorbidity in STAGE2 + STAGE3
        }

        # Aim 2 Mortality
        self.mortality_in_care_co = pd.read_hdf(self.parameters_path, "mortality_in_care_co").loc[
            group_name
        ]
        self.mortality_in_care_post_art_bmi = pd.read_hdf(
            self.parameters_path, "mortality_in_care_post_art_bmi"
        ).loc[group_name]
        self.mortality_out_care_co = pd.read_hdf(
            self.parameters_path, "mortality_out_care_co"
        ).loc[group_name]
        self.mortality_out_care_post_art_bmi = pd.read_hdf(
            self.parameters_path, "mortality_out_care_post_art_bmi"
        ).loc[group_name]

        # Year and age ranges
        self.AGES = np.arange(18, 87)
        self.AGE_CATS = np.arange(2, 8)
        self.SIMULATION_YEARS = np.arange(2010, final_year + 1)
        self.ALL_YEARS = np.arange(2000, final_year + 1)
        self.INITIAL_YEARS = np.arange(2000, 2010)
        self.CD4_BINS = np.arange(2001)

        # Sensitivity Analysis
        self.sa_variables = sa_variables
        self.sa_scalars = {}

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

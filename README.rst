===============
The PEARL Model
===============

The ProjEcting Age, multimoRbidity, and poLypharmacy (PEARL) model is an agent-based simulation 
model of persons living with HIV (PWH) using antiretroviral therapy (ART) in the US (2009 - 2030). 
Due to effective treatment, PWH accessing care in the US now have a life expectancy approaching the 
general population. As these people survive to older ages, the burden of multimorbidity and 
polypharmacy will change to reflect this older age distribution and non-HIV-related healthcare 
costs will become increasingly important in caring for these individuals. Since the relevant 
results vary greatly by demographic among PWH in the US, the PEARL model accounts race, sex, HIV 
risk group explicitly. Among these key populations, outcomes depend further upon age, age at ART 
initiation, CD4 count at ART initiation, current CD4 count, and BMI. For most of its machinery, the 
PEARL model utilizes data from th North American AIDS Cohort Collaboration on Research and Design 
(NA-ACCORD). The NA-ACCORD is comprised of data from over 20 cohorts, 200 sites, and 190,000 
HIV-infected participants, providing a data set that is widely representative of HIV care in 
North America. Data on overall PWH population size comes mostly from CDC surveillance data.

The PEARL model has been constructed to achieve the following:

**Aim 1:** To fill the gap in knowledge by projecting age distributions of PWH using ART in the US 
through 2030 broken down by key population.

**Aim 2:** To project the burden of multimorbidity and polypharmacy among PWH using ART in the US 
through 2030.

**Aim 3:** To project the annual costs of non-HIV-related healthcare for PWH using ART in the US 
through 2030.

==========================
Installation and First Run
==========================

Clone the repository onto your machine, enter the directory and install pearl::

    git clone https://github.com/PearlHivModelingTeam/pearl-core.git
    cd pearl-core
    pip install .

=======================
Development Environment
=======================

For development, and usage, we suggest using docker and vscode, with instructions outlined below:

^^^^^^^^^^^^^^^^^^^^^^
Step 1: Install VSCode
^^^^^^^^^^^^^^^^^^^^^^
1. Navigate to the `Visual Studio Code website <https://code.visualstudio.com/>`_.
2. Download the appropriate installer for your operating system (Windows, Linux, or macOS).
3. Run the installer and follow the on-screen instructions to install VSCode on your system.
4. After installation, launch VSCode.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 2: Install DevContainer Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. In VSCode, go to the Extensions view by clicking on the Extensions icon in the Activity Bar on 
the side of the window.
2. Search for "Dev Containers" in the Extensions view search bar.
3. Find the "Dev Containers" extension in the search results and click on the install button to 
install it.

You can also go to the extension's 
`homepage <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_ 
and 
`documentation page <https://code.visualstudio.com/docs/devcontainers/containers>`_ 
to find more details.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 3: Install Docker and Add Current Login User to Docker Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Follow the `official guide <https://docs.docker.com/get-docker/>`_ to install Docker. Don't forget 
the `post installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_.

If you are using `Visual Studio Code Remote - SSH <https://code.visualstudio.com/docs/remote/ssh>`_, 
then you only need to install Docker in the remote host, not your local computer. And the following steps should be run in the remote host.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 4: Open in DevContainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the 
"Dev Containers: Open Folder in Container..." command.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 5: Wait for Building the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. After opening the folder in a DevContainer, VSCode will start building the container. This 
process can take some time as it involves downloading necessary images and setting up the 
environment.

2. You can monitor the progress in the VSCode terminal.

3. Once the build process completes, you'll have a fully configured development environment in a 
container.

4. The next time you open the same dev container, it will be much faster, as it does not require 
building the image again.

-------------
Documentation
-------------
To power automatic documentation generation, we use `Sphinx <https://www.sphinx-doc.org/en/master/>`
with numpydoc and napoleon extensions. The documentation generates automatically as long as
docstrings are properly formatted as per numpydoc style.

-------
Testing
-------
To ensure that the package is working as intended, can run the test suit with:

``pytest tests``

TODO: add config details

^^^^^^^^^^^^^^^
``group_names``
^^^^^^^^^^^^^^^
A list of the sex, race, and hiv-acquisition groups to include in the simulation. 
Can be any number of 
```
['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
```

^^^^^^^^^^^^^^^^
``replications``
^^^^^^^^^^^^^^^^
Number of replications of each simulation to run with different seeds. Any positive integer.

^^^^^^^^^^
``new_dx``
^^^^^^^^^^
String indicating which set of parameters to use for number of new diagnoses. 
``base``, ``ehe``, ``sa``. 
The alternate models correspond to models used in some previous papers.

^^^^^^^^^^^^^^^^^^^
``mortality_model``
^^^^^^^^^^^^^^^^^^^
String corresponding to which model to use for mortality. 
``by_sex_race_risk``, ``by_sex_race``, ``by_sex``, ``overall``. 
These models are presented in the mortality paper.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``mortality_threshold_flag``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Flag indicating whether simulation should include the mortality threshold functionality. 0 or 1.

^^^^^^^^^^^^^^
``final_year``
^^^^^^^^^^^^^^
Year to end the simulation. Integer between 2010 and 2035.

^^^^^^^^^^^^^^^
``sa_variable``
^^^^^^^^^^^^^^^
Supports all comorbidities

^^^^^^^^^^^^^^^^^
``idu_threshold``
^^^^^^^^^^^^^^^^^
String corresponding to the different multipliers available for setting the mortality threshold 
for the idu population above other risk groups. ``2x``, ``5x``, ``10x``.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_scenario``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BMI scenario to run from ``0``, ``1``, ``2``, or ``3``

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_start_year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Year to begin BMI intervention in simulation

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_end_year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Year to end BMI intervention in simulation

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_coverage``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probability of an eligible agent receiving an intervention

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_effectiveness``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Efficacy of intervention received by agents receiving intervention.

TODO: Add the range for each of the variables.

--------------------
Population Variables
--------------------
The following variables are included in the model to track the population of agents:

- ``age`` (int): age of the agent in years
- ``age_cat`` (int): age category of the agent
- ``anx`` (bool): Whether the agent has anxiety
- ``ckd`` (bool): Whether the agent has chronic kidney disease
- ``delta_bmi`` (float): Change in BMI for the agent at ART initiation
- ``dm`` (bool): Whether the agent has diabetes
- ``dpr`` (bool): Whether the agent has depression
- ``esld`` (bool): Whether the agent has end-stage liver disease
- ``h1yy`` (int): The year of ART initiation for the agent
- ``hcv`` (bool): Whether the agent has hepatitis C
- ``ht`` (bool): Whether the agent has hypertension
- ``init_age`` (int): Age at ART initiation for the agent
- ``init_sqrtcd4n`` (float): Square root of CD4 count at ART initiation for the agent
- ``intercept`` (int): Intercept variable that stores 1
- ``last_h1yy`` (int): ?
- ``last_init_sqrtcd4n`` (float): ?
- ``lipid`` (bool): Whether the agent has dyslipidemia
- ``ltfu_year`` (int): Year the agent was lost to follow-up, if applicable
- ``malig`` (bool): Whether the agent has malignancy
- ``mi`` (bool): Whether the agent has myocardial infarction
- ``mm`` (int): Multimorbidity count and code for the agent
- ``n_lost`` (int): Number of times the agent has been lost to follow-up
- ``post_art_bmi`` (float): BMI of the agent after ART initiation
- ``pre_art_bmi`` (float): BMI of the agent before ART initiation
- ``return_year`` (int): Year the agent returned to care, if applicable
- ``smoking`` (bool): Whether the agent is a smoker
- ``sqrtcd4n_exit`` (float): Square root of CD4 count at exit for the agent
- ``status`` (int): Whether the agent is alive, dead, or lost to follow-up
- ``t_anx`` (int): Year the agent developed anxiety, if applicable
- ``t_ckd`` (int): Year the agent developed chronic kidney disease, if applicable
- ``t_dm`` (int): Year the agent developed diabetes, if applicable
- ``t_dpr`` (int): Year the agent developed depression, if applicable
- ``t_esld`` (int): Year the agent developed end-stage liver disease, if applicable
- ``t_hcv`` (int): Year the agent developed hepatitis C, if applicable
- ``t_ht`` (int): Year the agent developed hypertension, if applicable
- ``t_lipid`` (int): Year the agent developed dyslipidemia, if applicable
- ``t_malig`` (int): Year the agent developed malignancy, if applicable
- ``t_mi`` (int): Year the agent developed myocardial infarction, if applicable
- ``t_smoking`` (int): Year the agent started smoking, if applicable
- ``time_varying_sqrtcd4n`` (float): Square root of CD4 count for the agent at each time step
- ``year`` (int): Current year in the simulation for the agent
- ``year_died`` (int): Year the agent died, if applicable
- ``years_out`` (int): Number of years the agent has been out of care, if applicable

---------------------------------
Note on "t_comorbidity" Variables
---------------------------------
An agent entering a model with a comorbidity will have the following states:

- ART user with comorbidity: Agent has comorbidity=True and t_comorbidity=-1
- Non-ART user with comorbidity: Agent has comorbidity=True and t_comorbidity=-1
- Any agent without a comorbidity will have comorbidity=False and t_comorbidity=0. If the agent later develops the comorbidity, then comorbidity will change to True and t_comorbidity will be set to the year of development.

"""Global Definitions for Codes used across PEARL."""

from pathlib import Path
from typing import Dict, List

# Django style root dir definition
ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()

POPULATION_TYPE_DICT = {
    "age": "int8",
    "age_cat": "int8",
    "anx": "bool",
    "ckd": "bool",
    "dm": "bool",
    "dpr": "bool",
    "esld": "bool",
    "h1yy": "int16",
    "hcv": "bool",
    "ht": "bool",
    "init_age": "int8",
    "last_h1yy": "int16",
    "lipid": "bool",
    "ltfu_year": "int16",
    "malig": "bool",
    "mi": "bool",
    "mm": "int8",
    "n_lost": "int32",
    "return_year": "int16",
    "smoking": "bool",
    "sqrtcd4n_exit": "float64",
    "status": "int8",
    "t_anx": "int16",
    "t_ckd": "int16",
    "t_dm": "int16",
    "t_dpr": "int16",
    "t_esld": "int16",
    "t_hcv": "int16",
    "t_ht": "int16",
    "t_lipid": "int16",
    "t_malig": "int16",
    "t_mi": "int16",
    "t_smoking": "int16",
    "year": "int16",
    "years_out": "int16",
    "intercept": "int8",
}
"""Dict[str, str]: Dictionary defining the data types for each column in the population DataFrame."""

# Status Constants
ART_NAIVE = 0
"""int: ART Naive status code."""
DELAYED = 1
"""int: ART Delayed status code."""
ART_USER = 2
"""int: ART User status code."""
ART_NONUSER = 3
"""int: ART Non-user status code."""
REENGAGED = 4
"""int: ART Reengaged status code."""
LTFU = 5
"""int: LTFU status code."""
DYING_ART_USER = 6
"""int: ART Dying User status code."""
DYING_ART_NONUSER = 7
"""int: ART Dying Non-user status code."""
DEAD_ART_USER = 8
"""int: ART Dead User status code."""
DEAD_ART_NONUSER = 9
"""int: ART Dead Non-user status code."""

# Smearing correction
SMEARING = 1.4
"""float: Smearing correction factor for log-normal distributions."""

# Comorbidity stages
STAGE0 = ["hcv", "smoking"]
"""List[str]: List of comorbidities in stage 0."""
STAGE1 = ["anx", "dpr"]
"""List[str]: List of comorbidities in stage 1."""
STAGE2 = ["ckd", "lipid", "dm", "ht"]
"""List[str]: List of comorbidities in stage 2."""
STAGE3 = ["malig", "esld", "mi"]
"""List[str]: List of comorbidities in stage 3."""
ALL_COMORBIDITIES = STAGE0 + STAGE1 + STAGE2 + STAGE3
"""List[str]: List of all comorbidities."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def calculate_prob(pop: pd.DataFrame, coeffs: NDArray[Any]) -> NDArray[Any]:
    """
    Calculate and return a numpy array of individual probabilities from logistic regression
    given the population and coefficient matrices.
    Used for multiple logistic regression functions.

    Parameters
    ----------
    pop : pd.DataFrame
        Population Dataframe that results from calling create_mortality_in_care_pop_matrix,
        create_mortality_out_care_pop_matrix, create_ltfu_pop_matrix, or
        create_comorbidity_pop_matrix.

    coeffs : NDArray[Any]
        Coefficients are stored in Parameters object with attribute names corresponding to
        the above population preparation functions.

    Returns
    -------
    NDArray[Any]
        Result of multiplying the population by the coefficients and converting to probability.
    """
    # Calculate log odds using a matrix multiplication
    log_odds = np.matmul(pop, coeffs)

    # Convert to probability
    prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
    return np.array(prob)

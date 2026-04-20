"""Base model for PEARL implementing the default population and events."""

from pearl.engine import Pearl
from pearl.events import PearlEvents
from pearl.parameters import Parameters
from pearl.population import PearlPopulation


class BasePearl(Pearl):
    """Base class for Pearl model. Upon initialization, creates a population and defines the events
    to be applied to the population during the run. The population generator and events are defined
    in the PearlPopulation and PearlEvents classes, respectively, and are initialized with the
    parameters passed to the BasePearl class.

    Parameters
    ----------
    Pearl : Extends Pearl Class
    """

    def __init__(self, parameters: Parameters):
        """Initiaize with the default PEARL population and events based on the given parameters.

        Parameters
        ----------
        parameters : Parameters
            The parameters for the model as defined in pearl.parmeters.Parameters
        """
        super().__init__(
            parameters,
            population_generator=PearlPopulation(parameters),
            events=PearlEvents(parameters),
        )

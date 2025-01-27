from pearl.engine import Pearl
from pearl.events import PearlEvents
from pearl.parameters import Parameters
from pearl.population import PearlPopulation


class BasePearl(Pearl):
    """Base class for Pearl model.

    Parameters
    ----------
    Pearl : Extends Pearl Class
    """
    def __init__(self, parameters: Parameters):
        """Initiaize with the default PEARL model components.

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

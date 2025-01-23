from pearl.engine import Pearl
from pearl.events import PearlEvents
from pearl.parameters import Parameters
from pearl.population import PearlPopulation


class BasePearl(Pearl):
    def __init__(self, parameters: Parameters):
        super().__init__(
            parameters,
            population_generator=PearlPopulation(parameters),
            events=PearlEvents(parameters),
        )

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from pearl.definitions import EventType
from pearl.parameters import Parameters


class Event(ABC):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.random_state = parameters.random_state

    @abstractmethod
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class EventGrouping:
    def __init__(self, events: List[EventType]):
        self.events: List[EventType] = events

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        for event in self.events:
            population = event(population)


class Pearl:
    def __init__(self, population_generator: EventType):
        self.population_generator = population_generator
        self.before_run: Optional[EventType] = None
        self.after_run: Optional[EventType] = None
        self.events: Optional[EventType] = None

        self.population = self.population_generator(pd.DataFrame([]))

    def run(self, start_year, end_year):
        self.population = self.before_run(self.population)
        for _ in range(end_year - start_year):
            self.population = self.events(self.population)
        self.population = self.after_run(self.population)

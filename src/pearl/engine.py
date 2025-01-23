from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, Union

import pandas as pd

from pearl.parameters import Parameters


class Event(ABC):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.random_state = parameters.random_state

    @abstractmethod
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class EventGrouping:
    def __init__(self, events: list[EventType]):
        self.events: list[EventType] = events

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        for event in self.events:
            population = event(population)
        return population


# custom type aliases
EventFunction: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]
EventType: TypeAlias = Union[Event, EventGrouping, EventFunction]


class Pearl:
    def __init__(
        self,
        parameters: Parameters,
        population_generator: EventType,
        events: EventType,
        before_run_events: EventType | None = None,
        after_run_events: EventType | None = None,
    ):
        self.parameters = parameters
        self.population_generator = population_generator
        self.before_run_events: EventType | None = before_run_events
        self.after_run_events: EventType | None = after_run_events
        self.events: EventType = events

        self.population = self.population_generator(pd.DataFrame([]))

    def run(self) -> None:
        if self.before_run_events is not None:
            self.population = self.before_run_events(self.population)
        for _ in range(self.parameters.final_year - self.parameters.start_year):
            self.population = self.events(self.population)
        if self.after_run_events is not None:
            self.population = self.after_run_events(self.population)

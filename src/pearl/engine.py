from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, Union

import pandas as pd

from pearl.parameters import Parameters


class Event(ABC):
    """Abstract class for the core function of the PEARL model.
    """
    def __init__(self, parameters: Parameters):
        """Store parameters and initialize a random state.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters
        """
        self.parameters = parameters
        self.random_state = parameters.random_state

    @abstractmethod
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Method to be implemented by subclasses that is invoked by the PEARL engine.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe to be acted upon.

        Returns
        -------
        pd.DataFrame
            Population Dataframe after the event or group of events have been applied.

        Raises
        ------
        NotImplementedError
            Raises an error if the method is not implemented by the subclass.
        """
        raise NotImplementedError


class EventGrouping:
    def __init__(self, events: list[EventType]):
        """A grouping of events to be applied to the population.

        Parameters
        ----------
        events : list[EventType]
            A list of events that are applied sequentially to the passed population during 
            __call__().
        """
        self.events: list[EventType] = events

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Apply a group of events to the population sequentially.

        Parameters
        ----------
        population : pd.DataFrame
            Population Dataframe to be acted upon.

        Returns
        -------
        pd.DataFrame
            Population Dataframe after the event or group of events have been applied.
        """
        for event in self.events:
            population = event(population)
        return population


# custom type aliases
EventFunction: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]
EventType: TypeAlias = Union[Event, EventGrouping, EventFunction]


class Pearl:
    """Base Structure for all PEARL models."""
    def __init__(
        self,
        parameters: Parameters,
        population_generator: EventType,
        events: EventType,
        before_run_events: EventType | None = None,
        after_run_events: EventType | None = None,
    ):
        """Initialize the PEARL model with population and events.

        Parameters
        ----------
        parameters : Parameters
            Parameters object definining a run as defined in pearl.parameters.Parameters.
        population_generator : EventType
            Population generator to be used to create the initial population.
        events : EventType
            Events to be applied to the population during the run.
        before_run_events : EventType | None, optional
            Events to be applied to the population before the run, by default None.
        after_run_events : EventType | None, optional
            Events to be applied to the population after the run, by default None.
        """
        self.parameters = parameters
        self.population_generator = population_generator
        self.before_run_events: EventType | None = before_run_events
        self.after_run_events: EventType | None = after_run_events
        self.events: EventType = events

        self.population = self.population_generator(pd.DataFrame([]))

    def run(self) -> None:
        """Run the pearl model.
        """
        if self.before_run_events is not None:
            self.population = self.before_run_events(self.population)
        for _ in range(self.parameters.final_year - self.parameters.start_year):
            self.population = self.events(self.population)
        if self.after_run_events is not None:
            self.population = self.after_run_events(self.population)

import pandas as pd

from abc import ABC
from typing import Union
from typing import Optional
from typing import List

class Event(ABC):
    def __init__(self):
        pass
    
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        pass
    
class EventGrouping():
    def __init__(self):
        self.events:List[Event] = []
    
    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        
        for event in self.events:
            population = event(population)

class Pearl():
    def __init__(self, population_generator: Union[Event, EventGrouping]):
        
        self.population_generator = population_generator
        self.before_run: Optional[Union[Event, EventGrouping]] = None
        self.after_run: Optional[Union[Event, EventGrouping]] = None
        self.events: Optional[Union[Event, EventGrouping]] = None
        
        self.population = self.population_generator(pd.DataFrame([]))

    def run(self, start_year, end_year):
        self.population = self.before_run(self.population)      
        for _ in range(end_year - start_year):
            self.population = self.events(self.population)
        self.population = self.after_run(self.population)

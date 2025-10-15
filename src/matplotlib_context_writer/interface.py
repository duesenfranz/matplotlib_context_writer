
from abc import ABC, abstractmethod
from matplotlib.figure import Figure

class EnteredVisualizer(ABC):
    @abstractmethod
    def step(self):
        ...

class EnterVisualizer(ABC):
    @abstractmethod
    def __enter__(self) -> EnteredVisualizer:
        ...
    
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        ...

class Visualizer(ABC):
    @abstractmethod
    def enter(self, fig: Figure) -> EnterVisualizer:
        ...
    
class NoopEnteredVisualizer(EnteredVisualizer):
    def step(self):
        pass

class NoopEnterVisualizer(EnterVisualizer):
    def __enter__(self) -> NoopEnteredVisualizer:
        return NoopEnteredVisualizer()
    
    def __exit__(self, exc_type, exc_value, traceback):
        ...

class NoopVisualizer(Visualizer):
    def enter(self, fig: Figure) -> EnterVisualizer:
        return NoopEnterVisualizer()
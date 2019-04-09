from abc import ABC, abstractmethod


class IUpdatable(ABC):
    @abstractmethod
    def update(self, dt):
        pass
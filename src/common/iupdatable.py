from abc import ABCMeta, abstractmethod


class IUpdatable(metaclass=ABCMeta):
    @abstractmethod
    def update(self, dt_sec):
        pass

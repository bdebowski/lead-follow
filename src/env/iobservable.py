from abc import ABCMeta, abstractmethod


class IObservable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def position(self):
        pass

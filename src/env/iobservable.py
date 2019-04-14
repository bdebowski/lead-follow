from abc import ABCMeta, abstractmethod


class IObservable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def pos(self):
        pass

    @property
    @abstractmethod
    def rot(self):
        pass

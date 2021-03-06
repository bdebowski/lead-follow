from abc import ABCMeta, abstractmethod


class IDrawable(metaclass=ABCMeta):
    @abstractmethod
    def draw(self):
        pass

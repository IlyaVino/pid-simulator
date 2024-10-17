from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy import typing as npt
import numpy as np

@dataclass
class InitialConditions:
    r: npt.NDArray
    t: npt.NDArray
    active: npt.NDArray
    initial_u: float

    @property
    def dt(self) -> float:
        return np.mean(np.diff(self.t))


class History(ABC):
    """Currently just used to keep track of type."""
    # TODO: possible refactor is to have tightly coupled "element" and "history" classes
    #  where element is plant, controller, diff loop, setpoint, etc and history represents the
    #  input and output states along with helper methods to manage
    pass


class PlantInterface(ABC):
    """Interface to abstract the plant block in the control loop"""

    @abstractmethod
    def s_domain(self, s: npt.NDArray[np.complex64]) -> npt.NDArray:
        """Returns the plant response function in the Laplace domain evaluated at complex-valued s."""
        pass

    @abstractmethod
    def impulse_response(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns the impulse response function evaluated at real-valued t.

        This is the inverse Laplace transform of the transfer function. """
        # TODO: consider implementing symbolically?

    @abstractmethod
    def process_value(self, history: History) -> npt.NDArray[np.float64]:
        """Returns the plant output given new input and input history."""


class ControllerInterface(ABC):
    """Interface to abstract the controller in the control loop.

    Expect this to be generic for all controller types, including PID, PII2, continuous, difference, bilinear, etc.
    """

    @abstractmethod
    def actuator(self, history: History) -> float:
        """Returns the actuator value depending on the approximation scheme and implementation."""


class LoopSimulator(ABC):
    # TODO: consider refactoring simulation so that different elements connect, with a loop being a special type of
    #   of element
    def __init__(self, plant: PlantInterface, controller: ControllerInterface):
        self.plant = plant
        self.controller = controller

    @abstractmethod
    def simulate(self, initial: InitialConditions) -> History:
        """Runs a PID simulation implemented by the plant and controller"""
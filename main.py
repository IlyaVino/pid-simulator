# PID simulator
#
# For conventions, see https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller
#
# impulse response (i.e. plant response time domain)
# solver (i.e. PID implementation)
# define parameters --> specific to solver class
# define inputs --> actuator, readout, setpoint vs t
import typing
# Calculation overview:
# 1. calculate error -> implement in control class
# 2. calculate PID -> implement in control class
# 3. calculate response -> implement in plant class
# 4. calculate error

# input and output values dataclass
# Initial
#   setpoint(t)
#   controller_active(t)
#   initial_u
#   initial_r
#   initial_e
# History
#   u(t)
#   r(t)
#   e(t)

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass
class InitialConditions:
    r: npt.NDArray
    active: npt.NDArray
    initial_u: float
    initial_r: float
    initial_e: float


@dataclass
class History:
    u: npt.NDArray
    y: npt.NDArray
    e: npt.NDArray
    initial: InitialConditions


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
        # TODO: consider how to handle actuator history


class SecondOrderDelay(PlantInterface):

    def __init__(self, delta: float, w0: float, w1: float, gain: float):
        self.delta = delta
        self.w0 = w0
        self.w1 = w1
        self.gain = gain

    def s_domain(self, s: npt.NDArray) -> npt.NDArray:
        """formula"""
        raise NotImplemented

    def impulse_response(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """inverse laplace transform of s_domain"""
        raise NotImplemented

    def process_value(self, history: History) -> float:
        """convolution of history.u with impulse response."""
        raise NotImplemented


class ControllerInterface(ABC):
    """Interface to abstract the controller in the control loop.

    Expect this to be generic for all controller types, including PID, PII2, continuous, difference, bilinear, etc.
    """

    @abstractmethod
    def actuator(self, history: History) -> float:
        """Returns the actuator value depending on the approximation scheme and implementation."""


class PIDController(ControllerInterface):

    def __init__(self, kp: float, kd: float, ki: float):
        self.kp = kp
        self.kd = kd
        self.ki = ki

    def actuator(self, history: History) -> float:
        """Implement wikipedia difference equation"""
        raise NotImplemented


class SimulateLoop(ABC):
    def __init__(self, plant: PlantInterface, controller: ControllerInterface):
        self.plant = plant
        self.controller = controller

    @abstractmethod
    def simulate(self, initial: InitialConditions) -> History:
        """Runs a PID simulation implemented by the plant and controller"""

if __name__ == '__main__':
    pass

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

from interfaces import InitialConditions, PlantInterface, History, ControllerInterface, LoopSimulator


class GriddedHistory(History):

    def __init__(self, initial: InitialConditions):
        """

        Param:
            initial: InitialConditions object defining setpoint and actuator value
            init_pad: Optional padding for t < 0 (necessary for numerical implementation). Defaults to 2.
        """
        # TODO: implement pre-padding on t
        self._u = initial.initial_u * np.ones(initial.t.shape)
        self._y = np.zeros(initial.t.shape)
        self.initial = initial
        self._current_ind: int = np.argmin(np.abs(initial.t))

    @property
    def t_curr(self) -> float:
        return self.initial.t[self._current_ind]

    @property
    def r_curr(self) -> float:
        return self.initial.r[self._current_ind]

    @property
    def u_curr(self) -> float:
        return self._u[self._current_ind]

    @u_curr.setter
    def u_curr(self, val: float) -> None:
        self._u[self._current_ind] = val

    def u_shift(self, shift: int = 0) -> float:
        """Returns the actuator output shifted from the current index by shift."""
        return self._u[self._current_ind + shift]

    @property
    def y_curr(self) -> float:
        return self._y[self._current_ind]

    @y_curr.setter
    def y_curr(self, val: float) -> None:
        self._y[self._current_ind] = val

    def e_shift(self, shift: int = 0) -> float:
        """Returns the error shifted from the current index by shift."""
        return self.r[self._current_ind + shift] - self._y[self._current_ind + shift]

    @property
    def t(self) -> npt.NDArray:
        return self.initial.t

    @property
    def dt(self) -> float:
        return np.mean(np.diff(self.t))

    @property
    def r(self) -> npt.NDArray:
        return self.initial.r

    @property
    def u(self) -> npt.NDArray:
        return self._u

    @property
    def y(self) -> npt.NDArray:
        return self._y

    @property
    def e(self) -> npt.NDArray:
        return self.r - self._y

    @property
    def running(self) -> bool:
        return self._current_ind < len(self.t)

    def increment(self) -> None:
        self._current_ind += 1

    def backwards_u(self, t: npt.NDArray):
        """Returns the actuator value u, starting from current time index, going backward based on t."""
        # TODO: convert interp into function and create methods for u, y, and e
        return np.interp(
            t,
            self.t_curr-np.flip(self.t[0:self._current_ind]),
            np.flip(self.u[0:self._current_ind]),
            left=self.initial.initial_u
        )


class SecondOrderDelay(PlantInterface):

    def __init__(
            self,
            delta: float,
            w0: float,
            w1: float,
            gain: float,
            dt: float = 100e-9,   # TODO: figure out a better way to manage time grid
            conv_n_tail: float = 8.0,
    ):
        """Instantiate a new plant with a 2nd order transfer function with a propagation delay.

        Param:
            delta: plant propagation delay in s.
            w0: 1st pole in Hz. Cannot be 0 Hz.
            w1: 2nd pole in Hz. Cannot be 0 Hz.
            gain: overall plant gain converting actuator to output.
            conv_n_points: number of points to evaluate the impulse response at for time-domain solutions.
            conv_n_tail: scale factor for determining max delay of the impulse response (conv_n_tail/min(w0, w1)).
        """
        self.delta = delta
        self.w0 = w0
        self.w1 = w1
        self.gain = gain
        self.dt = dt
        self.conv_n_tail = conv_n_tail

    @property
    def impulse_t(self):
        """Returns a time grid for evaluating the impulse response."""
        # TODO: consider abstracting into own class
        max_t = self.conv_n_tail / np.min([self.w0, self.w1])
        return np.arange(0.0, max_t, self.dt)

    def s_domain(self, s: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
        """Laplace-domain linear transfer function evaluated on the laplace domain s = sigma + j * omega."""
        return self.gain * np.exp(s * self.delta) / (1 + s / self.w0) / (1 + s / self.w1)

    def impulse_response(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """inverse laplace transform of s_domain evaluated on a time grid t."""
        shifted_t = t - self.delta
        if self.w0 == self.w1:
            return self.gain * (shifted_t > 0)*(self.w0 ** 2)*np.exp(-self.w0 * shifted_t)
        return self.gain * (shifted_t > 0)*(self.w0 * self.w1) / (self.w1 - self.w0) * (np.exp(-self.w0 * shifted_t)
                                                                            - np.exp(-self.w1 * shifted_t))

    def process_value(self, history: GriddedHistory) -> float:
        """convolution of history.u with impulse response, numerically evaluated on a time grid."""
        return np.sum(history.backwards_u(self.impulse_t) * self.impulse_response(self.impulse_t)) * self.dt


class NoController(ControllerInterface):

    def __init__(self, gain: float):
        self.gain = gain

    def actuator(self, history: GriddedHistory) -> float:
        """Returns the actuator output given the setpoint (i.e. does not use error signal)."""
        return self.gain * history.r_curr


class DiscretePIDController(ControllerInterface):

    def __init__(self, kp: float, kd: float, ki: float):
        self.kp = kp
        self.kd = kd
        self.ki = ki

    def actuator(self, history: GriddedHistory) -> float:
        """Implement wikipedia difference equation"""
        dt = history.dt
        k1 = self.kp + self.ki * dt + self.kd / dt
        k2 = -self.kp - 2 * self.kd / dt
        k3 = self.kd / dt
        return history.u_shift(-1) + k1 * history.e_shift(0) + k2 * history.e_shift(-1) + k3 * history.e_shift(-2)


class TimeDomainLoopSimulator(LoopSimulator):

    def simulate(self, initial: InitialConditions) -> GriddedHistory:
        history = GriddedHistory(initial=initial)
        while history.running:
            history.y_curr = self.plant.process_value(history)
            history.u_curr = self.controller.actuator(history)
            history.increment()

        return history

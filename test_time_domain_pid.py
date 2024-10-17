import pytest
from time_domain_pid import SecondOrderDelay, InitialConditions, GriddedHistory, DiscretePIDController, NoController, \
    TimeDomainLoopSimulator
import numpy as np

@pytest.fixture
def plant():
    return SecondOrderDelay(
        delta=500e-9,
        w0=100e3,
        w1=500e3,
        gain=1.0,
    )

@pytest.fixture
def history():
    t = np.linspace(-10e-6, 50e-6, 1000)
    init_conds = InitialConditions(
        t=t,
        r=0.5 * (t > 0),
        active=np.ones(t.shape, dtype=np.bool),
        initial_u=0.0,
    )
    return GriddedHistory(
        initial=init_conds,
    )

@pytest.fixture
def pid_controller():
    return DiscretePIDController(
        kp=0.0,
        ki=1e6,
        kd=0.0,
    )

@pytest.fixture
def no_controller():
    return NoController(
        gain=1.0
    )

def test_plant_impulse_response(
    plant
):
    response = plant.impulse_response(plant.impulse_t)
    assert response.shape == plant.impulse_t.shape


def test_plant_process_value(
        plant,
        history,
):
    pv = plant.process_value(history)
    assert np.isclose(pv, 0.0)


@pytest.mark.parametrize(
    "controller", ["pid_controller", "no_controller"]
)
def test_actuator(
    controller,
    history,
    request,
):
    controller = request.getfixturevalue(controller)
    actuator_value = controller.actuator(history)
    assert np.isclose(actuator_value, 0.0)


def test_simulator(
    pid_controller,
    plant,
    history
):
    """Only check that it runs for now."""
    _ = TimeDomainLoopSimulator(
        plant,
        pid_controller,
    )
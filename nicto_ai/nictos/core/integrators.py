import numpy as np
from nicto_ai.nictos.core.types import GenericType, Effect

BOLTZMANN = 1.380649e-23  # J/K


def verlet_step(entities: list[GenericType], effects: list[Effect], dt: float) -> list[GenericType]:
    """Velocity-Verlet step using stored acceleration from the previous step.

    Stores ``_accel`` on each entity's attributes so consecutive calls give
    the correct velocity Verlet evolution.

    *First call*: ``_accel`` is absent → initialized from current forces
    (equivalent to starting with a half-step Euler kick, which is fine).

    *Subsequent calls*: stored ``_accel`` = a(t) from the last step; current
    forces = a(t+dt); the step completes the correct velocity-Verlet composite.
    """
    dt2 = dt * dt
    force_map: dict[int, np.ndarray] = {}
    for e in effects:
        if e.force is not None:
            force_map[e.entity_id] = np.asarray(e.force, dtype=np.float64)

    for ent in entities:
        if ent.position is None or "velocity" not in ent.attributes or "mass" not in ent.attributes:
            continue
        f = force_map.get(ent.id)
        if f is None:
            continue
        mass = ent.attributes["mass"]
        a_tp1 = f / mass                                  # a(t+dt)
        a_t = ent.attributes.get("_accel")
        if a_t is not None:
            a_t = np.asarray(a_t, dtype=np.float64)
        else:
            a_t = a_tp1.copy()                             # first step

        vel = np.asarray(ent.attributes["velocity"], dtype=np.float64)
        pos = np.asarray(ent.position, dtype=np.float64)

        # velocity Verlet
        pos = pos + vel * dt + 0.5 * a_t * dt2
        vel = vel + 0.5 * (a_t + a_tp1) * dt

        ent.position = pos
        ent.attributes["velocity"] = vel.tolist()
        ent.attributes["_accel"] = a_tp1.tolist()

    return entities


def symplectic_euler_step(entities: list[GenericType], effects: list[Effect], dt: float) -> list[GenericType]:
    """Simpler symplectic integrator (kick-drift). Less accurate than
    velocity Verlet but still energy-stable."""
    force_map: dict[int, np.ndarray] = {}
    for e in effects:
        if e.force is not None:
            force_map[e.entity_id] = np.asarray(e.force, dtype=np.float64)

    for ent in entities:
        if ent.position is None or "velocity" not in ent.attributes or "mass" not in ent.attributes:
            continue
        f = force_map.get(ent.id)
        if f is None:
            continue
        mass = ent.attributes["mass"]
        vel = np.asarray(ent.attributes["velocity"], dtype=np.float64)
        acc = f / mass
        vel = vel + acc * dt                     # kick
        ent.position = ent.position + vel * dt   # drift
        ent.attributes["velocity"] = vel.tolist()
    return entities

import numpy as np
from nicto_ai.nictos.core.engine import SimulationState


def plot_trajectory(states: list[SimulationState], ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(states[0].entities)):
        xs = [s.entities[i].position[0] if s.entities[i].position is not None else 0 for s in states]
        ys = [s.entities[i].position[1] if s.entities[i].position is not None else 0 for s in states]
        zs = [s.entities[i].position[2] if s.entities[i].position is not None else 0 for s in states]
        ax.plot(xs, ys, zs, label=f"Entity {states[0].entities[i].id}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    return ax


def plot_energy(states: list[SimulationState], ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    times = [s.time for s in states]
    energies = [s.energy for s in states if s.energy is not None]
    if energies:
        ax.plot(times[:len(energies)], energies)
        ax.set_xlabel("Time"); ax.set_ylabel("Energy"); ax.set_title("Energy vs Time")
    return ax


def visualize_3d(states: list[SimulationState], step: int = -1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    s = states[step]
    for e in s.entities:
        if e.position is not None:
            ax.scatter(*e.position[:3], s=50)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(f"Simulation at t={s.time:.3f}")
    plt.show()

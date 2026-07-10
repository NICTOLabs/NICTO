"""Knowledge module — physical constants and domain templates."""

PHYSICAL_CONSTANTS = {
    "G": 6.67430e-11,           # gravitational constant (m^3 kg^-1 s^-2)
    "k_e": 8.9875517923e9,      # Coulomb constant (N m^2 C^-2)
    "c": 299792458,             # speed of light (m/s)
    "h": 6.62607015e-34,        # Planck constant (J·s)
    "hbar": 1.054571817e-34,    # reduced Planck constant (J·s)
    "k_B": 1.380649e-23,        # Boltzmann constant (J/K)
    "N_A": 6.02214076e23,       # Avogadro number
    "e": 1.602176634e-19,       # elementary charge (C)
    "m_e": 9.10938356e-31,      # electron mass (kg)
    "m_p": 1.67262192369e-27,   # proton mass (kg)
    "m_n": 1.67492749804e-27,   # neutron mass (kg)
    "a_0": 5.29177210903e-11,   # Bohr radius (m)
    "eV": 1.602176634e-19,      # electronvolt (J)
    "AMU": 1.66053906660e-27,   # atomic mass unit (kg)
}

ELEMENT_DATA = {
    "H": {"Z": 1,  "mass_amu": 1.008,  "covalent_radius": 0.31e-10},
    "He": {"Z": 2, "mass_amu": 4.003,  "covalent_radius": 0.28e-10},
    "C": {"Z": 6,  "mass_amu": 12.011, "covalent_radius": 0.76e-10},
    "N": {"Z": 7,  "mass_amu": 14.007, "covalent_radius": 0.71e-10},
    "O": {"Z": 8,  "mass_amu": 15.999, "covalent_radius": 0.66e-10},
    "Fe": {"Z": 26, "mass_amu": 55.845, "covalent_radius": 1.32e-10},
    "Au": {"Z": 79, "mass_amu": 196.967, "covalent_radius": 1.36e-10},
}

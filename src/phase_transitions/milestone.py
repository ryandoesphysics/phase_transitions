import numpy as np
import matplotlib.pyplot as plt
# transition from paramagnetic material to ferromagnetic material
# or transition from solid to liquid

# generate n_x * n_y 2d array


def generate_lattice(n_x, n_y):
    lattice = np.random.rand(n_y, n_x)
    for j in range(n_y):
        for i in range(n_x):
            lattice[j][i] = -1 if lattice[j][i] < 0.5 else 1
    return lattice


def neighbour_spin_sum(lattice, j, i):
    n_y = len(lattice)
    n_x = len(lattice[0])
    n = (
        lattice[(j + 1) % n_y, i]
        + lattice[(j - 1) % n_y, i]
        + lattice[j, (i + 1) % n_x]
        + lattice[j, (i - 1) % n_x]
    )
    return n


def total_energy(lattice, B, J):
    n_x = len(lattice[0])
    n_y = len(lattice)

    spin_sum = 0
    neighbour_sum = 0

    for j in range(n_y):
        for i in range(n_x):
            current = lattice[j][i]
            spin_sum += current
            neighbour_sum += current * neighbour_spin_sum(lattice, j, i)

    return -B * spin_sum - J * neighbour_sum


def energy_change(J, f, B, s_alpha):
    # print(-2 * (J * f + B) * s_alpha)
    return -2 * (J * f + B) * s_alpha


def flip(spin):
    spin = -1 if spin == 1 else 1
    return spin


def metropolis(lattice, J, B):
    n_x = len(lattice[0])
    n_y = len(lattice)

    # for each lattice site,
    # flip the spin, calculate the energy change
    # keep the flip if exp(-delta E/kT) > 1
    # if not, draw from U[0,1],
    # if exp(-delta E/kT) > random number, keep
    # else revert

    # f = {-4, -2, 0, 2, 4}

    for j in range(n_y):
        for i in range(n_x):
            lattice[j][i] = flip(lattice[j][i])
            f = neighbour_spin_sum(lattice, j, i)
            delta_E = energy_change(J, f, B, lattice[j][i])
            if np.exp(-delta_E) > 1:
                continue
            else:
                r = np.random.uniform(0, 1)
                if np.exp(-delta_E) > r:
                    continue
                else:
                    lattice[j][i] = flip(lattice[j][i])
    return lattice


def metropolis_n(lattice, n, J, B):
    l = lattice
    for i in range(n):
        l = metropolis(l, J, B)
    return l


def m_per_spin(lattice):
    n_x = len(lattice[0])
    n_y = len(lattice)
    s_alpha = 0
    for j in range(n_y):
        s_alpha += sum(lattice[j])
    return s_alpha / (n_x * n_y)


def main():
    J = 0.5
    B = -0.05
    N_x = 32
    N_y = 32
    n_sweeps = 30
    M_per_spin = []
    for n in range(n_sweeps):
        lattice = generate_lattice(N_y, N_x)
        lattice = metropolis_n(lattice, n + 1, J, B)
        M_per_spin.append(m_per_spin(lattice))

    plt.plot(np.arange(1, n_sweeps + 1), M_per_spin)
    plt.xlabel("Number of Sweeps")
    plt.ylabel("Magnetisation per Spin")
    plt.show()


def main2():
    J = 0.5
    B = -0.05
    N_x = 32
    N_y = 32
    n_sweeps = 30
    M_per_spin = []
    lattice = generate_lattice(N_y, N_x)
    for n in range(n_sweeps):
        lattice = metropolis(lattice, J, B)
        M_per_spin.append(m_per_spin(lattice))
    plt.plot(np.arange(1, n_sweeps + 1), M_per_spin)
    plt.xlabel("Number of Sweeps")
    plt.ylabel("Magnetisation per Spin")
    plt.show()


main2()

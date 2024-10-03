import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def build_Ybus(bus_data, branch_data, convert_data=True):
    """
    Construct the Y-bus matrix from bus and branch data. Matpower data format is assumed.
    more efficient implementation could refer to the matpower package
    """
    bus_idx_col, bus_type, pd_col, qd_col, G_col, B_col = (0, 1, 2, 3, 4, 5)
    (
        from_bus_idx_col,
        to_bus_idx_col,
        r_g_col,
        x_b_col,
        chrg_col,
        rate_a_col,
        rate_b_col,
        rate_c_col,
        tap_ratio_col,
        tap_phase_shift_col,
    ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    bus_data[:, bus_idx_col] = bus_data[:, bus_idx_col].astype(int) - 1
    branch_data[:, from_bus_idx_col] = branch_data[:, from_bus_idx_col].astype(int) - 1
    branch_data[:, to_bus_idx_col] = branch_data[:, to_bus_idx_col].astype(int) - 1

    nbus = len(bus_data)
    nline = len(branch_data)

    Y = np.zeros((nbus, nbus), dtype=complex)

    for i in range(nline):
        from_bus = int(branch_data[i, from_bus_idx_col])
        to_bus = int(branch_data[i, to_bus_idx_col])
        _v2, _v3 = branch_data[
            (branch_data[:, from_bus_idx_col] == from_bus)
            & (branch_data[:, to_bus_idx_col] == to_bus),
            [r_g_col, x_b_col],
        ]
        if convert_data:
            r, x = _v2, _v3  # line resistance, line reactance
            z = r + 1j * x  # line impedance
            y = 1 / z  # line admittance
        else:
            y = _v2 + 1j * _v3  # line admittance

        chrg = 1j * branch_data[i, chrg_col] * 0.5  # half of the line charging
        tap = branch_data[i, tap_ratio_col] or 1  # tap ratio
        phase_shift = branch_data[i, tap_phase_shift_col]  # phase shift
        tap = (
            1 / tap * np.exp(-1j * phase_shift * np.pi / 180)
        )  # tap ratio with phase shift

        Y[from_bus, to_bus] -= y * tap  # off-diagonal elements
        Y[to_bus, from_bus] -= y * tap  # off-diagonal elements
        Y[from_bus, from_bus] += (y + chrg) * tap**2  # diagonal elements
        Y[to_bus, to_bus] += y + chrg  # diagonal elements, for to_bus, no tap ratio

    for i in range(nbus):
        bus_idx = int(bus_data[i, bus_idx_col])
        gb, bb = bus_data[
            bus_data[:, bus_idx_col] == bus_idx, [G_col, B_col]
        ]  # bus G shunt, B shunt
        yb = gb + 1j * bb  # bus shunt admittance
        Y[bus_idx, bus_idx] += yb  # add shunt admittance to diagonal elements
    return Y


# Function to compute power flow equations
def power_flow(V, delta):
    n = len(V)
    P_inj = np.zeros(n)
    Q_inj = np.zeros(n)
    for i in range(n):
        for j in range(n):
            P_inj[i] += V[i] * V[j] * (Ybus[i, j].real * np.cos(delta[i] - delta[j]) + Ybus[i, j].imag * np.sin(delta[i] - delta[j]))
            Q_inj[i] -= V[i] * V[j] * (Ybus[i, j].real * np.sin(delta[i] - delta[j]) - Ybus[i, j].imag * np.cos(delta[i] - delta[j]))
    return P_inj, Q_inj

# Function to compute generator equations
def generator_dynamics(x, t):
    delta = x[:n_gen]
    omega = x[n_gen:]

    # Compute electrical power output
    V = np.abs(bus_data[:, 7])
    theta = np.zeros(n_bus)
    P_inj, Q_inj = power_flow(V, theta)

    # bus power flow balance for generator buses
    P_l = bus_data[:, 2] / S_base
    Q_l = bus_data[:, 3] / S_base

    P_g = P_inj + P_l
    Q_g = Q_inj + Q_l
    
    P_g = P_g[:n_gen]
    Q_g = Q_g[:n_gen]

    E_i = P_g * X_d_prime / V[:n_gen] / np.sin(delta - theta[:n_gen])

    # Swing equation
    omega_s = 2 * np.pi * 60  # Synchronous speed in rad/s
    # P_g = E_i * V[:n_gen] * np.sin(delta - theta[:n_gen]) / X_d_prime
    # Q_g = (-V[:n_gen]**2 + E_i*V[:n_gen]*np.cos(theta[:n_gen] - delta)) / X_d_prime

    ddelta = omega - omega_s
    domega = (omega_s / (2 * H)) * (T_M - P_g)  # pu
    
    return np.concatenate([ddelta, domega])

# Function to simulate the system
def simulate_system(t_span, initial_conditions):
    sol = odeint(generator_dynamics, initial_conditions, t_span)
    return sol


if __name__ == '__main__':
    S_base = 100  # MVA

    # Bus data
    # bus data
    #	bus_i	type	Pd	Qd	  Gs	Bs	area	Vm	    Va	baseKV	zone	Vmax	Vmin
    bus_data = np.array([
        [1, 3, 0, 0, 0, 0, 1, 1.04, 0, 345, 1, 1.1, 0.9],
        [2, 2, 0, 0, 0, 0, 1, 1.025, 0, 345, 1, 1.1, 0.9],
        [3, 2, 0, 0, 0, 0, 1, 1.025, 0, 345, 1, 1.1, 0.9],
        [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [5, 1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [6, 1, 90, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [7, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [8, 1, 100, 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [9, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9]
    ])

    # Generator data
    # bus	Pg	Qg	Qmax Qmin	Vg	    mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
    gen_data = np.array([
        [1, 0, 0, 300, -300, 1.04, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 163, 0, 300, -300, 1.025, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 85, 0, 300, -300, 1.025, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Branch data
    # fbus	tbus	r	   x	    b	    rateA rateB	rateC	ratio	    angle	status	angmin	angmax
    branch_data = np.array([
        [1, 4, 0, 0.0576, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 6, 0.017, 0.092, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
        [6, 9, 0.039, 0.17, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 9, 0, 0.0586, 0, 300, 300, 300, 0, 0, 1, -360, 360],
        [8, 9, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085, 0.072, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 7, 0, 0.0625, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [5, 7, 0.032, 0.161, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.01, 0.085, 0.176, 250, 250, 250, 0, 0, 1, -360, 360]
    ])


    # Construct the Y-bus matrix
    Ybus = build_Ybus(bus_data, branch_data)

    # Extract relevant data
    n_bus = bus_data.shape[0]
    n_gen = gen_data.shape[0]
    n_branch = branch_data.shape[0]

    # Generator parameters (example values, adjust as needed)
    H = np.array([6.5, 6.5, 6.5])  # Inertia constants
    X_d_prime = np.array([0.3, 0.25, 0.25])  # Transient reactances
    T_M = gen_data[:, 1] / S_base  # Mechanical torque (assuming initial P_e = T_M)
    D = np.array([1, 0.8, 0.8])  # Damping coefficients

    # Example usage
    t_span = np.linspace(0, 10, 1000)
    initial_conditions = np.zeros(2 * n_gen)
    initial_conditions[:n_gen] = np.zeros(n_gen)  # Initial angles (in rad)
    initial_conditions[n_gen:] = 2 * np.pi * 60  # Initial speeds (in rad/s)

    result = simulate_system(t_span, initial_conditions)

    # Plot generator angles
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for i in range(n_gen):
        plt.plot(t_span, result[:, i], label=f'Generator {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Generator Angle (rad)')
    plt.title('Generator Angle Dynamics')
    plt.legend()
    plt.grid(True)

    # Plot machine speeds
    plt.subplot(2, 1, 2)
    for i in range(n_gen):
        plt.plot(t_span, result[:, n_gen + i] / (2 * np.pi * 60), label=f'Generator {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Machine Speed (pu)')
    plt.title('Machine Speed Dynamics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print initial and final values
    print("Initial conditions:")
    for i in range(n_gen):
        print(f"Generator {i+1}: Angle = {initial_conditions[i]:.4f} rad, Speed = {initial_conditions[n_gen+i]/(2*np.pi*60):.4f} pu")

    print("\nFinal values:")
    for i in range(n_gen):
        print(f"Generator {i+1}: Angle = {result[-1, i]:.4f} rad, Speed = {result[-1, n_gen+i]/(2*np.pi*60):.4f} pu")
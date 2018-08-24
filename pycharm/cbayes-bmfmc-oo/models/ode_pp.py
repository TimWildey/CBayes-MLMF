import math

import matplotlib.pyplot as plt
import numpy as np


# Lotka - Volterra PP Model
# dxdt = alpha * x - beta * x * y
# dydt = delta * x * y - gamma * y


# A very simple data structure for settings
class Settings:
    def __init__(self, finalt=50.0, dt=0.01, u0=np.array([100, 10])):
        self.finalt = finalt
        self.nt = math.ceil(finalt / dt)
        self.u0 = u0
        self.deltat = dt
        self.timespan = np.linspace(0, finalt, self.nt+1)


def get_qoi_samples(rv_samples, settings):

    qvals = np.zeros((np.shape(rv_samples)[0], ))
    for i in range(np.shape(rv_samples)[0]):
        sol = ode_pp(rv_samples[i, :], settings)
        qvals[i] = sol[0, -1]

    return qvals


def ode_pp(params, settings):

    # Random variables
    alpha = params[0]
    beta = params[1]
    delta = params[2]
    gamma = params[3]

    # Settings
    nt = settings.nt
    u = settings.u0
    finalt = settings.finalt
    sol = np.zeros((2, nt + 1))
    sol[:, 0] = u

    # Apply an explicit, forth-order Runge-Kutta scheme to integrate
    time = 0.0  # current time
    tol = 1e-6  # tolerance for time mismatch
    for k in range(nt):

        # Reduce time step size of the last step if necessary
        dt = settings.deltat
        if (k+1)*dt > finalt:
            dt = finalt - k*dt

        # 4th order RK slope estimates
        y1 = u
        k1 = np.array([alpha * y1[0] - beta * y1[0] * y1[1], delta * y1[0] * y1[1] - gamma * y1[1]])
        y2 = u + dt * k1 / 2
        k2 = np.array([alpha * y2[0] - beta * y2[0] * y2[1], delta * y2[0] * y2[1] - gamma * y2[1]])
        y3 = u + dt * k2 / 2
        k3 = np.array([alpha * y3[0] - beta * y3[0] * y3[1], delta * y3[0] * y3[1] - gamma * y3[1]])
        y4 = u + dt * k3
        k4 = np.array([alpha * y4[0] - beta * y4[0] * y4[1], delta * y4[0] * y4[1] - gamma * y4[1]])

        # Average and update
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[:, k + 1] = u
        time = time + dt

        # Check times
        if k == nt-1 and time - finalt > tol:
            print('Final time mismatch: ' + str(time) + 'vs.' + str(finalt))

    return sol


def get_prior_samples(n_samples):
    return np.random.uniform(low=0.4, high=0.6, size=(n_samples, 4))


if __name__ == '__main__':

    # Model settings
    u0 = np.array([5, 1])
    finalt = 1.0
    dt_hf = 0.1
    dt_lf_2 = 0.6
    dt_lf_1 = 0.9
    dt_lf_0 = 1.0

    # Samples
    n_hf = 200
    samples = np.random.uniform(low=0.4, high=0.6, size=(n_hf, 4))

    # High-fidelity model
    hf_settings = Settings(finalt=finalt, dt=dt_hf, u0=u0)
    hf_sol = np.zeros((n_hf, len(hf_settings.u0), hf_settings.nt+1))
    for i in range(n_hf):
        hf_sol[i, :, :] = ode_pp(samples[i, :], hf_settings)

    # Low-fidelity model 2
    n_lf_2 = n_hf
    lf_2_settings = Settings(finalt=finalt, dt=dt_lf_2, u0=u0)
    lf_2_sol = np.zeros((n_lf_2, len(lf_2_settings.u0), lf_2_settings.nt + 1))
    for i in range(n_lf_2):
        lf_2_sol[i, :, :] = ode_pp(samples[i, :], lf_2_settings)

    # Low-fidelity model 1
    n_lf_1 = n_hf
    lf_1_settings = Settings(finalt=finalt, dt=dt_lf_1, u0=u0)
    lf_1_sol = np.zeros((n_lf_1, len(lf_1_settings.u0), lf_1_settings.nt + 1))
    for i in range(n_lf_1):
        lf_1_sol[i, :, :] = ode_pp(samples[i, :], lf_1_settings)

    # Low-fidelity model 0
    n_lf_0 = n_hf
    lf_0_settings = Settings(finalt=finalt, dt=dt_lf_0, u0=u0)
    lf_0_sol = np.zeros((n_lf_0, len(lf_0_settings.u0), lf_0_settings.nt + 1))
    for i in range(n_lf_0):
        lf_0_sol[i, :, :] = ode_pp(samples[i, :], lf_0_settings)

    # Solution plots
    # plt.figure()
    # plt.plot(lf_1_settings.timespan, np.transpose(lf_1_sol[:, 0, :]), 'C0-')
    # plt.plot(lf_1_settings.timespan, np.transpose(lf_1_sol[:, 1, :]), 'C1-')
    # plt.figure()
    # plt.plot(hf_settings.timespan, np.transpose(hf_sol[:, 0, :]), 'C0-')
    # plt.plot(hf_settings.timespan, np.transpose(hf_sol[:, 1, :]), 'C1-')
    # plt.figure()
    # plt.plot(lf_2_settings.timespan, np.transpose(lf_2_sol[:, 0, :]), 'C0-')
    # plt.plot(lf_2_settings.timespan, np.transpose(lf_2_sol[:, 1, :]), 'C1-')
    # plt.show()
    # exit()

    x0_qvals = lf_0_sol[:, 0, -1]
    x1_qvals = lf_1_sol[:, 0, -1]
    x2_qvals = lf_2_sol[:, 0, -1]
    y_qvals = hf_sol[:, 0, -1]
    lin = np.linspace(np.min([x0_qvals, y_qvals]), np.max([x0_qvals, y_qvals]), len(x1_qvals))

    # Correlate x0 and x1
    plt.figure()
    plt.plot(x0_qvals, x1_qvals, 'oC2', label='$x_0$ vs. $x_1$')
    plt.plot(lin, lin, '--k', label='$x=y$')
    plt.legend()

    # Correlate x1 and x2
    plt.figure()
    plt.plot(x1_qvals, x2_qvals, 'oC1', label='$x_1$ vs. $x_2$')
    plt.plot(lin, lin, '--k', label='$x=y$')
    plt.legend()

    # Correlate x2 and y
    plt.figure()
    plt.plot(x2_qvals, y_qvals, 'oC0', label='$x_1$ vs. $y$')
    plt.plot(lin, lin, '--k', label='$x=y$')
    plt.legend()

    # Correlate x0 and y
    plt.figure()
    plt.plot(x0_qvals, y_qvals, 'oC3', label='$x_2$ vs. $y$')
    plt.plot(lin, lin, '--k', label='$x=y$')
    plt.legend()

    plt.show()
    exit()
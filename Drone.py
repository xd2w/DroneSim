# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt


class Drone:
    """
    A class to represent the dynamics of a drone.

    Parameters
    ----------
    I : [I_xx, I_yy, I_zz] - moment of inertia of the drone about its principal axes
    g : float - acceleration due to gravity
    m : float - mass of the drone

    """

    def __init__(self, I, g=9.81, m=1):

        self.init_state_matrix(I, g, m)

        # 𝜙 - rotation about x (roll)
        # 𝜃 - rotation about y (pitch)
        # 𝜓 - rotation about x (yaw)

        # State variables [x, y, z, x', y', z', 𝜙, 𝜃, 𝜓, 𝜙', 𝜃', 𝜓']
        self.x = np.zeros((12))

        # y[0:3] = x[0:3]
        # y[3:6] = x[6:9]

        # output variables (x, y, z, 𝜙, 𝜃, 𝜓)
        self.y = np.zeros((6))

        # input variables (total thrust, tourque_x, tourque_y, tourque_z)
        self.u = np.zeros((4))

    def init_state_matrix(self, I, g, m):
        # initialises the matrix for state space model where:
        # x' = Ax + Bu
        # y = Cx + Du

        # moment of Inertia of drone about its principal axes
        I_xx, I_yy, I_zz = I

        self.A = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        self.B = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1 / m, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, I_xx, 0, 0],
                [0, 0, I_yy, 0],
                [0, 0, 0, I_zz],
            ]
        )

    def new_x(self, dt, u):
        self.x[:] = self.x[:] + dt * (
            np.matmul(self.A, self.x).T + np.matmul(self.B, u.T)
        )

        self.x[6:9] = self.x[6:9] % (2 * np.pi)

    def get_y(self):
        self.y[0:3] = self.x[0:3]
        self.y[3:6] = self.x[6:9]
        return self.y

    def simulate(self, dt, N, u=None):
        if u is None:
            n0 = np.linspace(5, -5, N)
            # u1 = np.linspace(-np.pi / 120, 0, N - 1)
            u[:, 0] = n0

        res_y = np.zeros((N, 6))
        for i in range(N):
            self.new_x(dt, u[i, :])
            res_y[i] = self.get_y()
            # print(self.x)

        ax = plt.figure().add_subplot(projection="3d")
        ax.plot(res_y[:, 0], res_y[:, 1], res_y[:, 2])
        ax.title.set_text("Drone Trajectory")

        plt.show()


if __name__ == "__main__":
    drone = Drone([1, 1, 2], m=0.5)

    # n0 = np.linspace(5, -5, N)

    # time step
    N = 100

    # input force
    u = np.zeros((N, 4))
    t_param = np.linspace(0, 20 * np.pi, N)
    u1 = (np.pi / 120) * np.sin(t_param)
    u3 = (np.pi / 120) * np.cos(t_param)

    u[:, 0] = 0.1
    u[:, 1] = u1
    u[:, 3] = u3

    drone.simulate(0.1, N, u)

from fixed_call import FixedCall
import math
import numpy

class AsianNewVecerFixedCall(FixedCall):
    def __init__(self, maxt, numx, numt, r, sigma, initial_price, strike):
        super().__init__(maxt, numx, numt, r, sigma, initial_price, strike)
        self.dx = self.maxx / self.numx
        self.j0 = int(self.xi_initial / self.dx)
        self.set_boundary_conditions()

    def xi(self, s, t):
        return math.exp(-self.r * (self.maxt - t)) * s / self.avr(s, t)

    def q(self, t):
        return (math.exp(self.r * self.maxt - t) - 1) / (self.r * self.maxt)

    def avr(self, s, t):
        return self.q(t) * math.exp(-self.r * (self.maxt - t)) * s

    def initial_value_at_height(self, row):
        return max(1 - self.strike * row * self.dx / self.s0, 0)

    def initial_value_at_botom(self, col):
        return 1

    def alpha(self, height, time):
        return .5 * self.sigma ** 2 * (height * self.dx) ** 2 * (
            (math.exp(self.r * (time + 0.5) * self.dt) - 1) /
            (self.r * self.maxt) *
            (height * self.dx) -
            1
        )**2 / (2 * self.dx**2) * self.dt

    def A_matrix(self, time):
        return super().A_matrix(time, lambda a, b: a, lambda a, b: 1 - 2*a, lambda a, b: a)

    def B_matrix(self, time):
        return super().B_matrix(time, lambda a, b: a, lambda a, b: - 2*a, lambda a, b: a)

    def solve(self):
        return super().solve(lambda time: numpy.identity(self.numx + 1) - self.B_matrix(time), lambda time: self.A_matrix(time))
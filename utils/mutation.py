import numpy as np
from pymoo.model.mutation import Mutation


class PolynomialMutation(Mutation):
    def __init__(self, problem, eta=20):
        super().__init__()
        self.eta = float(eta)
        self.problem = problem

    def mutate(self, X, p_mutate=1):

        X = X.astype(float)
        Y = np.full(X.shape, np.inf)

        # self.prob = 1.0 / self.problem.var_nums
        self.prob = p_mutate / self.problem.var_nums

        do_mutation = np.random.random(X.shape) < self.prob
        # do_mutation = np.random.random(X.shape) < p_mutate

        Y[:, :] = X

        xl = np.array(self.problem.var_bound_l)
        xu = np.array(self.problem.var_bound_h)
        xl = np.repeat(xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        # in case obj_out of bounds repair (very unlikely)
        Y = self.set_to_bounds_if_outside_by_problem(Y)

        return Y[0]

    def set_to_bounds_if_outside_by_problem(self, variables):
        variables = list(variables[0])
        for i in range(len(variables)):
            variables[i] = variables[i] if variables[i] > self.problem.var_bound_l[i] else self.problem.var_bound_l[i]
            variables[i] = variables[i] if variables[i] < self.problem.var_bound_h[i] else self.problem.var_bound_h[i]
        return np.array([variables])

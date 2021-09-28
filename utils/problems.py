import numpy as np
import autograd.numpy as anp
from pymoo.factory import get_problem


class SCH:

    def __init__(self, var_nums=1, var_bound=None):
        self.problem_name = 'sch'
        self.var_nums = var_nums
        self.var_bound = [-1000, 1000] if not var_bound else var_bound
        self.obj_nums = 2

    def get_objectives(self, variables):
        f1 = self.f1(variables)
        f2 = self.f2(variables)
        return [f1, f2]

    def f1(self, variables):
        return variables[0] ** 2

    def f2(self, variables):
        return (variables[0] - 2) ** 2


class ZDT1:

    def __init__(self, var_nums=30, var_bound=None):
        self.problem_name = 'zdt1'
        self.var_nums = var_nums
        self.var_bound = [0, 1] if not var_bound else var_bound
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = 2

    def get_objectives(self, variables):
        f1 = variables[0]
        g = 1 + 9.0 / (len(variables) - 1) * anp.sum(variables[1:], axis=0)
        f2 = g * (1 - anp.power((f1 / g), 0.5))

        return [f1, f2]


class ZDT2:

    def __init__(self, var_nums=30, var_bound=None):
        self.problem_name = 'zdt2'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = 2

    def get_objectives(self, variables):
        f1 = variables[0]
        c = anp.sum(variables[1:], axis=0)
        g = 1.0 + 9.0 * c / (len(variables) - 1)
        f2 = g * (1 - anp.power((f1 * 1.0 / g), 2))

        return [f1, f2]


class ZDT3:

    def __init__(self, var_nums=30, var_bound=None):
        self.problem_name = 'zdt3'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = 2

    def get_objectives(self, x):
        f1 = x[0]
        c = anp.sum(x[1:], axis=0)
        g = 1.0 + 9.0 * c / (len(x) - 1)
        f2 = g * (1 - anp.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * anp.sin(10 * anp.pi * f1))

        return [f1, f2]


class ZDT4:
    def __init__(self, var_nums=10, obj_nums=2):
        self.problem_name = 'zdt4'
        self.var_nums = var_nums
        self.obj_nums = obj_nums
        self.var_bound_l = [-5 for _ in range(var_nums)]
        self.var_bound_h = [5 for _ in range(var_nums)]
        self.var_bound_l[0] = 0.0
        self.var_bound_h[0] = 1.0

    def get_objectives(self, x):
        f1 = x[0]
        g = 1.0
        g += 10 * (self.var_nums - 1)
        for i in range(1, self.var_nums):
            g += x[i] * x[i] - 10.0 * anp.cos(4.0 * anp.pi * x[i])
        h = 1.0 - anp.sqrt(f1 / g)
        f2 = g * h

        return np.array([f1, f2])


class ZDT5:

    def __init__(self, m=11, n=5):
        self.problem_name = 'zdt5'
        self.m = m
        self.n = n
        self.var_nums = 30 + n * (m - 1)

    def get_objectives(self, x):
        _x = [x[:30]]
        for i in range(self.m - 1):
            _x.append(x[30 + i * self.n: 30 + (i + 1) * self.n])

        u = anp.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[1:].sum(axis=1)

        f1 = 1 + u[0]
        f2 = g * (1 / f1)

        return np.array([list(f1), list(f2)])


class ZDT6:

    def __init__(self, var_nums=10, obj_nums=2):
        self.var_nums = var_nums
        self.problem_name = 'zdt6'
        self.obj_nums = obj_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]

    def get_objectives(self, x):
        f1 = 1 - anp.exp(-4 * x[0]) * anp.power(anp.sin(6 * anp.pi * x[0]), 6)
        g = 1 + 9.0 * anp.power(anp.sum(x[1:]) / (self.var_nums - 1.0), 0.25)
        f2 = g * (1 - anp.power(f1 / g, 2))

        return np.array([f1, f2])


class DTLZ1:
    def __init__(self, var_nums=7, obj_nums=3, k=None):
        super().__init__()
        self.problem_name = 'dtlz1'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = obj_nums
        if var_nums:
            self.k = var_nums - obj_nums + 1
        elif k:
            self.k = k
            self.var_nums = k + obj_nums - 1

    def g(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def f(self, X_, g):
        f = []

        for i in range(0, self.obj_nums):
            _f = 0.5 * (1 + g)
            _f *= anp.prod(X_[:X_.shape[0] - i])
            if i > 0:
                _f *= 1 - X_[X_.shape[0] - i]
            f.append(_f)

        return f

    def get_objectives(self, x):
        x = np.array(x)
        X_, X_M = x[:self.obj_nums - 1], x[self.obj_nums - 1:]
        g = self.g(X_M)
        return self.f(X_, g)


class DTLZ2:
    def __init__(self, var_nums=12, obj_nums=3):
        self.problem_name = 'dtlz2'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = obj_nums
        if var_nums:
            self.k = var_nums - obj_nums + 1

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5))

    def f(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.obj_nums):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:X_.shape[0] - i], alpha) * anp.pi / 2.0))
            if i > 0:
                _f *= anp.sin(anp.power(X_[X_.shape[0] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        return f

    def get_objectives(self, x):
        x = np.array(x)
        X_, X_M = x[:self.obj_nums - 1], x[self.obj_nums - 1:]
        g = self.g2(X_M)
        return self.f(X_, g, alpha=1)


class DTLZ3:
    def __init__(self, var_nums=12, obj_nums=3):
        self.problem_name = 'dtlz3'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = obj_nums
        if var_nums:
            self.k = var_nums - obj_nums + 1

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def f(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.obj_nums):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:X_.shape[0] - i], alpha) * anp.pi / 2.0))
            if i > 0:
                _f *= anp.sin(anp.power(X_[X_.shape[0] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        return f

    def get_objectives(self, x):
        x = np.array(x)
        X_, X_M = x[:self.obj_nums - 1], x[self.obj_nums - 1:]
        g = self.g1(X_M)
        return self.f(X_, g, alpha=1)


class DTLZ4:
    def __init__(self, var_nums=12, obj_nums=3, alpha=100, d=100):
        self.problem_name = 'dtlz4'
        self.alpha = alpha
        self.d = d
        self.var_nums = var_nums
        self.obj_nums = obj_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5))

    def f(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.obj_nums):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:X_.shape[0] - i], alpha) * anp.pi / 2.0))
            if i > 0:
                _f *= anp.sin(anp.power(X_[X_.shape[0] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        return f

    def get_objectives(self, x):
        x = np.array(x)
        X_, X_M = x[:self.obj_nums - 1], x[self.obj_nums - 1:]
        g = self.g2(X_M)
        return self.f(X_, g, alpha=self.alpha)


class DTLZ5:
    def __init__(self, var_nums=12, obj_nums=3):
        self.problem_name = 'dtlz5'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = obj_nums
        if var_nums:
            self.k = var_nums - obj_nums + 1

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5))

    def f(self, X_, g, alpha=1):
        f = []
        X_ = X_[0]
        for i in range(0, self.obj_nums):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:X_.shape[0] - i], alpha) * anp.pi / 2.0))
            if i > 0:
                _f *= anp.sin(anp.power(X_[X_.shape[0] - i], alpha) * anp.pi / 2.0)
            f.append(_f)
        return f

    def get_objectives(self, x):
        x = np.array(x)
        X_, X_M = x[:self.obj_nums - 1], x[self.obj_nums - 1:]
        g = self.g2(X_M)
        theta = 1 / (2 * (1 + g[None])) * (1 + 2 * g[None] * X_)
        theta = anp.column_stack([x[0], theta[1:]])
        return self.f(theta, g)


class DTLZ6:
    def __init__(self, var_nums=12, obj_nums=3):

        self.problem_name = 'dtlz6'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = obj_nums
        if var_nums:
            self.k = var_nums - obj_nums + 1

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5))))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5))

    def f(self, X_, g, alpha=1):
        f = []
        X_ = X_[0]
        for i in range(0, self.obj_nums):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:X_.shape[0] - i], alpha) * anp.pi / 2.0))
            if i > 0:
                _f *= anp.sin(anp.power(X_[X_.shape[0] - i], alpha) * anp.pi / 2.0)
            f.append(_f)
        return f

    def get_objectives(self, x):
        x = np.array(x)
        X_, X_M = x[:self.obj_nums - 1], x[self.obj_nums - 1:]
        g = anp.sum(anp.power(X_M, 0.1))
        theta = 1 / (2 * (1 + g[None])) * (1 + 2 * g[None] * X_)
        theta = anp.column_stack([x[0], theta[1:]])
        return self.f(theta, g)


class DTLZ7:
    def __init__(self, var_nums=22, obj_nums=3):
        self.problem_name = 'dtlz7'
        self.var_nums = var_nums
        self.var_bound_l = [0 for _ in range(var_nums)]
        self.var_bound_h = [1 for _ in range(var_nums)]
        self.obj_nums = obj_nums
        if var_nums:
            self.k = var_nums - obj_nums + 1

    def get_objectives(self, x):
        x = np.array(x)
        f = []
        for i in range(0, self.obj_nums - 1):
            f.append(x[i])
        f = np.array(f)
        g = 1 + 9 / self.k * anp.sum(x[-self.k:])
        h = self.obj_nums - anp.sum(f / (1 + g[None]) * (1 + anp.sin(3 * anp.pi * f)))

        return np.array([f[0], f[1], (1 + g) * h]).astype(float)

import numpy as np
from pymoo.util.dominator import Dominator
import EMOOConfig


class Individual:

    def __init__(self):
        self.num_dominators = 0  # 支配当前个体的数量
        self.dominating_individuals = []  # 当前个体支配的个体集（所在种群编号）
        self.front_rank = 0  # 当前的front级别
        self.crowding_distance = 0  # 拥挤度
        self.F_values = []
        self.fitness = 0
        self.variables = [bound_l + (bound_h - bound_l) * np.random.rand()
                          for bound_l, bound_h in zip(Individual.problem.var_bound_l, Individual.problem.var_bound_h)]
        self.F_values = np.array(Individual.problem.get_objectives(self.variables)).astype(np.float64)
        self.F_values = self.F_values

    def update_objectives(self):
        self.F_values = np.array(self.problem.get_objectives(self.variables)).astype(np.float64)
        return self.F_values

    def dominates(self, individual):
        return Dominator.get_relation(self.F_values, individual.F_values) > 0

    # def dominates(self, individual):
    #     # a是否支配b
    #     a_f = self.F_values
    #     b_f = individual.F_values
    #     i = 0
    #     for av, bv in zip(a_f, b_f):
    #         if av < bv:
    #             i = i + 1
    #         if av > bv:
    #             return False
    #     if i != 0:
    #         return True
    #     return False

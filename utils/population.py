from utils.individual import Individual
import copy


class Population:

    def __init__(self, population_size=200, individuals=None):
        self.population_size = population_size
        if individuals:
            self.individuals = copy.deepcopy(individuals)
        else:
            self.individuals = [Individual() for _ in range(population_size)]

    def new(self):
        pop = Population(0)
        pop.population_size = self.population_size
        return pop

    def update_population_objectives(self):
        for individual in self.individuals:
            individual.update_objectives()

    @staticmethod
    def empty_population():
        return Population(population_size=0)

    def __len__(self):
        return len(self.individuals)

    def __iter__(self):
        for individual in self.individuals:
            yield individual

    def __next__(self):
        for individual in self.individuals:
            yield individual

    def __getitem__(self, item):
        return self.individuals[item]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    # def crowding_dist(self, population):
    #     # 拥挤度计算,只计算P内Fi位置部分的拥挤度
    #     f_max = population[0].F_values[:]
    #     f_min = population[0].F_values[:]
    #     f_num = len(f_max)
    #     for p in population:
    #         p.crowding_distance = 0
    #         for fm in range(f_num):
    #             if p.F_values[fm] > f_max[fm]:
    #                 f_max[fm] = p.F_values[fm]
    #             if p.F_values[fm] < f_min[fm]:
    #                 f_min[fm] = p.F_values[fm]
    #     Fi_len = len(population)
    #     for m in range(f_num):
    #         population = self.sort_func(population, m)
    #         population[0].crowding_distance = 1000000
    #         population[Fi_len - 1].crowding_distance = 1000000
    #         for f in range(1, Fi_len - 1):
    #             a = population[f + 1].F_values[m] - population[f - 1].F_values[m]
    #             b = f_max[m] - f_min[m]
    #             # population[f].crowding_distance = population[f].crowding_distance + a / b
    #             population[f].crowding_distance = abs(population[f].crowding_distance + a / b)
    #
    #
    # def sort_func(self, Fi, m):
    #     # 对P中Fi索引对应的个体按照第m个函数排序
    #     FL = len(Fi)
    #     for i in range(FL - 1):
    #         p = Fi[i]
    #         for j in range(i + 1, FL):
    #             q = Fi[j]
    #             if p != q and p.F_values[m] > q.F_values[m]:
    #                 Fi.individuals[i], Fi.individuals[j] = Fi.individuals[j], Fi.individuals[i]
    #     return Fi

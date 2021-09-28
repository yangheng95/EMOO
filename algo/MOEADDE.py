import math

import hvwfg
import numpy as np
import tqdm
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

import EMOOConfig
from utils.file_utils import normalize
from utils.individual import Individual
from utils.operators import Operator
from utils.population import Population


class DistInfo:
    def __init__(self, idx=0, value=0):
        self.idx = idx
        self.value = value


class Neighbor:
    def __init__(self, idx=0, pop_size=100):
        self.idx = idx
        self.neighbor = [0 for _ in range(pop_size)]


class IdealPoint:
    def __init__(self, obj_nums):
        self.variable = math.inf
        self.obj = np.array([math.inf] * obj_nums)


class MOEADDE:

    def __init__(
            self,
            ref_point=1,
            indicator=None,
            problem=None
    ):
        self.name = 'MOEADDE'
        self.max_population_size = EMOOConfig.population_size
        self.max_evolution_iteration = EMOOConfig.evolution_iteration
        self.parent_population = None
        self.offspring_population = None
        self.mixed_population = None
        self.problem = problem
        self.operator = None

        self.objectives = []
        self.solutions = []
        self.hv = math.inf
        self.indicator = indicator
        self.ref_point = ref_point
        self.plot = None

        self.num_weight = 0
        self.neighbor_table = [Neighbor(i, EMOOConfig.neighbor_size) for i in range(self.max_population_size)]
        self.ideal_point = None
        self.lambda_list = None

    def set_problem(self, problem=None, ref_point=None):
        if not problem or not ref_point:
            raise ValueError('specify a test problem and reference point!')
        self.max_population_size = EMOOConfig.population_size
        self.max_evolution_iteration = EMOOConfig.evolution_iteration
        self.operator = Operator(self, problem=problem)
        self.problem = problem
        Individual.problem = problem
        self.parent_population = Population(self.max_population_size)
        self.offspring_population = Population(self.max_population_size)
        self.ref_point = ref_point
        self.ideal_point = IdealPoint(self.problem.obj_nums)
        self.lambda_list = self.initialize_uniform_point()

    def init_ideal_point(self, pop):
        for individual in pop:
            for i in range(self.problem.obj_nums):
                self.ideal_point.obj[i] = min(self.ideal_point.obj[i], individual.F_values[i])

    def update_ideal_point(self, _offspring):
        for i in range(self.problem.obj_nums):
            if _offspring.F_values[i] < self.ideal_point.obj[i]:
                self.ideal_point.obj[i] = _offspring.F_values[i]

    def set_weight(self, weight, unit, sum, dim, column):
        if dim == self.problem.obj_nums:
            for i in range(self.problem.obj_nums):
                weight[i] = 0
        if dim == 1:
            weight[0] = unit - sum
            for i in range(self.problem.obj_nums):
                self.lambda_list[column][i] = weight[i]
            column = column + 1
            return column

        for i in range(unit - sum + 1):
            weight[dim - 1] = i
            column = self.set_weight(weight, unit, sum + i, dim - 1, column)

        return column

    def initialize_uniform_point(self):
        vec = [0] * self.problem.obj_nums
        gaps = 1
        column = 0
        while True:
            layer_size = combination(self.problem.obj_nums + gaps - 1, gaps)
            if layer_size > self.max_population_size:
                break
            self.num_weight = layer_size
            gaps += 1

        gaps -= 1
        self.lambda_list = [[0] * self.problem.obj_nums for _ in range(self.num_weight)]

        for i in range(self.problem.obj_nums):
            vec[i] = 0
        self.set_weight(vec, gaps, 0, self.problem.obj_nums, column)
        for i in range(self.num_weight):
            for j in range(self.problem.obj_nums):
                self.lambda_list[i][j] = self.lambda_list[i][j] / gaps

        return self.lambda_list

    def init_distance_list(self):
        sort_list = [DistInfo() for _ in range(self.max_population_size)]
        for i in range(self.num_weight):
            for j in range(self.num_weight):
                distance_temp = 0
                for k in range(self.problem.obj_nums):
                    difference = abs(self.lambda_list[i][k] - self.lambda_list[j][k])
                    distance_temp += (difference * difference)

                euc_distance = math.sqrt(distance_temp)
                sort_list[j].value = euc_distance
                sort_list[j].idx = j

            distance_quick_sort(sort_list, 0, self.num_weight - 1)

            for j in range(EMOOConfig.neighbor_size):
                self.neighbor_table[i].neighbor[j] = sort_list[j].idx

    def initialize(self):
        self.init_distance_list()
        self.parent_population = Population(self.max_population_size)
        self.offspring_population = Population(self.max_population_size)
        self.init_ideal_point(self.parent_population)
        self.evaluate_population()
        self.cal_moead_fitness()
        return self

    def cal_TCH_fitness(self, individual, weight_vector):
        eps = 10e-7
        max_fit = -1.0e30
        for i in range(self.problem.obj_nums):
            diff = abs(individual.F_values[i] - self.ideal_point.obj[i])

            if weight_vector[i] < eps:
                fitness = diff * 0.000001
            else:
                fitness = diff * weight_vector[i]

            if max_fit < fitness:
                max_fit = fitness

        fitness = max_fit
        individual.fitness = fitness
        return fitness

    def cal_moead_fitness(self):
        for i in range(self.num_weight):
            self.cal_TCH_fitness(self.parent_population[i], self.lambda_list[i])

    def update_moead_fitness(self, _offspring, lam):
        return self.cal_TCH_fitness(_offspring, lam)

    def crossover_MOEAD(self, parent, parent_idx, neighbor_type):
        selected_flag = [0] * self.max_population_size
        parent_id = [0, 0]
        choose_num = 0

        while choose_num < 2:

            if neighbor_type == 'neighbor':
                rand = np.random.randint(0, EMOOConfig.neighbor_size)
                selected_id = self.neighbor_table[parent_idx].neighbor[rand]

            else:
                rand = np.random.randint(0, self.num_weight)
                selected_id = rand

            if selected_flag[selected_id] == 0:
                selected_flag[selected_id] = 1
                parent_id[choose_num] = selected_id
                choose_num += 1

        return self.de_crossover(parent, self.parent_population[parent_id[0]],
                                 self.parent_population[parent_id[0]])

    def de_crossover(self, parent1, parent2, parent3):
        _offspring = Individual()
        r = np.random.randint(self.problem.var_nums)
        for i in range(self.problem.var_nums):
            yl = self.problem.var_bound_l[i]
            yu = self.problem.var_bound_h[i]
            if np.random.random() < EMOOConfig.de_p_crossover or i == r:
                value = parent3.variables[i] + \
                        EMOOConfig.de_F * (parent1.variables[i] - parent2.variables[i])
                value = max(yl, value)
                value = min(yu, value)

            else:
                value = parent3.variables[i]

            _offspring.variables[i] = value

        return _offspring

    def mutation_ind(self, individual):
        self.operator.mutation_individual(individual)
        return individual

    def update_subproblem(self, _offspring, pop_index, neighbor_type):

        replace_num = 0

        if 'neighbor' == neighbor_type:
            size = EMOOConfig.neighbor_size
        else:
            size = self.max_population_size

        perm = random_permutation(size)
        # perm = np.random.permutation(size)
        for i in range(size):
            if replace_num >= EMOOConfig.de_maximumNumberOfReplacedSolutions:
                break

            if 'neighbor' == neighbor_type:
                index = self.neighbor_table[pop_index].neighbor[perm[i]]
            else:
                index = perm[i]
                if index >= self.num_weight:
                    index = np.random.randint(self.num_weight-1)
                # while perm[i]

            temp = self.update_moead_fitness(_offspring, self.lambda_list[index])
            old_fit = self.update_moead_fitness(self.parent_population[index], self.lambda_list[index])
            if temp < old_fit:
                self.parent_population[index].F_values = _offspring.F_values
                self.parent_population[index].variables = _offspring.variables
                self.parent_population[index].fitness = temp
                replace_num += 1

    def evolve(self, visualize_steps=None):

        for generation_id in tqdm.tqdm(range(self.max_evolution_iteration), postfix='evolving...'):

            #  crossover and mutation
            for parent_idx in range(self.num_weight):

                rand = np.random.rand()
                if rand < EMOOConfig.p_neighbor_select:
                    neighbor_type = 'neighbor'
                else:
                    neighbor_type = 'global_parent'

                parent = self.parent_population[parent_idx]

                _offspring = self.crossover_MOEAD(parent, parent_idx, neighbor_type)

                # _offspring = self.mutation_ind(_offspring)
                _offspring = self.mutation_ind(_offspring)

                self.evaluate_individual(_offspring)

                self.update_ideal_point(_offspring)

                self.update_subproblem(_offspring, parent_idx, neighbor_type)

            if visualize_steps and generation_id % visualize_steps == 0:
                self.get_indicator(visualize=True, generation_id=generation_id)

        # self.evaluate_population()
        # for generation_id in tqdm.tqdm(range(self.max_evolution_iteration), postfix='evolving...'):
        #     self.offspring_population = self.operator.crossover(self.parent_population)
        #     self.offspring_population = self.operator.mutation_pop(self.offspring_population)
        #     self.evaluate_population()
        #     self.mixed_population = self.operator.merge_population(self.parent_population, self.offspring_population)
        #     self.parent_population = self.operator.nsga2_select(self.parent_population, self.mixed_population)
        #     self.evaluate_population()
        #     if visualize_steps and generation_id % visualize_steps == 0:
        #         self.get_indicator(visualize=True, generation_id=generation_id)

        self.get_indicator(visualize=True, generation_id=self.max_evolution_iteration)

    def get_indicator(self, visualize=False, generation_id=0):
        self.objectives = []
        self.solutions = []
        for individual in self.parent_population:
            self.objectives.append([individual.F_values[i] for i in range(len(individual.F_values))])
            self.solutions.append([individual.variables[i] for i in range(len(individual.variables))])

        objectives = np.array(self.objectives)
        if visualize:
            plot_title = '{} on problem {} pop_size:{} generation:{}'.format(
                self.name,
                self.problem.problem_name,
                self.max_population_size,
                generation_id
            )
            self.plot = Scatter(title=plot_title)
            self.plot.add(objectives, color='red', s=15, marker='.')
            if len(self.parent_population[0].F_values) == 2:
                test_problem = get_problem(self.problem.problem_name)
                self.plot.add(test_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            self.plot.show()
        if not isinstance(self.ref_point, list):
            ref = np.array([self.ref_point] * len(objectives[0])).astype(float)
        else:
            ref = np.array(self.ref_point).astype(float)
        if EMOOConfig.obj_norm:
            objectives = normalize(objectives)
        self.hv = hvwfg.wfg(objectives, ref)

        return self.hv

    def evaluate_population(self):
        if self.offspring_population:
            self.offspring_population.update_population_objectives()
        if self.parent_population:
            self.parent_population.update_population_objectives()


    def evaluate_individual(self, individual):
        individual.update_objectives()

def combination(n, k):
    if n < k:
        return -1
    ans = 1.
    for i in range(k + 1, n + 1):
        ans = ans * i
        ans = ans / (i - k)

    return int(ans)


def distance_quick_sort(distanceInfo, left, right):
    if left < right:
        pos = partition_by_distance(distanceInfo, left, right)
        distance_quick_sort(distanceInfo, pos + 1, right)
        distance_quick_sort(distanceInfo, left, pos - 1)


def partition_by_distance(distanceInfo, left, right):
    temp_fit = distanceInfo[left].value
    temp_index = distanceInfo[left].idx
    while left < right:

        while left < right and distanceInfo[right].value >= temp_fit:
            right -= 1

        if left < right:
            distanceInfo[left].idx = distanceInfo[right].idx
            distanceInfo[left].value = distanceInfo[right].value
            left += 1

        while left < right and distanceInfo[left].value < temp_fit:
            left += 1

        if left < right:
            distanceInfo[right].idx = distanceInfo[left].idx
            distanceInfo[right].value = distanceInfo[left].value
            right -= 1

    distanceInfo[left].value = temp_fit
    distanceInfo[left].idx = temp_index

    return left


def random_permutation(size):
    index = [i for i in range(size)]
    flag = [1 for _ in range(size)]
    perm = [0] * size
    num = 0
    while num < size:

        start = np.random.randint(0, size)
        while True:

            if flag[start]:
                perm[num] = index[start]
                flag[start] = 0
                num += 1
                break

            if start == (size - 1):
                start = 0
            else:
                start += 1

    return perm

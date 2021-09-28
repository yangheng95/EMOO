from tkinter import _flatten

import numpy as np
from hvwfg import wfg

import EMOOConfig
from utils.crossover import SBX
from utils.mutation import PolynomialMutation
from utils.population import Population
from utils.selection import TournamentSelection, check_dominance


class Operator:

    def __init__(self, algo, problem=None):
        self.algo = algo
        self.crossover_rate = EMOOConfig.p_crossover
        self.mutation_rate = EMOOConfig.p_mutation
        self.problem = problem
        self.SBX_operator = SBX(self.problem, eta=EMOOConfig.eta_crossover)
        self.Mutate_operator = PolynomialMutation(self.problem, eta=EMOOConfig.eta_mutation)
        self.select_operator = TournamentSelection()
        self.max_indicator_value = -np.inf
        self.indicator_array = None

    # refactored
    def crossover(self, parent_pop):
        parent_pop_ = Population(population_size=parent_pop.population_size, individuals=parent_pop.individuals)
        winners = self.select_operator.select(parent_pop_)
        for i in range(0, len(winners), 2):
            # if np.random.rand() < self.crossover_rate:
                winners[i].variables, winners[i + 1].variables = self.SBX_operator.sbx_crossover(
                    np.array([[winners[i].variables], [winners[i + 1].variables]])
                )

        return winners

    def mutation_pop(self, off_population):
        for i in range(len(off_population)):
            off_population[i].variables = self.Mutate_operator.mutate(np.array([off_population[i].variables]))
        return off_population

    def mutation_individual(self, individual):
        individual.variables = self.Mutate_operator.mutate(np.array([individual.variables]), 2)
        return individual

    def nsga2_select(self, parent_pop, mixed_pop):
        # mixed_pop = self.select_operator.select(mixed_pop)
        fronts = list(self.fast_non_dominated_sort(mixed_pop))
        PF = [mixed_pop[idx] for idx in fronts[0]]
        solutions_sorted_by_fronts = _flatten(list(fronts[1:]))
        parent_pop.individuals = PF
        idx = 0
        while len(parent_pop) < parent_pop.population_size:
            parent_pop.individuals.append(mixed_pop[solutions_sorted_by_fronts[idx]])
            idx += 1
        while len(parent_pop) > parent_pop.population_size:
            rand = np.random.randint(0, len(parent_pop) - 1)
            parent_pop.individuals.remove(parent_pop[rand])

        return parent_pop

    def merge_population(self, parent_pop, off_pop):
        mix_pop = []
        for individual in (parent_pop.individuals + off_pop.individuals):
            mix_pop.append(individual)
        mix_pop = Population(len(mix_pop), mix_pop)
        return mix_pop

    def fast_non_dominated_sort(self, pop):
        F = [[individual.F_values[i] for i in range(len(individual.F_values))] for individual in pop]
        M = self.calc_domination_matrix(np.array(F))
        # calculate the dominance matrix
        n = M.shape[0]
        fronts = []
        # final rank that will be returned
        n_ranked = 0
        ranked = np.zeros(n, dtype=int)

        # for each individual a list of all individuals that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)
        current_front = []

        for i in range(n):
            for j in range(i + 1, n):
                rel = M[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        # append the first front to the current front
        fronts.append(current_front)

        # while not all solutions are assigned to a pareto front
        while n_ranked < n:
            next_front = []
            # for each individual in the current front
            for i in current_front:
                # all solutions that are dominated by this individuals
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts

    def calc_domination_matrix(self, F, epsilon=0.0):
        _F = F
        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1
        return M

    def ibea_select(self, parent_population, mixed_pop):
        objectives = np.array([individual.F_values for individual in mixed_pop]).astype(np.float64)
        # objectives = self.scale_F_values(objectives)
        self.fitness_assign(mixed_pop, objectives)
        eliminate_flags = [0] * len(mixed_pop)  # reachable individual
        while np.count_nonzero(eliminate_flags) < len(parent_population):
            min_fitness = np.inf
            worst_idx = 0
            for i in range(len(mixed_pop)):
                if mixed_pop[i].fitness < min_fitness and not eliminate_flags[i]:
                    min_fitness = mixed_pop[i].fitness
                    worst_idx = i
            eliminate_flags[worst_idx] = 1
            self.update_fitness(eliminate_flags, worst_idx, objectives, mixed_pop)
        for i in reversed(range(len(eliminate_flags))):
            if eliminate_flags[i]:
                mixed_pop.individuals.remove(mixed_pop[i])
        parent_population = Population(len(mixed_pop.individuals), mixed_pop.individuals)
        return parent_population

    def update_fitness(self, eliminate_flags, worst_idx, objectives, pop):
        for i, obj in enumerate(objectives):
            if worst_idx != i and not eliminate_flags[i]:
                pop[i].fitness += np.exp(
                    -self.indicator_array[worst_idx][i] / self.max_indicator_value / EMOOConfig.kappa
                )

    def scale_F_values(self, objectives):
        _min = objectives.min(axis=0)
        _max = objectives.max(axis=0)
        _, ndims = objectives.shape
        for dim in range(ndims):
            objectives[:, dim] = (objectives[:, dim] - _min[dim]) / (_max[dim] - _min[dim])
        return objectives

    def get_indicator_value(self, objective1, objective2):
        if self.algo.name == 'IBEA_EPLUS':
            indicator_value = max(objective1 - objective2)
        elif self.algo.name == 'IBEA_HV':
            ref = np.array([EMOOConfig.ref_point] * len(objective1)).astype(float)
            relation = check_dominance(objective1, objective2)
            if relation > 0:
                indicator_value = wfg(np.array([objective2]), ref) - wfg(np.array([objective1]), ref)
            else:
                indicator_value = wfg(np.array([objective1, objective2]), ref) - wfg(np.array([objective1]), ref)
        else:
            raise NotImplementedError('Unimplemented indicator')
        return indicator_value

    def get_max_indicator_value(self, objectives):
        self.indicator_array = [[0 for _ in range(objectives.shape[0])] for _ in range(objectives.shape[0])]
        for i, obj1 in enumerate(objectives):
            for j, obj2 in enumerate(objectives):
                self.indicator_array[i][j] = self.get_indicator_value(objectives[i], objectives[j])
                self.max_indicator_value = max(self.max_indicator_value, abs(self.indicator_array[i][j]))
        self.max_indicator_value = max(10e-10, self.max_indicator_value)
        return self.max_indicator_value

    def fitness_assign(self, mixed_pop, objectives):
        max_indicator_value = self.get_max_indicator_value(objectives)
        for i in range(objectives.shape[0]):
            fit_sum = 0
            for j in range(objectives.shape[0]):
                if i != j:
                    fit_sum += - np.exp(- self.indicator_array[j][i] / max_indicator_value / EMOOConfig.kappa)
            mixed_pop[i].fitness = fit_sum

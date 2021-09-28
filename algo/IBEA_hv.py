import math
import EMOOConfig
import hvwfg
import numpy as np
import tqdm
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

from utils.individual import Individual
from utils.operators import Operator
from utils.population import Population
from utils.file_utils import normalize


class IBEA_HV:

    def __init__(
            self,
            ref_point=1,
            indicator=None,
            problem=None
    ):
        self.name = 'IBEA_HV'
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

    def set_problem(self, problem=None, ref_point=None):
        if not problem or not ref_point:
            raise ValueError('specify a test problem and reference point!')
        self.max_population_size = EMOOConfig.population_size
        self.max_evolution_iteration = EMOOConfig.evolution_iteration
        self.operator = Operator(self, problem=problem)
        self.problem = problem
        Individual.problem = problem
        self.parent_population = Population(self.max_population_size)
        self.ref_point = ref_point

    def initialize(self):
        self.parent_population = Population(self.max_population_size)

        self.evaluate_population()

        return self

    def evolve(self, visualize_steps=None):

        for generation_id in tqdm.tqdm(range(self.max_evolution_iteration), postfix='evolving...'):

            self.offspring_population = self.operator.crossover(self.parent_population)

            self.offspring_population = self.operator.mutation_pop(self.offspring_population)

            self.evaluate_population()

            self.mixed_population = self.operator.merge_population(self.parent_population, self.offspring_population)

            self.parent_population = self.operator.ibea_select(self.parent_population, self.mixed_population)

            self.evaluate_population()

            if visualize_steps and generation_id % visualize_steps == 0:
                self.get_indicator(visualize=True, generation_id=generation_id)

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
        self.hv = hvwfg.wfg(objectives, ref)

        return self.hv

    def evaluate_population(self):
        if self.offspring_population:
            self.offspring_population.update_population_objectives()
        if self.parent_population:
            self.parent_population.update_population_objectives()

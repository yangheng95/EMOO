algo = 'ibea_eplus'
# algo = 'nsga2'
# algo = 'moead'
# problem = 'dtlz5'
problem = ''
population_size = 300
evolution_iteration = 250
p_crossover = 1
p_mutation = 1

de_p_crossover = 0.5
de_F = 0.5
de_maximumNumberOfReplacedSolutions = 20
p_neighbor_select = 0.9
neighbor_size = 20

kappa = 0.05

eta_crossover = 15
eta_mutation = 20
exp_rounds = 30
ref_point = 1.1  # automatic expand dimension
obj_norm = False  # normalize objectives
watching_step = 0


# --------------------------------------------------------------------------------- #
from utils.problems import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from collections import OrderedDict

# algo = algo_set[algo]()
problem_set = OrderedDict(

    dtlz7=DTLZ7()
)

evaluate_problems = [problem_set[problem]] if problem in problem_set else problem_set.values()

def load_default_parameter(problem_name=None):
    point = 1.1
    if problem_name in 'zdt1':
        generation = 250
        population = 100
    elif problem_name == 'zdt2':
        generation = 250
        population = 100
    elif problem_name == 'zdt3':
        generation = 250
        population = 100
    elif problem_name == 'zdt4':
        generation = 400
        population = 100
    elif problem_name == 'zdt6':
        generation = 250
        population = 100
    elif problem_name == 'dtlz1':
        generation = 300
        population = 100
    elif problem_name == 'dtlz2':
        generation = 300
        population = 100
    elif problem_name == 'dtlz3':
        point = 1.1
        generation = 800
        population = 100
    elif problem_name == 'dtlz4':
        generation = 200
        population = 100
    elif problem_name == 'dtlz5':
        generation = 200
        population = 100
    elif problem_name == 'dtlz6':
        generation = 500
        population = 100
    elif problem_name == 'dtlz7':
        generation = 200
        population = 100
        point = [1.1, 1.1, 8]
    else:
        raise NotImplementedError('Unimplemented test problem!')
    return generation, population, point

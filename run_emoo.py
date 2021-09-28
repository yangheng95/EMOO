import EMOOConfig
from EMOOConfig import load_default_parameter
from utils.file_utils import save_result

# --------------------------------------------------------------------------------- #
from algo.IBEA_eplus import IBEA_EPLUS
from algo.IBEA_hv import IBEA_HV
from algo.NSGA2 import NSGA2
from algo.MOEADDE import MOEADDE
algo_set = {'nsga2': NSGA2, 'ibea_hv': IBEA_HV, 'ibea_eplus': IBEA_EPLUS, 'moead': MOEADDE}


if __name__ == '__main__':
    print('Algo:', EMOOConfig.algo, 'problem set:', [p.problem_name for p in EMOOConfig.evaluate_problems])
    for p in EMOOConfig.evaluate_problems:

        # reload default parameter setting
        generation, population, ref_point = load_default_parameter(p.problem_name.lower())
        if 'moead' not in EMOOConfig.algo:
            EMOOConfig.evolution_iteration = generation
            EMOOConfig.population_size = population
            EMOOConfig.ref_point = ref_point
        algo = algo_set[EMOOConfig.algo]()
        algo.set_problem(p, ref_point=EMOOConfig.ref_point)  # set test problem

        print('Algo:', algo.name,
              'problem:', p.problem_name,
              'pop_size:', algo.max_population_size,
              'generations:', algo.max_evolution_iteration
              )

        indicator_path = 'output/hv/algo_{}_problem_{}_generation_{}_popsize_{}.hv'.format(
            algo.name,
            algo.problem.problem_name,
            algo.max_evolution_iteration,
            algo.max_population_size
        )
        fout = open(indicator_path, 'w')
        indicators = []
        for i in range(EMOOConfig.exp_rounds):
            objective_path = 'output/objective/algo_{}_problem_{}_{}th_generation_{}_popsize_{}.objective'.format(
                algo.name,
                algo.problem.problem_name,
                i + 1,
                algo.max_evolution_iteration,
                algo.parent_population.population_size
            )
            solution_path = 'output/solution/algo_{}_problem_{}_{}th_generation_{}_popsize_{}.solution'.format(
                algo.name,
                algo.problem.problem_name,
                i + 1,
                algo.max_evolution_iteration,
                algo.parent_population.population_size
            )
            print(i + 1, 'th Running')
            algo = algo.initialize()
            algo.evolve(EMOOConfig.watching_step)
            indicators.append(algo.hv)
            print(algo.hv)
            save_result(algo, objective_path=objective_path, solution_path=solution_path)
            fout.write('{}th Hyper volume:{} \t ref point:({}) \n'.format(
                i + 1,
                algo.hv,
                ','.join([str(EMOOConfig.ref_point)] * len(algo.parent_population[0].F_values)))
            )
        print('Avg HV:', sum(indicators)/len(indicators))
        fout.write('Avg HV:'+str(sum(indicators)/len(indicators)))
        fout.close()

    exit(0)

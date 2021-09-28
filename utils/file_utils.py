import numpy as np


def read_hv(hv_path):
    fin = open(hv_path, 'r')
    lines = fin.readlines()
    hv = []
    for line in lines:
        line = line.split(':')[1].split()[0]
        hv.append(np.array(float(line)))
    return hv


def get_avg_hv(hv_path):
    fin = open(hv_path, 'r')
    lines = fin.readlines()
    hv = []
    for line in lines:
        if 'Avg HV' not in line:
            line = line.split(':')[1].split()[0]
            hv.append(np.array(float(line)))
    return np.average(hv)


def read_objective(objective_path):
    fin = open(objective_path, 'r')
    lines = fin.readlines()
    objectives = []
    for line in lines:
        objectives.append(np.array([float(s) for s in line.split()]))
    return objectives


def save_result(algo, objective_path, solution_path, figure_path=None):
    figure_path = objective_path.replace('objective', 'figure')
    obj_fout = open(objective_path, mode='w')
    solution_fout = open(solution_path, mode='w')
    for objective in algo.objectives:
        objective = ' '.join([str(float(objective[i])) for i in range(len(objective))]) + '\n'
        obj_fout.write(objective)

    for solution in algo.solutions:
        solution = ' '.join([str(float(solution[i])) for i in range(len(solution))]) + '\n'
        solution_fout.write(solution)
    algo.plot.save(figure_path + '.png')

    obj_fout.close()
    solution_fout.close()


def normalize(x):
    x = np.array(x).astype(float)
    x_min_ = np.min(x)
    x_max_ = np.max(x)
    x = (x - x_min_) / (x_max_ - x_min_)
    return x

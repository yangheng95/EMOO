import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from utils.file_utils import read_objective


def get_pf_approximation(algo, problem):
    algo = algo.lower()
    pf_approximation_paths = {
        'nsga2_zdt1':'algo_NSGA2_problem_zdt1_th_generation_250_popsize_100.objective',
        'nsga2_zdt2':'algo_NSGA2_problem_zdt2_th_generation_250_popsize_100.objective',
        'nsga2_zdt3':'algo_NSGA2_problem_zdt3_th_generation_250_popsize_100.objective',
        'nsga2_zdt4':'algo_NSGA2_problem_zdt4_th_generation_400_popsize_100.objective',
        'nsga2_zdt6':'algo_NSGA2_problem_zdt6_th_generation_250_popsize_100.objective',
        'nsga2_dtlz1':'algo_NSGA2_problem_dtlz1_th_generation_300_popsize_100.objective',
        'nsga2_dtlz2':'algo_NSGA2_problem_dtlz2_th_generation_300_popsize_100.objective',
        'nsga2_dtlz3':'algo_NSGA2_problem_dtlz3_th_generation_800_popsize_100.objective',
        'nsga2_dtlz4':'algo_NSGA2_problem_dtlz4_th_generation_200_popsize_100.objective',
        'nsga2_dtlz5':'algo_NSGA2_problem_dtlz5_th_generation_200_popsize_100.objective',
        'nsga2_dtlz6':'algo_NSGA2_problem_dtlz6_th_generation_500_popsize_100.objective',
        'nsga2_dtlz7':'algo_NSGA2_problem_dtlz7_th_generation_200_popsize_100.objective',

        'ibea_eplus_zdt1': 'algo_IBEA_EPLUS_problem_zdt1_th_generation_250_popsize_100.objective',
        'ibea_eplus_zdt2': 'algo_IBEA_EPLUS_problem_zdt2_th_generation_250_popsize_100.objective',
        'ibea_eplus_zdt3': 'algo_IBEA_EPLUS_problem_zdt3_th_generation_250_popsize_100.objective',
        'ibea_eplus_zdt4': 'algo_IBEA_EPLUS_problem_zdt4_th_generation_400_popsize_100.objective',
        'ibea_eplus_zdt6': 'algo_IBEA_EPLUS_problem_zdt6_th_generation_250_popsize_100.objective',
        'ibea_eplus_dtlz1': 'algo_IBEA_EPLUS_problem_dtlz1_th_generation_300_popsize_100.objective',
        'ibea_eplus_dtlz2': 'algo_IBEA_EPLUS_problem_dtlz2_th_generation_300_popsize_100.objective',
        'ibea_eplus_dtlz3': 'algo_IBEA_EPLUS_problem_dtlz3_th_generation_800_popsize_100.objective',
        'ibea_eplus_dtlz4': 'algo_IBEA_EPLUS_problem_dtlz4_th_generation_200_popsize_100.objective',
        'ibea_eplus_dtlz5': 'algo_IBEA_EPLUS_problem_dtlz5_th_generation_200_popsize_100.objective',
        'ibea_eplus_dtlz6': 'algo_IBEA_EPLUS_problem_dtlz6_th_generation_500_popsize_100.objective',
        'ibea_eplus_dtlz7': 'algo_iIBEA_EPLUS_problem_dtlz7_th_generation_200_popsize_100.objective',

        'moeadde_zdt1': 'algo_MOEADDE_problem_zdt1_th_generation_250_popsize_100.objective',
        'moeadde_zdt2': 'algo_MOEADDE_problem_zdt2_th_generation_250_popsize_100.objective',
        'moeadde_zdt3': 'algo_MOEADDE_problem_zdt3_th_generation_250_popsize_100.objective',
        'moeadde_zdt4': 'algo_MOEADDE_problem_zdt4_th_generation_250_popsize_100.objective',
        'moeadde_zdt6': 'algo_MOEADDE_problem_zdt6_th_generation_250_popsize_100.objective',
        'moeadde_dtlz1': 'algo_MOEADDE_problem_dtlz1_th_generation_250_popsize_300.objective',
        'moeadde_dtlz2': 'algo_MOEADDE_problem_dtlz2_th_generation_250_popsize_300.objective',
        'moeadde_dtlz3': 'algo_MOEADDE_problem_dtlz3_th_generation_250_popsize_300.objective',
        'moeadde_dtlz4': 'algo_MOEADDE_problem_dtlz4_th_generation_250_popsize_300.objective',
        'moeadde_dtlz5': 'algo_MOEADDE_problem_dtlz5_th_generation_250_popsize_300.objective',
        'moeadde_dtlz6': 'algo_MOEADDE_problem_dtlz6_th_generation_250_popsize_300.objective',
        'moeadde_dtlz7': 'algo_MOEADDE_problem_dtlz7_th_generation_250_popsize_300.objective',

    }
    pf_approximation_path = 'output/objective/'+pf_approximation_paths[algo+'_'+problem]
    pf_approximations = []
    pf_path = {
        'dtlz1': 'output/pf/dtlz1.3D.pf',
        'dtlz2': 'output/pf/dtlz2.3D.pf',
        'dtlz3': 'output/pf/dtlz3.3D.pf',
        'dtlz4': 'output/pf/dtlz4.3D.pf',
        'dtlz5': 'output/pf/dtlz5.3D.pf',
        'zdt1': 'output/pf/zdt1.2D.pf',
        'zdt2': 'output/pf/zdt2.2D.pf',
        'zdt3': 'output/pf/zdt3.2D.pf',
        'zdt4': 'output/pf/zdt4.2D.pf',
        'zdt6': 'output/pf/zdt6.2D.pf',
    }
    for i in range(3):
        pf_approximations.append(read_objective(pf_approximation_path.replace('th', str(i + 1) + 'th')))
    pf = read_objective(pf_path[problem])
    return pf_approximations, pf


def visualize(algo='IBEA_HV', problem='dtlz1'):
    if problem in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5'}:
        visualize_3D(algo, problem)
    elif problem in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'}:
        visualize_2D(algo, problem)
    else:
        raise RuntimeError('No result')


def visualize_3D(algo, problem='dtlz1'):
    assert problem in {'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5'}
    pf_approximations, pf = get_pf_approximation(algo, problem)

    pf1 = np.array(pf_approximations[0])
    pf2 = np.array(pf_approximations[1])
    pf3 = np.array(pf_approximations[2])
    pf = np.array(pf)

    x1 = pf1[:, 0]
    y1 = pf1[:, 1]
    z1 = pf1[:, 2]

    x2 = pf2[:, 0]
    y2 = pf2[:, 1]
    z2 = pf2[:, 2]

    x3 = pf3[:, 0]
    y3 = pf3[:, 1]
    z3 = pf3[:, 2]

    x4 = pf[:, 0]
    y4 = pf[:, 1]
    z4 = pf[:, 2]

    # 绘制散点图
    fig = plt.figure()
    fig.suptitle(algo + ' on ' + problem)
    ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, s=2, c='r', label='pf_approximation1')
    ax.scatter(x2, y2, z2, s=2, c='b', label='pf_approximation2')
    ax.scatter(x3, y3, z3, s=2, c='g', label='pf_approximation3')
    ax.scatter(x4, y4, z4, s=2, c='y', label='pareto-optimal front')

    # 绘制图例
    ax.legend(loc='best')

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('f3', fontdict={'size': 10, 'color': 'black'})
    ax.set_ylabel('f2', fontdict={'size': 10, 'color': 'black'})
    ax.set_xlabel('f1', fontdict={'size': 10, 'color': 'black'})

    # 展示
    plt.show()


def visualize_2D(algo, problem='dtlz1'):
    assert problem in {'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'}
    pf_approximations, pf = get_pf_approximation(algo, problem)

    pf1 = np.array(pf_approximations[0])
    pf2 = np.array(pf_approximations[1])
    pf3 = np.array(pf_approximations[2])
    pf = np.array(pf)

    x1 = pf1[:, 0]
    y1 = pf1[:, 1]

    x2 = pf2[:, 0]
    y2 = pf2[:, 1]

    x3 = pf3[:, 0]
    y3 = pf3[:, 1]

    x4 = pf[:, 0]
    y4 = pf[:, 1]

    # 绘制散点图
    fig = plt.figure()
    fig.suptitle(algo + ' on ' + problem)
    ax = Axes3D(fig)
    ax.scatter(x1, y1, s=2, c='r', label='pf_approximation1')
    ax.scatter(x2, y2, s=2, c='b', label='pf_approximation2')
    ax.scatter(x3, y3, s=2, c='g', label='pf_approximation3')
    ax.scatter(x4, y4, s=0.1, c='y', label='pareto-optimal front')

    # 绘制图例
    ax.legend(loc='best')

    # 添加坐标轴(顺序是Z, Y, X)
    # ax.set_zlabel('f3', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('f2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('f1', fontdict={'size': 15, 'color': 'red'})

    # 展示
    plt.show()


if __name__ == '__main__':
    # visualize(algo='NSGA2', problem='dtlz1')
    # visualize(algo='NSGA2', problem='dtlz2')
    # visualize(algo='NSGA2', problem='dtlz3')
    # visualize(algo='NSGA2', problem='dtlz4')
    # visualize(algo='NSGA2', problem='dtlz5')

    # visualize(algo='IBEA_HV', problem='dtlz1')
    # visualize(algo='IBEA_HV', problem='dtlz2')
    # visualize(algo='IBEA_HV', problem='dtlz3')
    # visualize(algo='IBEA_HV', problem='dtlz4')
    # visualize(algo='IBEA_HV', problem='dtlz5')

    # visualize(algo='IBEA_EPLUS', problem='dtlz1')
    # visualize(algo='IBEA_EPLUS', problem='dtlz2')
    # visualize(algo='IBEA_EPLUS', problem='dtlz3')
    # visualize(algo='IBEA_EPLUS', problem='dtlz4')
    # visualize(algo='IBEA_EPLUS', problem='dtlz5')

    # visualize(algo='MOEADDE', problem='dtlz1')
    visualize(algo='MOEADDE', problem='dtlz2')
    visualize(algo='MOEADDE', problem='dtlz3')
    visualize(algo='MOEADDE', problem='dtlz4')
    visualize(algo='MOEADDE', problem='dtlz5')

exit(0)

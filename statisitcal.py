from scipy.stats import ranksums

from utils.file_utils import read_hv, get_avg_hv

hv_path = 'output_old/hv/algo_IBEA_EPLUS_problem_zdt6_generation_300_popsize200.log'


def get_hv(hv_path=''):
    return read_hv(hv_path)


def cal_avg_hv(algo='NSGA2', problem='zdt6'):
    hv_path = 'output/hv/algo_' + algo \
              + '_problem_' + problem + '_generation_300_popsize_500.hv'
    print(get_avg_hv(hv_path))
    return get_avg_hv(hv_path)


# def cal_wilcoxon(algo1, algo2, problem):
#     hv1 = get_hv(algo1, problem)
#     hv2 = get_hv(algo2, problem)
#     # result = wilcoxon(hv1, hv2)
#     result = ranksums(hv1, hv2)
#     print('statistic:', result[0], 'p-value:', result[1])
#     return result

def cal_wilcoxon(hv_path1, hv_path2):
    hv1 = get_hv(hv_path1)
    hv2 = get_hv(hv_path2)
    # result = wilcoxon(hv1, hv2)
    result = ranksums(hv1, hv2)
    print('statistic:', result[0], 'p-value:', result[1])
    return result



if __name__ == '__main__':
    # cal_avg_hv(algo='IBEA_HV', problem='zdt1')
    # cal_avg_hv(algo='IBEA_HV', problem='zdt2')
    # cal_avg_hv(algo='IBEA_HV', problem='zdt3')
    # cal_avg_hv(algo='IBEA_HV', problem='zdt4')
    # cal_avg_hv(algo='IBEA_HV', problem='zdt6')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz1')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz2')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz3')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz4')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz5')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz6')
    # cal_avg_hv(algo='IBEA_HV', problem='dtlz7')

    # cal_avg_hv(algo='IBEA_EPLUS', problem='zdt1')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='zdt2')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='zdt3')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='zdt4')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='zdt6')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz1')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz2')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz3')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz4')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz5')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz6')
    # cal_avg_hv(algo='IBEA_EPLUS', problem='dtlz7')

    # cal_avg_hv(algo='NSGA2', problem='zdt1')
    # cal_avg_hv(algo='NSGA2', problem='zdt2')
    # cal_avg_hv(algo='NSGA2', problem='zdt3')
    # cal_avg_hv(algo='NSGA2', problem='zdt4')
    # cal_avg_hv(algo='NSGA2', problem='zdt6')
    # cal_avg_hv(algo='NSGA2', problem='dtlz1')
    # cal_avg_hv(algo='NSGA2', problem='dtlz2')
    # cal_avg_hv(algo='NSGA2', problem='dtlz3')
    # cal_avg_hv(algo='NSGA2', problem='dtlz4')
    # cal_avg_hv(algo='NSGA2', problem='dtlz5')
    # cal_avg_hv(algo='NSGA2', problem='dtlz6')
    # cal_avg_hv(algo='NSGA2', problem='dtlz7')
    hv_path1 = 'output/hv/algo_' + 'IBEA_EPLUS' +'_problem_' + 'dtlz7' + '_generation_200_popsize_100.hv'
    hv_path2 = 'output/hv/algo_' + 'MOEADDE' +'_problem_' + 'dtlz7' + '_generation_300_popsize_300.hv'
    cal_wilcoxon(hv_path1, hv_path2)

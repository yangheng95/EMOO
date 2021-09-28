import math

import numpy as np
from pymoo.util.misc import random_permuations
import copy

class TournamentSelection:
    """
      The Tournament selection is used to simulated a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, pressure=2):
        self.pressure = pressure

    def binary_tournament(self, pop, P):

        S = np.full(P.shape[0], np.nan)

        for i in range(P.shape[0]):
            a, b = P[i, 0], P[i, 1]
            rel = check_dominance(pop[a].F_values, pop[b].F_values)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b
            else:
                pass

                # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].crowding_distance, b, pop[b].crowding_distance, method='larger_is_better')

        return S[:, None].astype(int, copy=False)

    def select(self, pop, n_parents=1):
        n_select = int(len(pop) / n_parents)
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure
        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))
        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.binary_tournament(pop, P)
        # S = self.binary_tournament(pop, S.reshape(int(S.shape[0]/2), -1))
        new_individuals = []
        # for idx in S:
        #     new_individuals.append(copy.deepcopy(pop[int(idx)]))
        # pop.individuals = new_individuals

        for idx in S:
            # if pop[int(idx)] and pop[int(idx)] not in new_individuals:
            #     new_individuals.append(pop[int(idx)])
            # else:
            #     rand = np.random.randint(0, len(pop))
            #     while pop[rand] in new_individuals:
            #         rand = np.random.randint(0, len(pop))
            #     new_individuals.append(pop[rand])
            # new_individuals.append(copy.deepcopy(pop[int(idx)]) if pop[int(idx)] in new_individuals else pop[int(idx)])
            new_individuals.append(copy.deepcopy(pop[int(idx)]))

        pop.individuals = new_individuals

        return pop


def check_dominance(a, b, cva=None, cvb=None):
    if cva is not None and cvb is not None:
        if cva < cvb:
            return 1
        elif cvb < cva:
            return -1
    val = 0
    for i in range(len(a)):
        if a[i] < b[i]:
            # indifferent because once better and once worse
            if val == -1:
                return 0
            val = 1
        elif b[i] < a[i]:
            # indifferent because once better and once worse
            if val == 1:
                return 0
            val = -1
    return val


def compare(a, a_val, b, b_val, method):
    if method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            return np.random.choice([a, b])

    elif method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            return np.random.choice([a, b])

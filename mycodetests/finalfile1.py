import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.EliminationOrder import WeightedMinFill, MinWeight, MinNeighbors, MinFill
model_1 = BayesianModel([('c', 'd'), ('d', 'g'), ('i', 'g'), ('i', 's'), ('s', 'j'), ('g', 'l'), ('l', 'j'), ('j', 'h'), ('g', 'h')])
model_2 = BayesianModel([('c', 'd'), ('d', 'g'), ('i', 'g'), ('i', 's'), ('s', 'j'), ('g', 'l'), ('l', 'j'), ('j', 'h'), ('g', 'h')])
model_3 = BayesianModel([('c', 'd'), ('d', 'g'), ('i', 'g'), ('i', 's'), ('s', 'j'), ('g', 'l'), ('l', 'j'), ('j', 'h'), ('g', 'h')])
cpd_c = TabularCPD('c', 2, np.random.rand(2, 1))
# print(cpd_c)
cpd_d = TabularCPD('d', 2, np.random.rand(2, 2), ['c'], [2])
# print(cpd_d)
cpd_g = TabularCPD('g', 3, np.random.rand(3, 4), ['d', 'i'], [2, 2])
# print(cpd_g)
cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
# print(cpd_i)
cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], [2])
# print(cpd_s)
cpd_j = TabularCPD('j', 2, np.random.rand(2, 4), ['l', 's'], [2, 2])
# print(cpd_j)
cpd_l = TabularCPD('l', 2, np.random.rand(2, 3), ['g'], [3])
# print(cpd_l)
cpd_h = TabularCPD('h', 2, np.random.rand(2, 6), ['g', 'j'], [3, 2])
# print(cpd_h)
model_1.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
model_2.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
model_3.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)

def run_model_with_heuristic(heuristic, model):
    return heuristic(model).get_elimination_order()

print("with model 1:")
print("with WeightedMinFill:")
print(run_model_with_heuristic(WeightedMinFill, model_1))
print("with MinFill:")
print(run_model_with_heuristic(MinFill, model_1))
print("with MinWeight:")
print(run_model_with_heuristic(MinWeight, model_1))
print("with MinNeighbors:")
print(run_model_with_heuristic(MinNeighbors, model_1))
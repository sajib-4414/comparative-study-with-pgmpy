import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.EliminationOrder import WeightedMinFill, MinWeight, MinNeighbors, MinFill
model = BayesianModel([('c', 'd'), ('d', 'g'), ('i', 'g'), ('i', 's'), ('s', 'j'), ('g', 'l'), ('l', 'j'), ('j', 'h'), ('g', 'h')])
cpd_c = TabularCPD('c', 2, np.random.rand(2, 1))
cpd_d = TabularCPD('d', 2, np.random.rand(2, 2), ['c'], [2])
cpd_g = TabularCPD('g', 3, np.random.rand(3, 4), ['d', 'i'], [2, 2])
cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], [2])
cpd_j = TabularCPD('j', 2, np.random.rand(2, 4), ['l', 's'], [2, 2])
cpd_l = TabularCPD('l', 2, np.random.rand(2, 3), ['g'], [3])
cpd_h = TabularCPD('h', 2, np.random.rand(2, 6), ['g', 'j'], [3, 2])

model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)

def run_model_with_heuristic(heuristic, model):
    return heuristic(model).get_elimination_order()

print("with WeightedMinFill:", run_model_with_heuristic(WeightedMinFill, model))
print("with MinFill:", run_model_with_heuristic(MinFill, model))
print("with MinWeight:", run_model_with_heuristic(MinWeight, model))
print("with MinNeighbors:", run_model_with_heuristic(MinNeighbors, model))
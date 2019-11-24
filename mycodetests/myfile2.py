import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.EliminationOrder import WeightedMinFill, MinWeight, MinNeighbors, MinFill
model = BayesianModel([('c', 'd'), ('d', 'g'), ('i', 'g'), ('i', 's'), ('s', 'j'), ('g', 'l'), ('l', 'j'), ('j', 'h'), ('g', 'h')])
cpd_c = TabularCPD('c', 2, np.random.rand(2, 1))
print(cpd_c)
cpd_d = TabularCPD('d', 2, np.random.rand(2, 2), ['c'], [2])
print(cpd_d)
cpd_g = TabularCPD('g', 3, np.random.rand(3, 4), ['d', 'i'], [2, 2])
print(cpd_g)
cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
print(cpd_i)
cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], [2])
print(cpd_s)
cpd_j = TabularCPD('j', 2, np.random.rand(2, 4), ['l', 's'], [2, 2])
print(cpd_j)
cpd_l = TabularCPD('l', 2, np.random.rand(2, 3), ['g'], [3])
print(cpd_l)
cpd_h = TabularCPD('h', 2, np.random.rand(2, 6), ['g', 'j'], [3, 2])
print(cpd_h)
model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
# v3 = MinNeighbors(model).get_elimination_order([])
# print(v3)
# subarry = ['c', 'd', 'g', 'l', 's','i','s','h']
# for x in subarry:
#     print(x + ":" + str(MinNeighbors(model).cost(x)) + ",")
# # ['c', 's', 'l', 'd', 'g']
# v2 = MinFill(model).get_elimination_order(['c', 'd', 'g', 'l', 's'])
# print(v2)
# # ['c', 's', 'l', 'd', 'g']
# v = MinWeight(model).get_elimination_order(['c', 'd', 'g', 'l', 's'])
# # ['c', 's', 'l', 'd', 'g']
# print(v)
# v4 = WeightedMinFill(model).get_elimination_order(['c'])
# print(WeightedMinFill(model).cost('d'))


# ['c', 's', 'l', 'd', 'g']
# print(v4)
subarry = ['c', 'd', 'g', 'l', 's']
# outputorder = MinWeight(model).get_elimination_order(subarry)
# print("output order is " , outputorder)
# print("printing cost of each elements to figure out why that was chosen\n")
# for x in subarry:
#     print(x + ":" + str(MinWeight(model).cost(x)) + ",")
# for x in subarry:
#     print("cost of ",x, " is ", MinWeight(model).cost(x))
#     model
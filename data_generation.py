import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#Creates the valuation Matrix
def instance(num_agents, num_houses, max_value=10):
    M = np.zeros((num_agents, num_houses), dtype=int)
    for agent in range(num_agents):
        for house in range(num_houses):
            M[agent, house] = np.random.randint(0, max_value + 1)
    print(M)
    return M




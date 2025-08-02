import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

def generate_single_dipped_preferences(n, m, value_min=0, value_max=100):
    houses = list(range(m))
    dips = [random.choice(houses) for _ in range(n)]
    preferences = np.zeros((n, m), dtype=int)
    for i in range(n):
        dip = dips[i]
        distances = [abs(h - dip) for h in houses]
        max_dist = max(distances)
        values = [int(round(value_min + (value_max - value_min) * (dist / max_dist))) for dist in distances]
        preferences[i, :] = values
    return preferences
        
num_agents = 10
houses_list = [10, 11, 12, 13, 14, 15, 16, 18, 20, 25]
num_instances = 1000


def max_weighted_matching(M):
    num_agents, num_houses = M.shape
    G = nx.Graph()
    for agent in range(num_agents):
        for house in range(num_houses):
            G.add_edge(agent, num_agents + house, weight=M[agent][house])
    matching = nx.max_weight_matching(G, maxcardinality=True)
    allocation = []
    for a, h in matching:
        if a < num_agents:
            agent, house = a, h - num_agents
        else:
            agent, house = h, a - num_agents
        allocation.append((agent, house))
    allocation.sort()
    return allocation

def calculate_total_welfare(M, allocation):
    return sum(M[agent, house] for agent, house in allocation)

def min_envy_single_dipped_allocation(M):
    n_agents, n_houses = M.shape
    ranks = np.argsort(-M, axis=1)
    S1 = set(ranks[:,0])
    allocation = [None] * n_agents
    allocated_houses = set()
    # Case 1: >1 distinct best houses
    if len(S1) > 1:
        assigned = 0
        used_houses = set()
        for h in S1:
            for i in range(n_agents):
                if ranks[i,0] == h and h not in used_houses:
                    allocation[i] = (i, h)
                    allocated_houses.add(h)
                    used_houses.add(h)
                    assigned += 1
                    break
            if assigned == 2:
                break
        for i in range(n_agents):
            if allocation[i] is None:
                for h in np.argsort(-M[i]):
                    if h not in allocated_houses:
                        allocation[i] = (i, h)
                        allocated_houses.add(h)
                        break
    else:
        h = list(S1)[0]
        peak_val = np.max(M[:, h])
        span = set()
        for idx in range(n_houses):
            if all(M[i, idx] == peak_val for i in range(n_agents)):
                span.add(idx)
        if n_houses - len(span) >= n_agents:
            vals = [M[i, idx] for i in range(n_agents) for idx in range(n_houses) if M[i, idx] != peak_val]
            next_best_val = sorted(set(vals), reverse=True)[0] if vals else peak_val
            Sspan_plus_1 = set()
            for idx in range(n_houses):
                if idx not in span and any(M[i, idx] == next_best_val for i in range(n_agents)):
                    Sspan_plus_1.add(idx)
            Sspan_plus_1 = list(Sspan_plus_1)
            h1 = Sspan_plus_1[0] if len(Sspan_plus_1) > 0 else None
            h2 = Sspan_plus_1[1] if len(Sspan_plus_1) > 1 else None
            assigned = 0
            for hid in [h1, h2]:
                if hid is not None:
                    for i in range(n_agents):
                        if allocation[i] is None and M[i, hid] == next_best_val:
                            allocation[i] = (i, hid)
                            allocated_houses.add(hid)
                            assigned += 1
                            break
                if assigned == 2:
                    break
            unavailable = set(span)
            for i in range(n_agents):
                if allocation[i] is None:
                    for hidx in np.argsort(-M[i]):
                        if hidx not in allocated_houses and hidx not in unavailable:
                            allocation[i] = (i, hidx)
                            allocated_houses.add(hidx)
                            break
        else:
            for i in range(n_agents):
                if ranks[i,0]==h:
                    allocation[i]=(i,h)
                    allocated_houses.add(h)
                    break
            for i in range(n_agents):
                if allocation[i] is None:
                    for hidx in np.argsort(-M[i]):
                        if hidx not in allocated_houses:
                            allocation[i] = (i, hidx)
                            allocated_houses.add(hidx)
                            break
    return allocation

# Main experiment
avg_welfare_max, avg_welfare_fair = [], []
ci_welfare_max, ci_welfare_fair = [], []


with open("all_single_dipped_preferences.csv", "w") as f:
    for num_houses in houses_list:
        welfare_max_values = []
        welfare_fair_values = []
        for _ in range(num_instances):
            M = generate_single_dipped_preferences(num_agents, num_houses)
            f.write(f"# n={num_agents}, m={num_houses}\n")
            np.savetxt(f, M, delimiter=",",fmt='%d')
            f.write("\n")
            alloc_max = max_weighted_matching(M)
            welfare_max = calculate_total_welfare(M, alloc_max)
            alloc_fair = min_envy_single_dipped_allocation(M)
            welfare_fair = calculate_total_welfare(M, alloc_fair)
            welfare_max_values.append(welfare_max)
            welfare_fair_values.append(welfare_fair)
        avg_welfare_max.append(np.mean(welfare_max_values))
        avg_welfare_fair.append(np.mean(welfare_fair_values))
        # 95% confidence interval using standard error of mean
        # ci_welfare_max.append(1.96 * sem(welfare_max_values))
        # ci_welfare_fair.append(1.96 * sem(welfare_fair_values))

# Plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(houses_list, avg_welfare_max, label='Maximum Welfare', marker='o', capsize=4)
plt.errorbar(houses_list, avg_welfare_fair, label='Welfare in the Allocation Minimizing #Envy', marker='x', capsize=4)
plt.xlabel('Number of Houses', fontsize=14)
plt.ylabel('Average Total Welfare', fontsize=14)
plt.title('Welfare Loss in the Min #Envy Allocation for Single-Dipped Preferences\n(With 95% Confidence Intervals)', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


# generate_single_dipped_preferences(n, m_list)

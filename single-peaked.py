import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


num_agents = 10
houses_list = [10 , 11 ,12 ,13 ,14, 15, 16, 18, 20, 25]
num_instances = 1000

def single_peaked_valuations(num_agents, num_houses, max_value=100):
    M = np.zeros((num_agents, num_houses), dtype=int)
    for agent in range(num_agents):
        peak = np.random.randint(num_houses)
        for h in range(num_houses):
            M[agent, h] = max_value - abs(h - peak) * (max_value // num_houses)
    return M

# num_agents = 10
# num_houses = 15
# M = single_peaked_valuations(num_agents, num_houses)

# plt.figure(figsize=(10, 6))
# for agent in range(num_agents):
#     plt.plot(range(num_houses), M[agent], marker='o', label=f'Agent {agent+1}')
# plt.title('Single-Peaked Valuations')
# plt.xlabel('House Index')
# plt.ylabel('Valuation')
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
# plt.grid(True)
# plt.tight_layout()
# plt.show()


def max_weighted_matching(M):
    G = nx.Graph()
    for agent in range(M.shape[0]):
        for house in range(M.shape[1]):
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
    total_welfare = 0
    for pair in allocation:
        # Accept both (agent_idx, house_idx) or ('A0', 'H3')
        if isinstance(pair[0], str):
            agent_idx = int(pair[0][1:])
            house_idx = int(pair[1][1:])
        else:
            agent_idx, house_idx = pair
        total_welfare += M[agent_idx, house_idx]
    return total_welfare

def compute_spans(peaks, base_dict, num_houses):
    spans = {}
    for h, agents in base_dict.items():
        if len(agents) > 1:
            left = right = h
            while left - 1 >= 0 and all(abs(peaks[a] - (left - 1)) < abs(peaks[a] - h) for a in agents):
                left -= 1
            while right + 1 < num_houses and all(abs(peaks[a] - (right + 1)) < abs(peaks[a] - h) for a in agents):
                right += 1
            spans[h] = set(range(left, right + 1))
        else:
            spans[h] = set()
    return spans

def min_envy_single_peaked_allocation(M):
    n_agents, n_houses = M.shape
    peaks = [np.argmax(M[i]) for i in range(n_agents)]
    base = {h: [] for h in range(n_houses)}
    for agent, peak in enumerate(peaks):
        base[peak].append(agent)
    spans = compute_spans(peaks, base, n_houses)
    allocation = [None] * n_agents
    allocated_houses = set()
    allocated_agents = set()
    for h, agents in base.items():
        if len(agents) == 1:
            agent = agents[0]
            allocation[agent] = (agent, h)
            allocated_houses.add(h)
            allocated_agents.add(agent)
    shared_peaks = [h for h, agents in base.items() if len(agents) > 1]
    unresolved_peaks = set(shared_peaks)
    remaining_agents = set(range(n_agents)) - allocated_agents
    remaining_houses = set(range(n_houses)) - allocated_houses
    while unresolved_peaks:
        min_span_peak = min(unresolved_peaks, key=lambda h: len(spans[h]))
        min_span = spans[min_span_peak]
        base_agents = [a for a in base[min_span_peak] if a in remaining_agents]
        if base_agents:
            agent = base_agents[0]
            allocation[agent] = (agent, min_span_peak)
            allocated_houses.add(min_span_peak)
            allocated_agents.add(agent)
            remaining_agents.remove(agent)
            remaining_houses.remove(min_span_peak)
        for h in min_span:
            if h in remaining_houses:
                remaining_houses.remove(h)
        unresolved_peaks.remove(min_span_peak)
    for agent in remaining_agents:
        available_houses = list(remaining_houses)
        best_house = max(available_houses, key=lambda h: M[agent, h])
        allocation[agent] = (agent, best_house)
        remaining_houses.remove(best_house)
    return allocation

# Main experiment
avg_welfare_max = []
avg_welfare_fair = []


with open("all_single_peaked_preferences.csv", "w") as f:
    for num_houses in houses_list:
        welfare_max_values = []
        welfare_fair_values = []
        for _ in range(num_instances):
            M = single_peaked_valuations(num_agents, num_houses)
            M = single_peaked_valuations(num_agents, num_houses)
            f.write(f"# n={num_agents}, m={num_houses}\n")
            np.savetxt(f, M, delimiter=",",fmt='%d')
            f.write("\n")
            alloc_max = max_weighted_matching(M)
            welfare_max = calculate_total_welfare(M, alloc_max)
            alloc_fair = min_envy_single_peaked_allocation(M)
            welfare_fair = calculate_total_welfare(M, alloc_fair)
            welfare_max_values.append(welfare_max)
            welfare_fair_values.append(welfare_fair)
        avg_welfare_max.append(np.mean(welfare_max_values))
        avg_welfare_fair.append(np.mean(welfare_fair_values))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(houses_list, avg_welfare_max, label='Maximum Welfare', marker='o')
plt.plot(houses_list, avg_welfare_fair, label='Welfare in the Allocation Minimizing #Envy', marker='x')
plt.xlabel('Number of Houses',fontsize=14)
plt.ylabel('Average Total Welfare',fontsize=14)
plt.title('Welfare Loss in the Min #Envy Allocation for Single Peaked Preferences',fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

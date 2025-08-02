import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

#Creates the valuation Matrix
# def instance(num_agents, num_houses, max_value=10):
#     M = np.zeros((num_agents, num_houses), dtype=int)
#     for agent in range(num_agents):
#         for house in range(num_houses):
#             M[agent, house] = np.random.randint(0, max_value + 1)
#     print(M)
#     return M

def instance(filename, instance_number):
    with open(filename, 'r') as f:
        lines = f.readlines()
    current_matrix = []
    instances_found = -1
    for line in lines:
        line = line.strip()
        if line.startswith('# Instance'):
            instances_found += 1
            if instances_found > instance_number:
                break  # We read the target instance already
            current_matrix = []
        elif line and not line.startswith('#') and instances_found == instance_number:
            # Parse the matrix line
            row = [int(x) for x in line.split(',') if x]
            current_matrix.append(row)

    if not current_matrix:
        raise ValueError(f"Instance number {instance_number} not found in file.")

    M = np.array(current_matrix, dtype=int)
    print(M)
    return M


def egalitarian_welfare(M, allocation):
    utility = np.zeros(len(M))
    for agent, house in allocation:
        utility[agent] = M[agent][house]
    min_util = np.min(utility)
    return min_util

def max_nsw_via_log_matching(M):
    n_agents, n_houses = M.shape
    G = nx.Graph()
    
    for agent in range(n_agents):
        for house in range(n_houses):
            if M[agent][house] > 0:
                weight = np.log(M[agent][house])
                G.add_edge(agent, n_agents + house, weight=weight)

    matching = nx.max_weight_matching(G, maxcardinality=True)
    
    allocation = []
    for a, h in matching:
        if a < n_agents:
            agent, house = a, h - n_agents
        else:
            agent, house = h, a - n_agents
        allocation.append((agent, house))
    
    allocation.sort()
    print("This is NSW maximizing allocation (via log matching):", allocation)
    return allocation


def nash_social_welfare(M, allocation):
    utility = np.zeros(len(M))
    for agent, house in allocation:
        utility[agent] = M[agent][house]
    if np.any(utility == 0):
        return 0  # NSW is zero if any agent has zero utility
    product = np.prod(utility)
    nth_root = product ** (1 / len(allocation))  # nth root of the product
    return nth_root

def max_egalitarian_allocation(M):
    n_agents, n_houses = M.shape
    houses_idx = list(range(n_houses))
    max_min_utility = -1
    best_alloc = None
    for perm in permutations(houses_idx, n_agents):
        alloc = [(agent, house) for agent, house in enumerate(perm)]
        min_util = egalitarian_welfare(M, alloc)
        if min_util > max_min_utility:
            max_min_utility = min_util
            best_alloc = alloc
    print("This is egalitarian-maximizing allocation:", best_alloc)
    return best_alloc


def calculate_num_envy(allocation_dict, valuations):
    if isinstance(allocation_dict, list):
        allocation_dict = {f'i{a}': f'h{h}' for (a, h) in allocation_dict}
    else:
        allocation_dict = allocation_dict

    allocation = [(int(a[1:]), int(h[1:])) for a, h in allocation_dict.items()]
    envy_count = 0
    for a in range(valuations.shape[0]):
        agent_houses = [h for (agent, h) in allocation if agent == a]
        if not agent_houses:
            continue
        current_house = agent_houses[0]
        current_value = valuations[a, current_house]
        
        for (other_agent, other_house) in allocation:
            if valuations[a, other_house] > current_value:
                envy_count += 1
                break
    return envy_count


#calculates what we call A \oplus P in the paper
def compute_allocation_oplus_paths(A_dict, P):
    def is_house(node): return node.startswith('h')
    def is_agent(node): return node.startswith('i')
    edges = []
    for i in range(len(P)-1):
        u, v = P[i], P[i+1]
        if is_house(u) and is_agent(v):
            edges.append((v, u)) 
        elif is_agent(u) and is_house(v):
            edges.append((u, v))
    new_A = A_dict.copy()
    for agent, house in edges:
        if new_A.get(agent) == house:
            del new_A[agent]
        else:
            if agent in new_A:
                del new_A[agent]
            new_A[agent] = house
    return new_A


# Finds alternating components in the symmetric difference of two allocations A and A_hat.
def find_alternating_components(A, A_hat, valuations):
    A_edges = {(f'i{a}', f'h{h}') for (a, h) in A}
    A_hat_edges = {(f'i{a}', f'h{h}') for (a, h) in A_hat}
    sym_diff = A_edges.symmetric_difference(A_hat_edges)
    # Induce subgraph on symmetric difference edges
    H = G.edge_subgraph(sym_diff)
    components = []
    for component in nx.connected_components(H):
        subgraph = H.subgraph(component)
        try:
            cycle = nx.find_cycle(subgraph)
            nodes = [cycle[0][0], cycle[0][1]]
            for edge in cycle[1:]:
                nodes.append(edge[1])
            components.append(nodes)
        except nx.NetworkXNoCycle:
            endpoints = [n for n, d in subgraph.degree() if d == 1]
            if len(endpoints) == 2:
                path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
                components.append(path)
    return components

#Based on the edges in the symmetric difference, computes a good coloring of the graph G.
def good_coloring(G, A, A_hat, n_agents, m_houses):
    A_edges = {(a, n_agents + h) for a, h in A}
    A_hat_edges = {(a, n_agents + h) for a, h in A_hat}
    T = A_edges.symmetric_difference(A_hat_edges)
    S = set()
    for u, v in T:
        S.add(u)
        S.add(v)
    vertex_colors = {node: 'blue' for node in G.nodes()}  # Default to blue
    edge_colors = {}
    for node in S:
        vertex_colors[node] = 'red'
    for u, v in G.edges():
        edge = (u, v)
        if edge in T or (v, u) in T:
            edge_colors[edge] = 'red'
        elif u in S and v in S:
            edge_colors[edge] = 'green'
        elif (u in S) ^ (v in S):  # XOR: exactly one endpoint in S
            edge_colors[edge] = 'blue'
        else:
            edge_colors[edge] = 'gray'  # Default for edges not involving S
    return vertex_colors, edge_colors


#This just applies all paths in the component. 
# Change this to seleceted paths which only realloacte at most q houses and witnesses the maximum possible reduction in the envy.
def apply_all_components(initial_allocation, paths):
    A_dict = {f'i{a}': f'h{h}' for (a, h) in initial_allocation}
    for path in paths:
        A_dict = compute_allocation_oplus_paths(A_dict, path)
    return [(int(a[1:]), int(h[1:])) for a, h in A_dict.items()]

from itertools import combinations

def apply_q_components_brute_force(max_welfare_allocation, reordered_labeled_paths, valuations, q):
    original_envy = calculate_num_envy(max_welfare_allocation, valuations)
    
    # Precompute agent counts for each path
    path_agent_counts = []
    for path in reordered_labeled_paths:
        agents_affected = len({node for node in path if node.startswith('i')})
        path_agent_counts.append(agents_affected)
    
    best_combination = []
    max_envy_reduction = -1
    best_allocation = max_welfare_allocation
    
    # Try all possible combinations of components
    n_paths = len(reordered_labeled_paths)
    
    for r in range(1, n_paths + 1):  # r is the size of combination
        for combination in combinations(range(n_paths), r):
            # Check if this combination respects the q constraint
            total_agents = sum(path_agent_counts[i] for i in combination)
            if total_agents > q:
                continue
            
            # Apply the combination and calculate actual envy reduction
            selected_paths = [reordered_labeled_paths[i] for i in combination]
            temp_allocation = apply_all_components(max_welfare_allocation, selected_paths)
            new_envy = calculate_num_envy(temp_allocation, valuations)
            envy_reduction = original_envy - new_envy
            
            # Update best if this combination is better
            if envy_reduction > max_envy_reduction:
                max_envy_reduction = envy_reduction
                best_combination = combination
                best_allocation = temp_allocation
    
    return best_allocation




def apply_q_components(max_welfare_allocation, reordered_labeled_paths, valuations, q):
    # Calculate original envy for reference
    original_envy = calculate_num_envy(max_welfare_allocation, valuations)
    # Precompute properties for each path
    items = []
    for path in reordered_labeled_paths:
        # Calculate number of affected agents
        agents_affected = len({node for node in path if node.startswith('i')})
        
        # Calculate envy reduction by applying this path
        temp_alloc = apply_all_components(max_welfare_allocation, [path])
        new_envy = calculate_num_envy(temp_alloc, valuations)
        envy_reduction = original_envy - new_envy
        
        items.append((envy_reduction, agents_affected, path))

    # Initialize DP table and path tracking
    dp = [-float('inf')] * (q + 1)
    dp[0] = 0
    path_selections = [[] for _ in range(q + 1)]

    # Populate DP table
    for er, na, path in items:
        for w in range(q, na - 1, -1):
            if dp[w - na] + er > dp[w]:
                dp[w] = dp[w - na] + er
                path_selections[w] = path_selections[w - na] + [path]

    # Find optimal solution
    max_reduction = max(dp)
    optimal_weight = max(i for i, val in enumerate(dp) if val == max_reduction)
    
    # Apply selected paths
    if optimal_weight > 0:
        return apply_all_components(max_welfare_allocation, path_selections[optimal_weight])
    
    return max_welfare_allocation

    #Use knapsack and choose which components from the reordered labeled paths should be applied to the max_welfare_allocation in order to get
    #  maximum reduction in the envy while changing the allocation of at most q agents.
    # return max_welfare_allocation  # Placeholder for the actual implementation


#From the good coloring, computes the feasible components based on the conditions specified in the problem.
def find_feasible_components(G, vertex_colors, edge_colors, A_hat, valuations):
    red_edges = [e for e, color in edge_colors.items() if color == 'red']
    G_red = G.edge_subgraph(red_edges)
    components = list(nx.connected_components(G_red))
    paths_in_red = []
    for component in components:
        induced_subgraphs = G.subgraph(component) 
        paths_in_red.append(list(induced_subgraphs.edges))
    feasible = []
    feasible_paths =[]
    for path in paths_in_red:  # Changed iteration target
        # Extract nodes from path edges
        nodes_in_path = {u for edge in path for u in edge}
        # Condition 1: All vertices red
        if any(vertex_colors[node] != 'red' for node in nodes_in_path):
            continue
        #Condition 2: 
        # Condition 3: No internal blue edges
        if has_internal_blue_edges(nodes_in_path, edge_colors):
            continue 
        # Condition 4: Niceness check
        if not is_nice_component(nodes_in_path, components, A_hat, valuations):
            continue
        feasible.append(nodes_in_path)
        feasible_paths.append(path)
    return feasible, feasible_paths

def convert_feasible_paths(feasible_paths, n_agents):
    labeled_paths = []
    for path in feasible_paths:
        # Extract unique nodes from edges
        nodes = {u for edge in path for u in edge}
        
        # Build node sequence (simple linear path for demonstration)
        node_sequence = []
        for edge in path:
            u, v = edge
            if u not in node_sequence:
                node_sequence.append(u)
            if v not in node_sequence:
                node_sequence.append(v)
        
        # Convert to labels
        labeled_nodes = []
        for node in node_sequence:
            if node < n_agents:
                labeled_nodes.append(f'i{node}')
            else:
                labeled_nodes.append(f'h{node - n_agents}')
        labeled_paths.append(labeled_nodes)
    return labeled_paths


def reorder_labeled_paths(labeled_paths, A_hat):
    allocated_houses = {f'h{h}' for (_, h) in A_hat}
    a_hat_mapping = {f'i{a}': f'h{h}' for (a, h) in A_hat}
    reordered = []
    for path in labeled_paths:
        # Find first unallocated house in path
        start_house = next((node for node in path if node.startswith('h') and node not in allocated_houses), None)
        if not start_house:
            continue  # Skip paths without unallocated houses
        new_path = []
        visited = set()
        current_node = start_house
        while len(new_path) < len(path):
            new_path.append(current_node)
            visited.add(current_node)
            # Alternate between agent and house
            if current_node.startswith('h'):
                # Find next agent in original path order
                agent = next((node for node in path if node.startswith('i') and node not in visited), None)
                if not agent:
                    break
                current_node = agent
            else:
                # Find allocated house for current agent
                current_node = a_hat_mapping.get(current_node, None)
                if not current_node or current_node not in path or current_node in visited:
                    break
        if len(new_path) == len(path):
            reordered.append(new_path)
    print("These are reordered labelled paths:", reordered)
    return reordered

def has_internal_blue_edges(component, edge_colors):
    for u in component:
        for v in component:
            if (u, v) in edge_colors and edge_colors[(u, v)] == 'blue':
                return True
    return False

def is_nice_component(current_comp, all_comps, A_hat, valuations):
    # for other_comp in all_comps:
    #     if other_comp == current_comp:
    #         continue
    #     # Check if combined effect equals sum of individual effects
    #     combined_envy_reduction = calculate_envy_reduction([current_comp, other_comp], optimal_allocation, valuations)
    #     individual_sum = (calculate_envy_reduction([current_comp], optimal_allocation, valuations) 
    #                     + calculate_envy_reduction([other_comp], optimal_allocation, valuations))
        
    #     if combined_envy_reduction != individual_sum:
    #         return False
            
    return True
    # Simplified check - full implementation requires comparing all combinations

def calculate_envy_reduction(path, A_hat, valuations):
    temp_alloc = apply_all_components(A_hat, path)
    agent_houses = {a: h for a, h in A_hat}
    original_envy = calculate_num_envy(A_hat, valuations)
    new_envy = calculate_num_envy(temp_alloc, valuations)
    return original_envy - new_envy

n_agents = 6
m_houses = 11
num_instances = 100
qs = list(range(0, 7))

avg_welfare_max = []
avg_welfare_fair = []
avg_welfare_fair_q = []
ci_max_upper = []
ci_max_lower = []
ci_fair_upper = []
ci_fair_lower = []
ci_fair_q_upper = []
ci_fair_q_lower = []

welfare_max_values = []
welfare_fair_values = []
welfare_fair_q_values = []

welfare_max_all = []
welfare_fair_q = [[] for _ in qs]

  # each sublist stores welfare for that q across instances

avg_welfare_max_egal = []
avg_welfare_fair_egal = []
avg_welfare_fair_q_egal = []
ci_max_upper_egal = []
ci_max_lower_egal = []
ci_fair_upper_egal = []
ci_fair_lower_egal = []
ci_fair_q_upper_egal = []
ci_fair_q_lower_egal = []

welfare_max_values_egal = []
welfare_fair_values_egal = []
welfare_fair_q_values_egal = []

welfare_max_all_egal = []
welfare_fair_q_egal = [[] for _ in qs] 

# filename = 'HouseAllocationInstances.csv'
# open(filename, 'w').close()

# for _ in range(num_instances):
    # valuations = instance(n_agents, m_houses)
    # with open(filename, 'a') as f:
    #     f.write(f"# Instance {_}\n")
    #     np.savetxt(f, valuations, delimiter=',', fmt='%d')
    #     f.write("\n") 


filename = 'HouseAllocationInstances.csv'

for i in range(num_instances):
    valuations = instance(filename, i)
    # Step 1: Optimal fair allocation (min-envy + max welfare among them)
    min_envy = float('inf')
    best_allocs = []
    houses_idx = list(range(m_houses))
    for perm in permutations(houses_idx, n_agents):
        alloc = [(agent, house) for agent, house in enumerate(perm)]
        envy = calculate_num_envy(alloc, valuations)
        if envy < min_envy:
            min_envy = envy
            best_allocs = [alloc]
        elif envy == min_envy:
            best_allocs.append(alloc)

    optimal_fair_alloc = max(best_allocs, key=lambda alloc: nash_social_welfare(valuations, alloc))
    optimal_fair_alloc_egal = max(best_allocs, key=lambda alloc: egalitarian_welfare(valuations, alloc))

    max_welfare_alloc = max_nsw_via_log_matching(valuations)
    max_w = nash_social_welfare(valuations, max_welfare_alloc)
    welfare_max_all.append(max_w)

    max_welfare_alloc_egal = max_egalitarian_allocation(valuations)
    max_w_egal = egalitarian_welfare(valuations, max_welfare_alloc_egal)
    welfare_max_all_egal.append(max_w_egal)

    # Step 3: Reuse component structure to apply q changes
    G = nx.Graph()
    G.add_nodes_from(range(n_agents + m_houses))
    for a in range(n_agents):
        for h in range(m_houses):
            G.add_edge(a, n_agents + h)

    vertex_colors, edge_colors = good_coloring(G, optimal_fair_alloc, max_welfare_alloc, n_agents, m_houses)
    feasible_comps, feasible_paths = find_feasible_components(G, vertex_colors, edge_colors, max_welfare_alloc, valuations)
    labeled_paths = convert_feasible_paths(feasible_paths, n_agents)
    reordered_labeled_paths = reorder_labeled_paths(labeled_paths, optimal_fair_alloc)

    # Step 4: For each q, apply up to q improvements and record welfare
    for idx, q in enumerate(qs):
        fair_alloc_q = apply_q_components(max_welfare_alloc, reordered_labeled_paths, valuations, q)
        fair_w = nash_social_welfare(valuations, fair_alloc_q)
        welfare_fair_q[idx].append(fair_w)


# egalitarian:
    vertex_colors_2, edge_colors_2 = good_coloring(G, optimal_fair_alloc_egal, max_welfare_alloc_egal, n_agents, m_houses)
    feasible_comps_2, feasible_paths_2 = find_feasible_components(G, vertex_colors_2, edge_colors_2, max_welfare_alloc_egal, valuations)
    labeled_paths_2 = convert_feasible_paths(feasible_paths_2, n_agents)
    reordered_labeled_paths_2 = reorder_labeled_paths(labeled_paths_2, optimal_fair_alloc_egal)

    # Step 4: For each q, apply up to q improvements and record welfare
    for idx, q in enumerate(qs):
        fair_alloc_q_egal = apply_q_components(max_welfare_alloc_egal, reordered_labeled_paths_2, valuations, q)
        fair_w_egal = egalitarian_welfare(valuations, fair_alloc_q_egal)
        welfare_fair_q_egal[idx].append(fair_w_egal)


mean_max = np.mean(welfare_max_all) 
std_max = np.std(welfare_max_all, ddof=1) 
ci_max = 1.96 * std_max / np.sqrt(num_instances)
ci_max_lower = [mean_max - ci_max] * len(qs)
ci_max_upper = [mean_max + ci_max] * len(qs)
mean_fair = [np.mean(wlist) for wlist in welfare_fair_q]
ci_lower = [np.mean(w) - 1.96 * np.std(w, ddof=1) / np.sqrt(num_instances) for w in welfare_fair_q]
ci_upper = [np.mean(w) + 1.96 * np.std(w, ddof=1) / np.sqrt(num_instances) for w in welfare_fair_q]



mean_max_egal = np.mean(welfare_max_all_egal)
std_max_egal = np.std(welfare_max_all_egal, ddof=1) 
ci_max_egal = 1.96 * std_max_egal / np.sqrt(num_instances)
ci_max_lower_egal = [mean_max_egal - ci_max_egal] * len(qs)
ci_max_upper_egal = [mean_max_egal + ci_max_egal] * len(qs)
mean_fair_egal = [np.mean(wlist) for wlist in welfare_fair_q_egal]
ci_lower_egal = [np.mean(w) - 1.96 * np.std(w, ddof=1) / np.sqrt(num_instances) for w in welfare_fair_q_egal]
ci_upper_egal = [np.mean(w) + 1.96 * np.std(w, ddof=1) / np.sqrt(num_instances) for w in welfare_fair_q_egal]

plt.figure(figsize=(10, 6))

# Max-welfare line + CI band
plt.plot(qs, [mean_max] * len(qs), label='Nash Welfare', color='blue',linestyle='-', marker='o')
plt.fill_between(qs,ci_max_lower, ci_max_upper, color='blue', alpha=0.2)

# Fair-with-q-changes line + CI band
plt.plot(qs, mean_fair, label='Nash Welfare after q Reallocations', color='blue',linestyle='--', marker='x')
plt.fill_between(qs,ci_lower, ci_upper, color='blue', alpha=0.2)

# Max-welfare line + CI band
plt.plot(qs, [mean_max_egal] * len(qs), label='Egalitarian Welfare', color='green',linestyle='-', marker='o')
plt.fill_between(qs,ci_max_lower_egal, ci_max_upper_egal, color='green', alpha=0.2)

# Fair-with-q-changes line + CI band
plt.plot(qs, mean_fair_egal, label='Egalitarian Welfare after q Reallocations', color='green',linestyle='--',marker='x')
plt.fill_between(qs,ci_lower_egal, ci_upper_egal, color='green', alpha=0.2)


plt.xlabel('Number of Reallocations (q)',fontsize=16)
plt.ylabel('Welfare',fontsize=16)
plt.title('Welfare Loss After q Reallocations',fontsize=16)
plt.legend()
# plt.yticks(np.arange(7.6, 9.6, 0.1))
plt.grid(True)
plt.tight_layout()
plt.show() 






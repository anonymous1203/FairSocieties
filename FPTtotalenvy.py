import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

#Creates the valuation Matrix
def instance(num_agents, num_houses, max_value=10):
    M = np.zeros((num_agents, num_houses), dtype=int)
    for agent in range(num_agents):
        for house in range(num_houses):
            M[agent, house] = np.random.randint(0, max_value + 1)
        # k = np.random.randint(4, num_houses)  # Between 0 and 5 inclusive
        # if k > 0:
        #     liked_houses = np.random.choice(num_houses, size=k, replace=False)
        #     M[agent, liked_houses] = np.random.randint(1, max_value + 1, size=k)
    print(M)
    return M


#Finds the maximum weighted matching in a bipartite graph
def max_weighted_matching(M):
    G = nx.Graph()
    for agent in range(M.shape[0]):
        for house in range(M.shape[1]):
            G.add_edge(agent, n_agents + house, weight=M[agent][house])
    matching = nx.max_weight_matching(G, maxcardinality=True)
    allocation = []
    for a, h in matching:
        if a < n_agents:
            agent, house = a, h - n_agents
        else:
            agent, house = h, a - n_agents
        allocation.append((agent, house))
    allocation.sort()
    print("This is welfare maximizing allocation:",allocation)
    return allocation

#calculates the utilitarian welfare of any given allocation
def calculate_total_welfare(valuations, allocation):
    total_welfare = 0
    for pair in allocation:
        # Accept both (agent_idx, house_idx) or ('A0', 'H3')
        if isinstance(pair[0], str):
            agent_idx = int(pair[0][1:])
            house_idx = int(pair[1][1:])
        else:
            agent_idx, house_idx = pair
        total_welfare += valuations[agent_idx, house_idx]
    return total_welfare


def nash_social_welfare(allocation, M):
    utility = np.zeros(M.shape[0])
    for agent, house in allocation:
        utility[agent] = M[agent][house]
    
    if np.any(utility == 0):
        return 0  # NSW is zero if any agent has zero utility
    return np.prod(utility)


# Nash via brute-force:
def nsw_maximizing_allocation(M):
    n_agents, n_houses = M.shape
    best_allocation = []
    max_nsw = -1
    houses = list(range(n_houses))
    # Try all possible injective mappings of agents to houses
    for perm in itertools.permutations(houses, n_agents):
        allocation = list(zip(range(n_agents), perm))
        nsw = nash_social_welfare(allocation, M)
        if nsw > max_nsw:
            max_nsw = nsw
            best_allocation = allocation
    print("This is NSW maximizing allocation:", best_allocation)
    return best_allocation

#Nash via maximmizing sum of logs:"
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


#calculates total envy in any allocation
def calculate_total_envy(allocation_dict, valuations):
    if isinstance(allocation_dict, list):
        allocation_dict = {f'i{a}': f'h{h}' for (a, h) in allocation_dict}
    else:
        allocation_dict = allocation_dict
    allocation = [(int(a[1:]), int(h[1:])) for a, h in allocation_dict.items()]
    envy = 0
    for a in range(valuations.shape[0]):
        agent_houses = [h for (agent, h) in allocation if agent == a]
        if not agent_houses:
            continue
        current_house = agent_houses[0]
        current_value = valuations[a, current_house]
        
        for (other_agent, other_house) in allocation:
            if valuations[a, other_house] > current_value:
                envy += valuations[a, other_house] - current_value
    return envy



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

def apply_q_components_brute_force(max_welfare_allocation, reordered_labeled_paths, valuations, q=4):
    original_envy = calculate_total_envy(max_welfare_allocation, valuations)
    
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
            new_envy = calculate_total_envy(temp_allocation, valuations)
            envy_reduction = original_envy - new_envy
            
            # Update best if this combination is better
            if envy_reduction > max_envy_reduction:
                max_envy_reduction = envy_reduction
                best_combination = combination
                best_allocation = temp_allocation
    
    return best_allocation

# def apply_q_components_optimized_brute_force(max_welfare_allocation, reordered_labeled_paths, valuations, q=4):
    original_envy = calculate_total_envy(max_welfare_allocation, valuations)
    
    # Precompute path properties
    path_data = []
    for i, path in enumerate(reordered_labeled_paths):
        agents_affected = len({node for node in path if node.startswith('i')})
        path_data.append((i, agents_affected, path))
    
    # Sort by agents affected to enable early pruning
    path_data.sort(key=lambda x: x[1])
    
    memo = {}  # Memoization for combinations
    
    def evaluate_combination(indices):
        """Evaluate envy reduction for a specific combination of path indices."""
        indices_tuple = tuple(sorted(indices))
        if indices_tuple in memo:
            return memo[indices_tuple]
        
        selected_paths = [path_data[i][2] for i in indices]
        temp_allocation = apply_all_components(max_welfare_allocation, selected_paths)
        new_envy = calculate_total_envy(temp_allocation, valuations)
        envy_reduction = original_envy - new_envy
        
        memo[indices_tuple] = envy_reduction
        return envy_reduction
    
    def backtrack(start_idx, current_combination, current_agents, max_agents):
        """Recursive backtracking with pruning."""
        nonlocal best_combination, max_envy_reduction, best_allocation
        
        # Evaluate current combination if non-empty
        if current_combination:
            envy_reduction = evaluate_combination(current_combination)
            if envy_reduction > max_envy_reduction:
                max_envy_reduction = envy_reduction
                best_combination = current_combination.copy()
                selected_paths = [path_data[i][2] for i in current_combination]
                best_allocation = apply_all_components(max_welfare_allocation, selected_paths)
        
        # Try adding more paths
        for i in range(start_idx, len(path_data)):
            _, agents_needed, _ = path_data[i]
            
            # Pruning: skip if adding this path exceeds budget
            if current_agents + agents_needed > max_agents:
                break  # Since paths are sorted by agents_needed, no point continuing
            
            # Add this path to combination
            current_combination.append(i)
            backtrack(i + 1, current_combination, current_agents + agents_needed, max_agents)
            current_combination.pop()  # Backtrack
    
    best_combination = []
    max_envy_reduction = -1
    best_allocation = max_welfare_allocation
    
    backtrack(0, [], 0, q)
    
    return best_allocation



def apply_q_components(max_welfare_allocation, reordered_labeled_paths, valuations, q=4):
    # Calculate original envy for reference
    original_envy = calculate_total_envy(max_welfare_allocation, valuations)
    # Precompute properties for each path
    items = []
    for path in reordered_labeled_paths:
        # Calculate number of affected agents
        agents_affected = len({node for node in path if node.startswith('i')})
        
        # Calculate envy reduction by applying this path
        temp_alloc = apply_all_components(max_welfare_allocation, [path])
        new_envy = calculate_total_envy(temp_alloc, valuations)
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

def calculate_envy_reduction(path, A_hat, valuations):
    temp_alloc = apply_all_components(A_hat, path)
    agent_houses = {a: h for a, h in A_hat}
    original_envy = calculate_total_envy(A_hat, valuations)
    new_envy = calculate_total_envy(temp_alloc, valuations)
    return original_envy - new_envy

def compute_min_envy(A, A_hat, valuations):
    n_agents = len(A)
    m_houses = valuations.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_agents + m_houses))
    for a in range(n_agents):
        for h in range(m_houses):
            G.add_edge(a, n_agents + h)

    vertex_colors, edge_colors = good_coloring(G, A, A_hat, n_agents, m_houses)
    
    feasible_comps, feasible_paths = find_feasible_components(G, vertex_colors, edge_colors, A_hat, valuations)

    labeled_paths = convert_feasible_paths(feasible_paths, n_agents)  
    reordered_labeled_paths = reorder_labeled_paths(labeled_paths, optimal_allocation)

    #Apply All Components gives the optimal solution. Converts max_welfare_alloctaion to min_total_envy_allocation.
    A_algo_output = apply_all_components(max_welfare_allocation, reordered_labeled_paths)

    #Apply only chosen q components by the knapsack. Choose those q components that gives maximum reduction in the envy
    A_algo_output_pruned_q = apply_q_components_brute_force(max_welfare_allocation, reordered_labeled_paths, valuations)


    #After that, just compute the welfare values etc.
    welfare_in_Fair = calculate_total_welfare(valuations, A_algo_output)
    welfare_in_max_welfare = calculate_total_welfare(valuations, max_welfare_allocation)
    welfare_in_Fair_q = calculate_total_welfare(valuations, A_algo_output_pruned_q)
    welfare_max_values.append(welfare_in_max_welfare)
    welfare_fair_values.append(welfare_in_Fair)
    welfare_fair_q_values.append(welfare_in_Fair_q)

    # 4. Prepare knapsack inputs
    items_n_C = []
    items_r_C = []
    items = []

    r_C = calculate_envy_reduction(reordered_labeled_paths, A_hat, valuations)
    for comp in feasible_comps:
        agents = {n for n in comp if n < n_agents}
        n_C = len(agents)
        items_n_C.append(n_C)
        items_r_C.append(r_C)
    for item in range(len(items_n_C)):
        items.append((items_r_C[item], items_n_C[item]))

    # 5. Solve knapsack problem

    # print(items)

    q = sum(n for _, n in items)

    k = r_C

    print("k, q:", k, q)

    # dp = [0]*(q+1)
    # print(dp)
    # for r, n in items:
    #     for w in range(q, n-1, -1):
    #         if dp[w - n] + r > dp[w]:
    #             dp[w] = dp[w - n] + r

    return k, q


n_agents = 6
houses_list = [7,8,9,10,11]
num_instances = 100


avg_welfare_max = []
avg_welfare_fair = []
avg_welfare_fair_q = []
ci_max_upper = []
ci_max_lower = []
ci_fair_upper = []
ci_fair_lower = []
ci_fair_q_upper = []
ci_fair_q_lower = []


for m_houses in houses_list:
    welfare_max_values = []
    welfare_fair_values = []
    welfare_fair_q_values = []

    for _ in range(num_instances):
        #find an optimal min total envy allocation with maximium welfare and store it in optimal allocation
        valuations = instance(n_agents, m_houses)
        # print(valuations)

        min_total_envy = float('inf')
        best_allocations = []
        houses_idx = list(range(m_houses))

        for perm in permutations(houses_idx, n_agents):
            allocation = [(agent, house) for agent, house in enumerate(perm)]
            current_envy = calculate_total_envy(allocation, valuations)
            if current_envy < min_total_envy:
                min_total_envy = current_envy
                best_allocations = [allocation]
            elif current_envy == min_total_envy:
                best_allocations.append(allocation)

        max_w = -1
        optimal_allocation = None
        for alloc in best_allocations:
            welfare = calculate_total_welfare(valuations, alloc)
            if welfare > max_w:
                max_w = welfare
                optimal_allocation = alloc

        

        G = nx.Graph()
        n_agents, m_houses = valuations.shape
        for agent in range(n_agents):
            for house in range(m_houses):
                if valuations[agent, house] > 0:
                    G.add_edge(f'i{agent}', f'h{house}')
        
        max_welfare_allocation = max_weighted_matching(valuations)

        print("This is optimal min total envy allocation:", optimal_allocation)

        alternating_components = find_alternating_components(optimal_allocation, max_welfare_allocation, valuations)
        A_hat_paths = apply_all_components(max_welfare_allocation, alternating_components)
        k, q = compute_min_envy(optimal_allocation, max_welfare_allocation, valuations)

    # Calculate mean and 95% CI for max welfare
    mean_max = np.mean(welfare_max_values)
    sem_max = np.std(welfare_max_values, ddof=1) / np.sqrt(num_instances)
    avg_welfare_max.append(mean_max)
    ci_max_upper.append(mean_max + 1.96 * sem_max)
    ci_max_lower.append(mean_max - 1.96 * sem_max)

    # Calculate mean and 95% CI for fair welfare
    mean_fair = np.mean(welfare_fair_values)
    sem_fair = np.std(welfare_fair_values, ddof=1) / np.sqrt(num_instances)
    avg_welfare_fair.append(mean_fair)
    ci_fair_upper.append(mean_fair + 1.96 * sem_fair)
    ci_fair_lower.append(mean_fair - 1.96 * sem_fair)

    # Calculate mean and 95% CI for fair_q welfare:
    mean_fair_q = np.mean(welfare_fair_q_values)    
    sem_fair_q = np.std(welfare_fair_q_values, ddof=1) / np.sqrt(num_instances)
    avg_welfare_fair_q.append(mean_fair_q)
    ci_fair_q_upper.append(mean_fair_q + 1.96 * sem_fair_q)
    ci_fair_q_lower.append(mean_fair_q - 1.96 * sem_fair_q)


    # avg_welfare_fair.append(np.mean(welfare_fair_values))
    # avg_welfare_max.append(np.mean(welfare_max_values))



plt.figure(figsize=(10, 6))
plt.plot(houses_list, avg_welfare_max, label='Max Welfare Allocation', marker='o', color='blue')
plt.fill_between(houses_list, ci_max_lower, ci_max_upper, color='blue', alpha=0.2)
plt.plot(houses_list, avg_welfare_fair, label='Min Total Envy Allocation', marker='x', color='orange')
plt.fill_between(houses_list, ci_fair_lower, ci_fair_upper, color='orange', alpha=0.2)
plt.plot(houses_list, avg_welfare_fair_q, label='Min Total Envy Allocation (q)', marker='s', color='green')
plt.fill_between(houses_list, ci_fair_q_lower, ci_fair_q_upper, color='green', alpha=0.2)
plt.xlabel('Number of Houses',fontsize=16)
plt.ylabel('Average Total Welfare',fontsize=16)
plt.title('Welfare Loss in the Min Total Envy Allocation',fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


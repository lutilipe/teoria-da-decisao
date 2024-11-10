import numpy as np
import pandas as pd
import statistics
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt

""" 
n = ativo
m = base
s = equipe
"""

pd.set_option('display.float_format', lambda x: '%.8f' % x)

class Struct:
    pass

def parse_ativos(df):
    return list(set(list(df.iloc[:, [2, 3]].itertuples(index=False, name=None))))

def parse_base(df):
    return list(set(list(df.iloc[:, [0, 1]].itertuples(index=False, name=None))))

def parse_dist(df, p):
    distance_matrix = pd.DataFrame(0.0, 
        index=pd.MultiIndex.from_tuples(p.n), 
        columns=pd.MultiIndex.from_tuples(p.m))
    
    for _, row in df.iterrows():
        distance_matrix.loc[(row['lat_a'], row['lon_a']), (row['lat_b'], row['lon_b'])] = row['distance']
    
    return distance_matrix

def convert_to_float(value):
    return float(value.replace(",", "."))

def probdef():
    df = pd.read_csv('probdata.csv', names=['lat_b', 'lon_b', 'lat_a', 'lon_a', 'distance'], sep=';')
    df['lat_b'] = df['lat_b'].apply(convert_to_float)
    df['lon_b'] = df['lon_b'].apply(convert_to_float)
    df['lat_a'] = df['lat_a'].apply(convert_to_float)
    df['lon_a'] = df['lon_a'].apply(convert_to_float)
    df['distance'] = df['distance'].apply(convert_to_float)

    probdata = Struct()
    probdata.s = [1, 2, 3] # 3
    probdata.n = parse_ativos(df)
    probdata.m = parse_base(df)
    probdata.d = parse_dist(df, probdata)
    probdata.eta = 0.2

    return probdata

def penalizacao_restricoes(solution, p):
    P = 10
    X, Y, H = solution.x, solution.y, solution.h

    # Restrição 1: Cada equipe deve ser alocada a uma única base de manutenção
    penalidade = np.abs(Y.sum(axis=0) - 1).sum() * P

    # Restrição 2: Cada ativo deve ser atribuído a exatamente uma base
    penalidade += np.abs(X.sum(axis=1) - 1).sum() * P

    # Restrição 3: Ativo só pode ser atribuído a uma base com equipe alocada
    penalidade += P * ((X.values @ (Y.sum(axis=1) < 1).values).sum())

    # Restrição 4: Cada ativo deve ser atribuído a exatamente uma equipe
    penalidade += np.abs(H.sum(axis=1) - 1).sum() * P

    # Restrição 5: Ativo só pode ser atribuído a equipe que está na mesma base
    for i, j in np.argwhere(X.values):
        penalidade += P * ((H.loc[p.n[i]] * (1 - Y.loc[p.m[j]])).sum())

    # Restrição 6: Cada equipe deve ser responsável por pelo menos η ativos
    penalidade += np.maximum(0, (p.eta * len(p.n) / len(p.s) - H.sum(axis=0))).sum() * P

    return penalidade

def f1(solution, p):
    result = (p.d.values * solution.x.values).sum()

    solution.fitness = result
    solution.penalty = penalizacao_restricoes(solution, p)
    solution.penalized_fitness = solution.fitness + solution.penalty
            
    return solution

def f2(solution, p):
    assets_maintained_by_teams = solution.h.sum(axis=0)
    
    max_assets_maintained = assets_maintained_by_teams.max()
    min_assets_maintained = assets_maintained_by_teams.min()

    result = max_assets_maintained - min_assets_maintained

    solution.fitness = result
    solution.penalty = penalizacao_restricoes(solution, p)
    solution.penalized_fitness = solution.fitness + solution.penalty
            
    return solution

def initialize_solution(p, apply_constructive_heuristic=False):
    num_ativos = len(p.n)
    num_bases = len(p.m)
    num_equipes = len(p.s)

    solution = Struct()

    X = pd.DataFrame(np.random.rand(num_ativos, num_bases), 
                    index=pd.MultiIndex.from_tuples(p.n), 
                    columns=pd.MultiIndex.from_tuples(p.m))
    Y = pd.DataFrame(np.random.rand(num_bases, num_equipes), 
                    index=pd.MultiIndex.from_tuples(p.m), 
                    columns=p.s)
    H = pd.DataFrame(np.random.rand(num_ativos, num_equipes), 
                    index=pd.MultiIndex.from_tuples(p.n), 
                    columns=p.s)

    if apply_constructive_heuristic:
        X.loc[:, :] = 0
        Y.loc[:, :] = 0
        H.loc[:, :] = 0
        
        occupied_bases = []
        for team in p.s:
            base = p.m[team % num_bases]
            Y.loc[base, team] = 1
            occupied_bases.append(base)

        occupied_bases = list(set(occupied_bases))
        num_occupied_bases = len(occupied_bases)

        for i, asset in enumerate(p.n):
            base = occupied_bases[i % num_occupied_bases]
            X.loc[asset, base] = 1

            for team in p.s:
                if Y.loc[base, team] == 1:
                    H.loc[asset, team] = 1
                    break 

    solution.x, solution.y, solution.h = X, Y, H
    return solution

def neighborhood_change(current_solution, new_solution, k):
    if new_solution.penalized_fitness < current_solution.penalized_fitness:
        return new_solution, 1
    return current_solution, k + 1

def local_search_first_improvement(solution, p):
    current_solution = solution


def GVNS(initial_solution, objective_function, p):
    max_num_evaluations = 10
    num_evaluations = 0
    k_max = 3
    
    current_solution = initial_solution
    new_solution = initial_solution

    history = Struct()
    history.fitness = []
    history.penalty = []
    history.penalized_fitness = []
    
    while num_evaluations < max_num_evaluations:
        k = 1
        while k <= k_max:
            #new_solution = shake(current_solution, k, p)
            #new_solution = local_search_first_improvement(new_solution, p)
            new_solution = objective_function(new_solution, p)
            num_evaluations += 1
            new_solution.id = num_evaluations

            current_solution, k = neighborhood_change(current_solution, new_solution, k)

            history.fitness.append(current_solution.fitness)
            history.penalty.append(current_solution.penalty)
            history.penalized_fitness.append(current_solution.penalized_fitness)

            print(num_evaluations)
            
    return current_solution, history

def optimize_instance(objective_function, p):
    initial_solution = initialize_solution(p, True)
    best_solution, history = GVNS(initial_solution, objective_function, p)
    return best_solution, history

def optimize(p, objective_function):
    results = []

    num_instances = 5 

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(optimize_instance, objective_function, p) for _ in range(num_instances)]
        results = [future.result() for future in futures]
    return results

def get_metrics(results):
    max_sol = None
    min_sol = None
    std_dev = statistics.stdev([x[0].penalized_fitness for x in results])

    for solution, _ in results:
        if (max_sol == None and min_sol == None):
            max_sol = min_sol = solution
            continue

        if solution.penalized_fitness > max_sol.penalized_fitness:
            max_sol = solution

        if solution.penalized_fitness < min_sol.penalized_fitness:
            min_sol = solution

    return std_dev, max_sol, min_sol

def plot_convergence(results, fobj):
    histories = [x[1] for x in results]
    plt.figure(figsize=(10, 6))
    for history in histories:
        s = len(history.penalized_fitness)
        plt.plot(np.linspace(0, s-1, s), history.penalized_fitness, 'k-', alpha=0.7)

    plt.title(f'Evolução da qualidade da solução candidata - {fobj}')
    plt.ylabel('fitness(x) penalizado')
    plt.xlabel('Número de avaliações')
    plt.grid(True)
    
    plt.savefig(f"ev_{fobj}.png", format='png', dpi=300, bbox_inches='tight')

def visualize_network(p, solution, fobj):
    active_base = []
    available_base = []
    for base in p.m:
        is_occupied = False
        for team in p.s:
            if solution.y.loc[base, team] == 1:
                is_occupied = True
                break
        node_type = "base" if not is_occupied else "active_base"
        if (node_type == "base"):
            available_base.append(base)
        else:
            active_base.append(base)

    active_base_lat, active_base_lon = zip(*active_base)  
    available_base_lat, available_base_lon = zip(*available_base)  
    lat_n, lon_n = zip(*p.n)

    team_colors = {
        team: color for team, color in zip(p.s, ['#1f78b4', '#33a02c', '#e31a1c'])
    }

    plt.figure(figsize=(10, 8))
    
    used_labels = set()

    for ativo in p.n:
        for base in p.m:
            if solution.x.loc[ativo, base] == 1:
                team_responsible = None
                for team in p.s:
                    if solution.h.loc[ativo, team] == 1:
                        team_responsible = team
                        break
                if team_responsible is not None:
                    label = f"Equipe {team_responsible}"
                    if label not in used_labels:
                        plt.plot(
                            [base[1], ativo[1]], [base[0], ativo[0]],
                            color=team_colors[team_responsible], linestyle='--', linewidth=0.5, label=label
                        )
                        used_labels.add(label)
                    else:
                        plt.plot(
                            [base[1], ativo[1]], [base[0], ativo[0]],
                            color=team_colors[team_responsible], linestyle='--', linewidth=0.5
                        )

    plt.scatter(active_base_lon, active_base_lat, color='#fc8d62', label='Base Ocupada', edgecolors='black', linewidths=1.5, marker='p', s=100)
    plt.scatter(available_base_lon, available_base_lat, color='#66c2a5', label='Base Disponível', edgecolors='black', linewidths=1.5, marker='p', s=100)
    plt.scatter(lon_n, lat_n, color='#8da0cb', label='Ativo', edgecolors='black', linewidths=1, marker='o')
    
    plt.legend(scatterpoints=1, markerscale=1, loc='upper left', 
               bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True, 
               borderpad=1.2, fontsize=10)
    plt.xlabel('Longitude', fontweight="bold")
    plt.ylabel('Latitude', fontweight="bold")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f"network_{fobj}.png", format='png', dpi=300, bbox_inches='tight')

p = probdef()

# F1
results_f1 = optimize(p, f1)
std_dev_f1, max_sol_f1, min_sol_f1 = get_metrics(results_f1)

print(f"Desvio Padrao - f1: {std_dev_f1}")
print(f"Máximo - f1: {max_sol_f1.penalized_fitness}")
print(f"Mínimo - f1: {min_sol_f1.penalized_fitness}")

plot_convergence(results_f1, "f1")
visualize_network(p, min_sol_f1, "f1")

# F2
results_f2 = optimize(p, f2)
std_dev_f2, max_sol_f2, min_sol_f2 = get_metrics(results_f2)

print(f"Desvio Padrao - f2: {std_dev_f2}")
print(f"Máximo - f2: {max_sol_f2.penalized_fitness}")
print(f"Mínimo - f2: {min_sol_f2.penalized_fitness}")

plot_convergence(results_f2, "f2")
visualize_network(p, min_sol_f2, "f2")

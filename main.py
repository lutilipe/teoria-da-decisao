import numpy as np
import pandas as pd
import statistics
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
import copy
from concurrent.futures import as_completed
from sklearn.cluster import KMeans
from itertools import combinations
import random

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
    P = 20
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

def heuristica_construtiva(X, Y, H, p):
    n_ativos = len(p.n)
    s_equipes = len(p.s)
    min_ativos_por_equipe = max(1, int(0.2 * n_ativos / s_equipes))

    bases_selecionadas = random.sample(p.m, 3)
    for idx, equipe in enumerate(p.s):
        base = bases_selecionadas[idx]
        Y.loc[base, equipe] = 1

    for ativo in p.n:
        base_mais_proxima = min(
            bases_selecionadas,
            key=lambda base: p.d.loc[ativo, base] if Y.loc[base].sum() > 0 else float('inf')
        )
        X.loc[ativo, base_mais_proxima] = 1

        for equipe in p.s:
            if Y.loc[base_mais_proxima, equipe] == 1:
                H.loc[ativo, equipe] = 1
                break

    for base in bases_selecionadas:
        ativos_na_base = X.loc[:, base].sum()
        while ativos_na_base < min_ativos_por_equipe:
            # Fix
            outra_base = max(
                [curr_base for curr_base in bases_selecionadas if curr_base != base],
                key=lambda curr_base: X.loc[:, curr_base].sum()
            )
            if X.loc[:, outra_base].sum() > min_ativos_por_equipe:
                ativos_outra_base = X.loc[:, outra_base][X.loc[:, outra_base] == 1].index
                ativo_realocado = min(ativos_outra_base, key=lambda ativo: p.d.loc[(ativo, base)])
                X.loc[ativo_realocado, outra_base] = 0
                X.loc[ativo_realocado, base] = 1
                for equipe in p.s:
                    if Y.loc[base, equipe] == 1:
                        H.loc[ativo_realocado, equipe] = 1
                    else:
                        H.loc[ativo_realocado, equipe] = 0
                ativos_na_base += 1
                break

    solution = Struct()
    solution.x, solution.y, solution.h = X, Y, H
    return solution

def heuristica_construtiva2(X, Y, H, p, objective_function):
    base_combinations = list(combinations(p.m, 3))

    best_start_solution = None

    for combination in base_combinations:
        kmeans = KMeans(n_clusters=len(combination), init=combination, n_init=1)
        kmeans.fit(p.n)
        
        labels = kmeans.labels_

        solution = Struct()

        solution.x, solution.y, solution.h = copy.copy(X), copy.copy(Y), copy.copy(H),

        for equipe in p.s:
            base = combination[equipe % len(combination)]
            solution.y.loc[base, equipe] = 1

        assets_per_team = { team: 0 for team in p.s }

        for idx, label in enumerate(labels):
            base = combination[label]
            ativo = p.n[idx]
            solution.x.loc[ativo, base] = 1

            teams_at_base = []

            for equipe in p.s:
                if solution.y.loc[base, equipe] == 1:
                    teams_at_base.append(equipe)

            min_team = min(teams_at_base, key=lambda team: assets_per_team[team])
            solution.h.loc[ativo, min_team] = 1
            assets_per_team[min_team] += 1

        solution = objective_function(solution, p)
        if best_start_solution == None or solution.penalized_fitness < best_start_solution.penalized_fitness:
            best_start_solution = solution
    
    return best_start_solution

def initialize_solution(p, objective_function):
    X = pd.DataFrame(0, 
                    index=pd.MultiIndex.from_tuples(p.n), 
                    columns=pd.MultiIndex.from_tuples(p.m))
    Y = pd.DataFrame(0, 
                    index=pd.MultiIndex.from_tuples(p.m), 
                    columns=p.s)
    H = pd.DataFrame(0, 
                    index=pd.MultiIndex.from_tuples(p.n), 
                    columns=p.s)

    solution = heuristica_construtiva(X,Y,H,p)
    #solution = heuristica_construtiva2(X,Y,H,p,objective_function)

    return solution

def neighborhood_change(current_solution, new_solution, k):
    if new_solution.penalized_fitness < current_solution.penalized_fitness:
        return new_solution, 1
    return current_solution, k + 1

def best_improvement_local_search(objective_function, solution, p):
    current_solution = copy.deepcopy(solution)
    improved = True
    while improved:
        improved = False
        for ativo in p.n:
            new_solution1 = objective_function(taskMove_Ativo(copy.deepcopy(current_solution), p), p)
            new_solution2 = objective_function(swap_Ativo_Base(copy.deepcopy(current_solution), p), p)
            new_solution = new_solution1 if new_solution1.penalized_fitness < new_solution2.penalized_fitness else new_solution2
            if new_solution.penalized_fitness < current_solution.penalized_fitness:
                print("LOCAL IMPROVED", current_solution.penalized_fitness, new_solution.penalized_fitness)
                current_solution = copy.deepcopy(new_solution)
                improved = True
                break

    return current_solution

def ativos_por_base(base, X):
    ativos = list()
    for i in p.n:
        if X.loc[i, base] == 1:
            ativos.append(i)
    return ativos

def bases_com_equipes(Y):
    bases = list()
    for base in p.m:
        if Y.loc[base, :].sum() >= 1:
            bases.append(base)
            
    return bases

def swap_Ativo_Base(solution, p):
    # Realocação de duas tarefas aleatórias entre duas máquinas.
    # Selecionar dois ativos aleatórios que estão alocados a bases diferentes e trocar suas alocações.

    bases_ativas = bases_com_equipes(solution.y)

    base1 = random.choice(bases_ativas)
    ativos1 = ativos_por_base(base1, solution.x)
    ativo1 = random.choice(ativos1)
    bases_ativas.remove(base1)

    base2 = random.choice(bases_ativas)
    ativos2 = ativos_por_base(base2, solution.x)
    ativo2 = random.choice(ativos2)

    solution.x.loc[ativo1, base1] = 0
    solution.x.loc[ativo2, base2] = 0
    solution.x.loc[ativo2, base1] = 1
    solution.x.loc[ativo1, base2] = 1

    for k in p.s:
        if solution.h.loc[ativo1, k] == 1:
            equipe1 = k
            break

    for k in p.s:
        if solution.h.loc[ativo2, k] == 1:
            equipe2 = k
            break

    solution.h.loc[ativo1, equipe1] = 0
    solution.h.loc[ativo2, equipe2] = 0
    solution.h.loc[ativo2, equipe1] = 1
    solution.h.loc[ativo1, equipe2] = 1

    return solution

def taskMove_Ativo(solution, p):
    # Um único ativo é movido de sua base atual para uma nova base sem uma troca direta com outro ativo. Não é uma troca, aqui todos os ativos seguintes à nova posição serão modificados também

    ativo = random.choice(p.n)

    bases_ativas = bases_com_equipes(solution.y)

    for j in bases_ativas:
        if solution.x.loc[ativo, j] == 1:
            base_atual = j
            break

    bases_ativas.remove(base_atual)
    base_nova = random.choice(bases_ativas)

    equipes = copy.deepcopy(p.s)

    for k in equipes:
        if solution.h.loc[ativo, k] == 1:
            equipe_atual = k
            break

    equipes.remove(equipe_atual)
    equipe_nova = np.random.choice(equipes)

    solution.x.loc[ativo, base_atual] = 0
    solution.x.loc[ativo, base_nova] = 1
    solution.h.loc[ativo, equipe_atual] = 0
    solution.h.loc[ativo, equipe_nova] = 1

    return solution

def taskMove_Equipe(solution, p):
    # Movimentação de uma tarefa de uma máquina de origem para uma outra máquina.
    # Realocação de uma equipe inteira de uma base para outra, mantendo seus ativos.

    equipe = np.random.choice(p.s)
    bases_disponiveis = copy.deepcopy(p.m)

    for base in bases_disponiveis:
        if solution.y.loc[base, equipe] == 1:
            base_atual = base
            bases_disponiveis.remove(base)
            break

    for base in bases_disponiveis:
        if solution.y.loc[base, :].sum() > 0:
            bases_disponiveis.remove(base)

    base_nova = random.choice(bases_disponiveis)
    solution.y.loc[base_atual, equipe] = 0
    solution.y.loc[base_nova, equipe] = 1

    for ativo in p.n:
        if solution.x.loc[ativo, base_atual] == 1:
            solution.x.loc[ativo, base_atual] = 0
            solution.x.loc[ativo, base_nova] = 1

    return solution

def taskMove_Equipe2(solution, p, k):
    # Realocação de uma equipe inteira de uma base para outra, mantendo seus ativos.

    bases_disponiveis = copy.deepcopy(p.m)

    bases_atuais = []
    for base in p.m:
        row = solution.y.loc[base, :]
        if row.sum() > 0:
            for team, is_allocated in row.items():
                if is_allocated == 1:
                    bases_atuais.append((base, team))
            bases_disponiveis.remove(base)

    random.shuffle(bases_atuais)
    random.shuffle(bases_disponiveis)

    selected_bases = random.sample(bases_atuais, k)
    bases_to_change = random.sample(bases_disponiveis, k)

    for idx in range(k):
        base_nova = bases_to_change[idx]
        base_atual = selected_bases[idx][0]
        equipe_atual = selected_bases[idx][1]
        solution.y.loc[base_atual, equipe_atual] = 0
        solution.y.loc[base_nova, equipe_atual] = 1

        for ativo in p.n:
            if solution.x.loc[ativo, base_atual] == 1:
                solution.x.loc[ativo, base_atual] = 0
                solution.x.loc[ativo, base_nova] = 1

    return solution

def shake(solution, k, p):
    new_solution = copy.deepcopy(solution)
    return taskMove_Equipe2(new_solution, p, k)

def GVNS(objective_function, p):
    initial_solution = initialize_solution(p, objective_function)

    max_num_evaluations = 100
    num_evaluations = 0
    k_max = 3
    
    initial_solution = objective_function(initial_solution, p)
    current_solution = initial_solution
    new_solution = initial_solution

    history = Struct()
    history.fitness = []
    history.penalty = []
    history.penalized_fitness = []
    
    while num_evaluations < max_num_evaluations:
        k = 1
        while k <= k_max:
            new_solution = shake(current_solution, k, p)
            new_solution = best_improvement_local_search(objective_function, new_solution, p)
            new_solution = objective_function(new_solution, p)
            
            num_evaluations += 1
            old_fit = current_solution.penalized_fitness
            print(old_fit, new_solution.penalized_fitness)
            current_solution, k = neighborhood_change(current_solution, new_solution, k)
            if old_fit > new_solution.penalized_fitness:
                print("MELHOROU")

            history.penalized_fitness.append(current_solution.penalized_fitness)

    return current_solution, history

def optimize_instance(objective_function, p, seed):
    np.random.seed(seed)
    best_solution, history = GVNS(objective_function, p)
    return best_solution, history

def optimize(p, objective_function):
    results = []

    num_instances = 1

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(optimize_instance, objective_function, copy.deepcopy(p), seed=np.random.randint(0, 10000)) for _ in range(num_instances)]
        for future in as_completed(futures):
            print("COMPLETED")
            result = future.result()
            results.append(result)
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
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(histories)))

    for idx, history in enumerate(histories):
        s = len(history.penalized_fitness)
        plt.plot(np.linspace(0, s-1, s), history.penalized_fitness, color=colors[idx], alpha=0.7, label=f'Teste {idx+1}')
    
    plt.title(f'Evolução da qualidade da solução candidata - {fobj}')
    plt.ylabel('fitness(x) penalizado')
    plt.xlabel('Número de avaliações')
    plt.legend(title="Testes")
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
print(f"Mínimo - f1: {min_sol_f1.penalty}")

plot_convergence(results_f1, "f1")
visualize_network(p, min_sol_f1, "f1")

""" # F2
results_f2 = optimize(p, f2)
std_dev_f2, max_sol_f2, min_sol_f2 = get_metrics(results_f2)

print(f"Desvio Padrao - f2: {std_dev_f2}")
print(f"Máximo - f2: {max_sol_f2.penalized_fitness}")
print(f"Mínimo - f2: {min_sol_f2.penalized_fitness}")
print(f"Mínimo - f1: {min_sol_f2.penalty}")

plot_convergence(results_f2, "f2")
visualize_network(p, min_sol_f2, "f2") """

import numpy as np
import pandas as pd
import statistics
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
import copy
from concurrent.futures import as_completed

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

def heuristica_construtiva(X, Y, H, p):
    n_bases = len(p.m)
    n_ativos = len(p.n)
    s_equipes = len(p.s)
    min_ativos_por_equipe = max(1, int(0.2 * n_ativos / s_equipes))

    bases_selecionadas = []
    for equipe in p.s:
        base = p.m[equipe % n_bases]
        Y.loc[base, equipe] = 1
        bases_selecionadas.append(base)

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
            for outra_base in bases_selecionadas:
                if X.loc[:, outra_base].sum() > min_ativos_por_equipe:
                    ativo_realocado = X.loc[:, outra_base].idxmax()
                    X.loc[ativo_realocado, outra_base] = 0
                    X.loc[ativo_realocado, base] = 1
                    for equipe in p.s:
                        if Y.loc[base, equipe] == 1:
                            H.loc[ativo_realocado, equipe] = 1
                            break
                    ativos_na_base += 1
                    break

def initialize_solution(p):
    solution = Struct()

    X = pd.DataFrame(0, 
                    index=pd.MultiIndex.from_tuples(p.n), 
                    columns=pd.MultiIndex.from_tuples(p.m))
    Y = pd.DataFrame(0, 
                    index=pd.MultiIndex.from_tuples(p.m), 
                    columns=p.s)
    H = pd.DataFrame(0, 
                    index=pd.MultiIndex.from_tuples(p.n), 
                    columns=p.s)

    heuristica_construtiva(X,Y,H,p)

    solution.x, solution.y, solution.h = X, Y, H
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

        for _ in range(3):
            new_solution = shake(current_solution, 2, p)
            new_solution = objective_function(new_solution, p)
            
            if new_solution.penalized_fitness < current_solution.penalized_fitness:
                current_solution = new_solution
                improved = True

    return current_solution

def ativos_por_base(base, X):
    ativos = list()
    for i in p.n:
        if X.loc[i, base] == 1:
            ativos.append(i)
    return ativos

def bases_com_equipes(Y):
    bases = list()
    for j in p.m:
        for k in p.s:
            if Y.loc[j, k] == 1:
                bases.append(j)
    return bases

def swap_Ativo_Base(solution, p):
    # Realocação de duas tarefas aleatórias entre duas máquinas.
    # Selecionar dois ativos aleatórios que estão alocados a bases diferentes e trocar suas alocações.

    bases_ativas = bases_com_equipes(solution.y)

    base1 = bases_ativas[np.random.randint(0, len(bases_ativas))]
    ativos1 = ativos_por_base(base1, solution.x)
    ativo1 = ativos1[np.random.randint(0, len(ativos1))]

    bases_ativas.remove(base1)
    base2 = bases_ativas[np.random.randint(0, len(bases_ativas))]
    ativos2 = ativos_por_base(base2, solution.x)
    ativo2 = ativos2[np.random.randint(0, len(ativos2))]

    solution.x.loc[ativo1, base1] = 0
    solution.x.loc[ativo2, base2] = 0
    solution.x.loc[ativo2, base1] = 1
    solution.x.loc[ativo1, base2] = 1

    for k in p.s:
        if solution.h.loc[ativo1, k] == 1:
            equipe1 = k
        if solution.h.loc[ativo2, k] == 1:
            equipe2 = k

    solution.h.loc[ativo1, equipe1] = 0
    solution.h.loc[ativo2, equipe2] = 0
    solution.h.loc[ativo2, equipe1] = 1
    solution.h.loc[ativo1, equipe2] = 1

    return solution

def taskMove_Ativo(solution, p):
    # Movimentação de uma tarefa de uma máquina de origem para uma outra máquina.
    # Um único ativo é movido de sua base atual para uma nova base sem uma troca direta com outro ativo. Não é uma troca, aqui todos os ativos seguintes à nova posição serão modificados também

    ativo = p.n[np.random.randint(0, len(p.n))]

    bases_ativas = bases_com_equipes(solution.y)

    for j in bases_ativas:
        if solution.x.loc[ativo, j] == 1:
            base_atual = j
            break

    bases_ativas.remove(base_atual)
    base_nova = bases_ativas[np.random.randint(0, len(bases_ativas))]

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

    for j in bases_disponiveis:
        if solution.y.loc[j, equipe] == 1:
            base_atual = j
            bases_disponiveis.remove(j)
            break
    base_nova = bases_disponiveis[np.random.randint(0, len(bases_disponiveis))]
    solution.y.loc[base_atual, equipe] = 0
    solution.y.loc[base_nova, equipe] = 1

    for i in p.n:
        if solution.x.loc[i, base_atual] == 1:
            solution.x.loc[i, base_atual] = 0
        if solution.x.loc[i, base_nova] == 0:
            solution.x.loc[i, base_nova] = 1

    return solution

def shake(solution, k, p):
    new_solution = copy.deepcopy(solution)

    if k == 1:
        return taskMove_Equipe(new_solution, p)
    elif k == 2:
        return taskMove_Ativo(new_solution, p)
    elif k == 3:
        return swap_Ativo_Base(new_solution, p)

def GVNS(objective_function, p):
    initial_solution = initialize_solution(p)

    max_num_evaluations = 30
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

            current_solution, k = neighborhood_change(current_solution, new_solution, k)

            history.fitness.append(current_solution.fitness)
            history.penalty.append(current_solution.penalty)
            history.penalized_fitness.append(current_solution.penalized_fitness)

    return current_solution, history

def optimize_instance(objective_function, p, seed):
    np.random.seed(seed)
    best_solution, history = GVNS(objective_function, p)
    return best_solution, history

def optimize(p, objective_function):
    results = []

    num_instances = 5 

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(optimize_instance, objective_function, copy.deepcopy(p), seed=np.random.randint(0, 10000)) for _ in range(num_instances)]
        for future in as_completed(futures):
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

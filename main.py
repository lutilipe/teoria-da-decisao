import numpy as np
import pandas as pd
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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
    result = (p.d.values * solution.x.values @ solution.y.values @ solution.h.values.T).sum()

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
        return copy.deepcopy(new_solution), 1
    return current_solution, k + 1

def local_search_first_improvement(solution, p):
    current_solution = copy.deepcopy(solution)


def GVNS(initial_solution, objective_function, p):
    max_num_evaluations = 2000
    num_evaluations = 0
    k_max = 3
    
    current_solution = initial_solution
    new_solution = initial_solution

    history = Struct()
    history.fit = []
    history.pen = []
    history.fit_pen = []
    
    while num_evaluations < max_num_evaluations:
        k = 1
        while k <= k_max:
            #new_solution = shake(current_solution, k, p)
            #new_solution = local_search_first_improvement(new_solution, p)
            new_solution = objective_function(new_solution, p)
            num_evaluations += 1
            new_solution.id = num_evaluations

            current_solution, k = neighborhood_change(current_solution, new_solution, k)

            history.fit.append(current_solution.fitness)
            history.pen.append(current_solution.penalty)
            history.fit_pen.append(current_solution.penalized_fitness)

            print(num_evaluations)
            
    return current_solution, history

def optimize_instance(objective_function, p):
    initial_solution = initialize_solution(p, True)
    best_solution, history = GVNS(initial_solution, objective_function, p)
    return best_solution, history

def optimize(objective_function):
    p = probdef()
    results = []

    num_instances = 5 

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(optimize_instance, objective_function, p) for _ in range(num_instances)]
        results = [future.result() for future in futures]
    return results

print(optimize(f1))
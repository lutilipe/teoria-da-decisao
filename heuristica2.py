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
"""
Parte 1 do trabalho: seleção de pontos de ônibus (guloso de cobertura ponderada).

Ideia
-----
Cada ponto escolhido "atende" os vértices ao redor, e o atendimento cai com a
distância:

    K(d) = exp(-d / sigma)

A cobertura de um vértice v é a do ponto escolhido mais próximo dele:

    cob[v] = max sobre os pontos escolhidos s de K(dist(s, v))

Cada vértice tem um peso w[v] (a "necessidade" dele, calculada a partir da
densidade e das cores R, G, B). O objetivo é escolher p pontos que maximizem

    cobertura_total = soma_v  w[v] * cob[v]

O algoritmo é guloso: a cada passo, entra o vértice que mais aumenta a
cobertura_total. Um ponto colado a outro já escolhido quase não aumenta nada,
então os pontos se espalham sozinhos, sem nenhuma penalização inventada.

Formato de entrada: o dicionário devolvido por ``gerar_simulacao``.
"""

import math

import numpy as np

CRITERIOS = ("dens", "R", "G", "B")


# --------------------------------------------------------------------------
# 1. Dicionário da simulação -> arrays
# --------------------------------------------------------------------------
def extrair_dados(sim: dict):
    """Converte a simulação em arrays numpy.

    Em C, seria preencher ``double dist[N][N]`` e vetores
    ``double dens[N], R[N], G[N], B[N]``. O dicionário usa nomes ("v_3") e
    aqui cada nome vira um índice inteiro (3).

    Returns:
        nomes: lista de nomes (a posição na lista é o índice do vértice).
        coords: array (N, 2) com as coordenadas.
        dist: matriz (N, N) de distâncias, lida dos pesos das arestas.
        atributos: dict critério -> array (N,).
    """
    nomes = list(sim["vertices"]["coordenadas"].keys())
    indice = {nome: i for i, nome in enumerate(nomes)}
    n = len(nomes)

    coords = np.array([sim["vertices"]["coordenadas"][v] for v in nomes], dtype=float)

    # O grafo é completo, então toda dist[i][j] existe nas arestas.
    # As arestas vêm com (i, j) uma só vez; preenchemos os dois lados.
    dist = np.zeros((n, n))
    for (u, v), peso in sim["arestas"]["pesos"].items():
        dist[indice[u], indice[v]] = peso
        dist[indice[v], indice[u]] = peso

    rotulos = sim["vertices"]["rotulos"]
    cores = sim["vertices"]["cores"]
    atributos = {
        "dens": np.array([rotulos[v] for v in nomes], dtype=float),
        "R": np.array([cores[v][0] for v in nomes], dtype=float),
        "G": np.array([cores[v][1] for v in nomes], dtype=float),
        "B": np.array([cores[v][2] for v in nomes], dtype=float),
    }
    return nomes, coords, dist, atributos


# --------------------------------------------------------------------------
# 2. Necessidade de cada vértice
# --------------------------------------------------------------------------
def _inverte(x):
    """Troca 'maior = menos necessidade' por 'maior = mais necessidade'."""
    return x.max() + x.min() - x


def _normaliza_soma(x):
    """Divide pelo total, para a demanda do critério somar 1."""
    return x / x.sum()


def construir_demandas(atributos, densidade="criterio", alpha=0.5, gamma=1.0, inverter=()):
    """Monta a matriz de demandas D, de formato (m, N): uma linha por critério.

    Cada linha é normalizada para somar 1. Assim os critérios têm o mesmo
    peso total, mesmo estando em escalas diferentes (densidade de 6 a 10,
    cores de 50 a 255).

    Args:
        densidade: como a densidade entra no modelo.
            ``"criterio"``: opção A, a densidade é um quarto critério (m = 4).
            ``"multiplicador"``: a densidade pondera R, G e B por
            ``dens ** alpha`` (m = 3). ``alpha = 0`` ignora a densidade e
            ``alpha = 1`` é o multiplicador pleno.
        alpha: expoente da densidade no modo ``"multiplicador"``.
        gamma: expoente de contraste aplicado antes de normalizar.
            ``gamma > 1`` dá mais peso aos vértices de valores extremos.
        inverter: critérios em que valor baixo significa mais necessidade
            (por exemplo, ``("G",)`` se oportunidade alta = menos necessidade).

    Returns:
        (nomes_dos_criterios, D)
    """
    attr = {c: (_inverte(x) if c in inverter else x) for c, x in atributos.items()}

    if densidade == "criterio":
        nomes = list(CRITERIOS)
        linhas = [_normaliza_soma(attr[c] ** gamma) for c in nomes]
    elif densidade == "multiplicador":
        nomes = ["R", "G", "B"]
        peso = attr["dens"] ** alpha
        linhas = [_normaliza_soma(peso * attr[c] ** gamma) for c in nomes]
    else:
        raise ValueError("densidade deve ser 'criterio' ou 'multiplicador'")

    return nomes, np.vstack(linhas)


def pesos_dos_vertices(D):
    """w[v] = média das demandas normalizadas de v. Soma 1 sobre todos os vértices."""
    return D.mean(axis=0)


def sigma_padrao(coords, p, fator=0.5):
    """Escala de cobertura: fração do espaçamento típico entre p pontos.

    Se p pontos ocupassem a região de forma uniforme, cada um ficaria a
    ``sqrt(area / p)`` do vizinho. O ``sigma`` é uma fração disso.
    """
    largura = max(coords[:, 0].max() - coords[:, 0].min(), 1.0)
    altura = max(coords[:, 1].max() - coords[:, 1].min(), 1.0)
    return fator * math.sqrt(largura * altura / p)


# --------------------------------------------------------------------------
# 3. O guloso
# --------------------------------------------------------------------------
def guloso_cobertura(w, K, p):
    """Escolhe p vértices maximizando a cobertura ponderada, de forma gulosa.

    Args:
        w: array (N,) com o peso (necessidade) de cada vértice.
        K: matriz (N, N), K[u][v] = exp(-dist(u, v) / sigma).
        p: quantidade de pontos.

    Returns:
        Lista com os índices dos p vértices escolhidos, na ordem em que entraram.
    """
    n = len(w)
    cob = np.zeros(n)                    # cob[v]: cobertura atual de cada vértice
    livre = np.ones(n, dtype=bool)       # ainda não escolhido
    escolhidos = []

    for _ in range(p):
        # ganho[u] = quanto a cobertura_total sobe se u entrar. Em C, seria:
        #   for u in 0..N-1:
        #       ganho[u] = 0
        #       for v in 0..N-1:
        #           melhora = K[u][v] - cob[v]
        #           if (melhora > 0) ganho[u] += w[v] * melhora
        # O numpy faz os dois loops de uma vez.
        ganho = np.maximum(K - cob[None, :], 0.0) @ w
        ganho[~livre] = -np.inf          # não repete vértice

        u = int(np.argmax(ganho))        # busca do maior, como em C
        escolhidos.append(u)
        livre[u] = False
        cob = np.maximum(cob, K[u])      # cob[v] = max(cob[v], K[u][v])

    return escolhidos


# --------------------------------------------------------------------------
# 4. Interface pública e avaliação
# --------------------------------------------------------------------------
def avaliar(sim, escolhidos, sigma, gamma=1.0, inverter=()):
    """Métricas de uma seleção, sempre com os quatro critérios puros.

    ``F[c]`` é a fração da demanda do critério c que está sendo atendida
    (entre 0 e 1). ``raio`` é a maior distância de um vértice qualquer até o
    ponto escolhido mais próximo. ``gamma`` e ``inverter`` devem ser os mesmos
    usados na seleção.
    """
    _, _, dist, atributos = extrair_dados(sim)
    nomes_c, D = construir_demandas(atributos, "criterio", gamma=gamma, inverter=inverter)
    K = np.exp(-dist / sigma)
    cob = K[list(escolhidos)].max(axis=0)
    F = D @ cob
    raio = float(dist[:, list(escolhidos)].min(axis=1).max())
    return {
        "F": {c: float(f) for c, f in zip(nomes_c, F)},
        "min": float(F.min()),
        "media": float(F.mean()),
        "raio": raio,
    }


def _indice_do_nome(nome):
    """'v_12' -> 12. Usado para manter a ordem dos vértices da simulação."""
    return int(nome.split("_")[1])


def extrair_subgrafo(sim, pontos):
    """Monta o grafo formado só pelos pontos escolhidos.

    Devolve um dicionário no MESMO formato do gerador, então qualquer função
    que lê uma simulação (inclusive o plot do gerador) funciona nele:

        {"vertices": {"coordenadas": ..., "rotulos": ..., "cores": ...},
         "arestas":  {"pesos": {(v_i, v_j): distância}}}

    Os vértices mantêm os nomes originais (ex.: 'v_60'), para dar para voltar
    ao mapa completo. As arestas seguem a convenção do gerador: cada par
    aparece uma vez, com o vértice de menor índice primeiro. Como o grafo
    original é completo, o subgrafo também é: p vértices, p*(p-1)/2 arestas.

    Args:
        sim: dicionário devolvido por ``gerar_simulacao``.
        pontos: nomes dos vértices escolhidos, em qualquer ordem.
    """
    nomes = sorted(pontos, key=_indice_do_nome)   # ordem da simulação original
    pesos = sim["arestas"]["pesos"]

    coordenadas = {v: sim["vertices"]["coordenadas"][v] for v in nomes}
    rotulos = {v: sim["vertices"]["rotulos"][v] for v in nomes}
    cores = {v: sim["vertices"]["cores"][v] for v in nomes}

    # Em C: dois laços aninhados com j > i, copiando dist[i][j].
    arestas = {}
    for i, u in enumerate(nomes):
        for v in nomes[i + 1:]:
            arestas[(u, v)] = pesos[(u, v)]

    return {
        "vertices": {"coordenadas": coordenadas, "rotulos": rotulos, "cores": cores},
        "arestas": {"pesos": arestas},
    }


def selecionar_pontos(sim, p, densidade="criterio", alpha=0.5, gamma=1.0,
                      sigma=None, fator_sigma=0.5, inverter=()):
    """Seleciona p pontos de ônibus a partir da simulação.

    Args:
        sim: dicionário devolvido por ``gerar_simulacao``.
        p: quantidade de pontos a escolher.
        densidade, alpha, gamma, inverter: ver ``construir_demandas``.
        sigma: escala de cobertura. Se None, usa ``sigma_padrao``.
        fator_sigma: fração do espaçamento típico usada em ``sigma_padrao``.

    Returns:
        dict com ``pontos`` (nomes dos vértices, na ordem de escolha),
        ``indices``, ``sigma``, as métricas de ``avaliar`` e ``subgrafo``: o
        grafo dos pontos escolhidos (coordenadas, rótulos, cores e arestas com
        pesos), no mesmo formato do gerador. É a entrada da Parte 2.
    """
    nomes, coords, dist, atributos = extrair_dados(sim)
    if not 1 <= p <= len(nomes):
        raise ValueError("p deve estar entre 1 e o número de vértices")

    if sigma is None:
        sigma = sigma_padrao(coords, p, fator_sigma)

    _, D = construir_demandas(atributos, densidade, alpha, gamma, inverter)
    w = pesos_dos_vertices(D)
    K = np.exp(-dist / sigma)

    idx = guloso_cobertura(w, K, p)

    pontos = [nomes[i] for i in idx]
    resultado = {"pontos": pontos, "indices": idx, "sigma": sigma}
    resultado.update(avaliar(sim, idx, sigma, gamma, inverter))
    resultado["subgrafo"] = extrair_subgrafo(sim, pontos)
    return resultado


if __name__ == "__main__":
    from gerador_de_simulacao import gerar_simulacao

    sim = gerar_simulacao(80, 20, seed=1, usar_seed=True, n_vertices_aleatorios=10)
    r = selecionar_pontos(sim, p=10)
    print("pontos escolhidos:", r["pontos"])
    print("cobertura por critério:", {c: round(v, 3) for c, v in r["F"].items()})
    print(f"pior critério = {r['min']:.3f}   raio de cobertura = {r['raio']:.1f}")

    sub = r["subgrafo"]
    print("coordenadas dos pontos:", sub["vertices"]["coordenadas"])
    print("número de arestas:", len(sub["arestas"]["pesos"]))
"""
Parte 3 do trabalho: geração das linhas de ônibus.

Entrada: o subgrafo devolvido por ``selecionar_pontos`` (chave ``"subgrafo"``),
no mesmo formato do gerador:

    {"vertices": {"coordenadas": {nome: (x, y)}, ...},
     "arestas":  {"pesos": {(v_i, v_j): distância}}}

O módulo só lê as coordenadas e os pesos das arestas, então mudanças internas
na seleção de pontos não o afetam, desde que o formato se mantenha.

Ideia
-----
Uma linha é um caminho simples (sem vértice repetido) e o conjunto de linhas
deve cobrir todos os pontos, com cada linha encostando em alguma outra.

1. MST (Prim): o esqueleto mais barato que liga todos os pontos. Nenhuma rede
   conexa custa menos que ele.
2. Se L >= m (m = vértices de grau ímpar da MST / 2): emparelhamento. Em cada
   vértice, as arestas são juntadas aos pares ("entra por uma, sai pela
   outra"), e a árvore se decompõe em m caminhos sem repetir aresta. O custo
   total é exatamente o da MST. Para chegar a L, divide-se a linha mais pesada.
3. Se L < m: lista e corte. Percorre-se a MST em DFS (pré-ordem), o que dá uma
   ordem dos pontos; cada par consecutivo vira uma aresta direta (atalho, pois
   o grafo é completo). A lista é cortada em L pedaços consecutivos, e cada
   pedaço termina onde o próximo começa. Custo <= 2 x MST (desigualdade
   triangular).
"""

import random

import numpy as np

# Os pesos do gerador são distâncias euclidianas arredondadas a 2 casas. Cada
# peso erra no máximo 0.005, então a desigualdade triangular pode falhar por no
# máximo 3 * 0.005 = 0.015 só por causa do arredondamento.
TOLERANCIA_TRIANGULAR = 0.015 + 1e-9


# --------------------------------------------------------------------------
# 1. Entrada
# --------------------------------------------------------------------------
def extrair_matriz(sub):
    """Subgrafo -> (nomes, matriz de distâncias simétrica).

    A posição de cada nome na lista é o índice do vértice. As arestas vêm uma
    vez por par, então os dois lados da matriz são preenchidos aqui.
    """
    nomes = list(sub["vertices"]["coordenadas"].keys())
    indice = {nome: i for i, nome in enumerate(nomes)}
    n = len(nomes)
    dist = np.zeros((n, n))
    for (u, v), peso in sub["arestas"]["pesos"].items():
        dist[indice[u], indice[v]] = peso
        dist[indice[v], indice[u]] = peso
    return nomes, dist


def validar(sub, tolerancia=TOLERANCIA_TRIANGULAR):
    """Confere a entrada.

    Returns:
        dict com ``problemas`` (lista de textos; vazia = entrada aceitável) e
        ``excesso_triangular`` (a pior violação de d(i,k) <= d(i,j) + d(j,k)).
    """
    nomes, dist = extrair_matriz(sub)
    n = len(nomes)
    pesos = sub["arestas"]["pesos"]
    problemas = []

    if n < 2:
        problemas.append("é preciso ter pelo menos 2 pontos")

    esperado = n * (n - 1) // 2
    pares = {frozenset(par) for par in pesos}
    if len(pesos) != esperado or len(pares) != esperado:
        problemas.append(f"grafo não é completo: {len(pares)} pares distintos de {esperado}")
    if any(peso <= 0 for peso in pesos.values()):
        problemas.append("há peso não positivo")

    # exc[i, j, k] = d(i,k) - d(i,j) - d(j,k). Em C: três laços aninhados.
    excesso = float((dist[:, None, :] - dist[:, :, None] - dist[None, :, :]).max())
    if excesso > tolerancia:
        problemas.append(f"desigualdade triangular violada por {excesso:.4f}")

    return {"problemas": problemas, "excesso_triangular": excesso}


# --------------------------------------------------------------------------
# 2. MST (Prim em O(V^2): o grafo é completo, então a versão com matriz
#    ganha da versão com heap, que custaria O(V^2 log V))
# --------------------------------------------------------------------------
def prim(dist, raiz=0):
    """Devolve as arestas da MST como lista de pares (pai, filho) de índices."""
    n = len(dist)
    na_arvore = np.zeros(n, dtype=bool)
    na_arvore[raiz] = True
    custo = dist[raiz].copy()          # custo[v]: aresta mais barata de v até a árvore
    pai = np.full(n, raiz)             # pai[v]: ponta dessa aresta, dentro da árvore
    arestas = []
    for _ in range(n - 1):
        candidatos = np.where(na_arvore, np.inf, custo)
        v = int(np.argmin(candidatos))
        arestas.append((int(pai[v]), v))
        na_arvore[v] = True
        melhor = dist[v] < custo
        custo = np.where(melhor, dist[v], custo)
        pai = np.where(melhor, v, pai)
    return arestas


def adjacencia(arestas, n):
    """Lista de vizinhos de cada vértice."""
    adj = [[] for _ in range(n)]
    for a, b in arestas:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def minimo_de_linhas(adj):
    """Mínimo de caminhos que cobrem todas as arestas de uma árvore.

    Cada caminho tem duas pontas e todo vértice de grau ímpar precisa ser ponta
    de algum caminho; o emparelhamento atinge esse mínimo.
    """
    impares = sum(1 for vizinhos in adj if len(vizinhos) % 2 == 1)
    return impares // 2


def custo_linha(linha, dist):
    """Soma dos pesos dos pares consecutivos da linha."""
    return float(sum(dist[a, b] for a, b in zip(linha, linha[1:])))


# --------------------------------------------------------------------------
# 3. Emparelhamento (L >= m)
# --------------------------------------------------------------------------
def decompor_em_caminhos(adj, rng):
    """Emparelha as arestas em cada vértice (ao acaso) e segue os pares.

    Em um vértice de grau ímpar sobra uma aresta sem par: a linha termina ali.
    Numa árvore, andar sem voltar pela aresta de chegada nunca repete vértice,
    então cada linha é um caminho simples.
    """
    n = len(adj)
    # parceira[v][u] = w: em v, a linha que chega por (v,u) segue por (v,w).
    # None: a linha termina em v.
    parceira = [dict() for _ in range(n)]
    for v in range(n):
        vizinhos = adj[v][:]
        rng.shuffle(vizinhos)
        while len(vizinhos) >= 2:
            a, b = vizinhos.pop(), vizinhos.pop()
            parceira[v][a] = b
            parceira[v][b] = a
        if vizinhos:
            parceira[v][vizinhos[0]] = None

    visitadas = set()
    linhas = []
    for v in range(n):
        for u in adj[v]:
            if parceira[v][u] is not None or frozenset((v, u)) in visitadas:
                continue
            linha = [v]
            atual, proximo = v, u
            while True:
                visitadas.add(frozenset((atual, proximo)))
                linha.append(proximo)
                seguinte = parceira[proximo][atual]
                if seguinte is None:
                    break
                atual, proximo = proximo, seguinte
            linhas.append(linha)
    return linhas


def _melhor_corte_de_uma_linha(linha, dist):
    """Posição k que divide a linha em linha[:k+1] e linha[k:] minimizando o maior custo."""
    melhor_k, melhor_valor = None, float("inf")
    for k in range(1, len(linha) - 1):
        valor = max(custo_linha(linha[:k + 1], dist), custo_linha(linha[k:], dist))
        if valor < melhor_valor:
            melhor_k, melhor_valor = k, valor
    return melhor_k


def dividir_ate(linhas, dist, L):
    """Divide a linha mais pesada (com >= 2 arestas) até haver L linhas.

    Dividir não custa nada: as duas metades compartilham o ponto de corte.
    """
    linhas = [linha[:] for linha in linhas]
    while len(linhas) < L:
        divisiveis = [i for i, linha in enumerate(linhas) if len(linha) >= 3]
        if not divisiveis:
            raise ValueError("L maior que o número de arestas da árvore")
        i = max(divisiveis, key=lambda j: custo_linha(linhas[j], dist))
        k = _melhor_corte_de_uma_linha(linhas[i], dist)
        linha = linhas.pop(i)
        linhas.extend([linha[:k + 1], linha[k:]])
    return linhas


def melhor_emparelhamento(adj, dist, L, rng, tentativas=300):
    """Sorteia várias formas de emparelhar e fica com a mais equilibrada.

    Critério: menor custo da linha mais pesada; empate: menor soma dos
    quadrados dos custos. O custo total não muda (é sempre o da MST).

    Devolve (linhas, base): ``base`` são as linhas do emparelhamento antes de
    dividir até L.
    """
    melhor, melhor_base, melhor_chave = None, None, None
    for _ in range(tentativas):
        base = decompor_em_caminhos(adj, rng)
        linhas = dividir_ate(base, dist, L)
        custos = [custo_linha(linha, dist) for linha in linhas]
        chave = (round(max(custos), 9), round(sum(c * c for c in custos), 9))
        if melhor_chave is None or chave < melhor_chave:
            melhor, melhor_base, melhor_chave = linhas, base, chave
    return melhor, melhor_base


# --------------------------------------------------------------------------
# 4. Lista e corte (L < m)
# --------------------------------------------------------------------------
def ordem_dfs(adj, inicio, dist, modo, rng):
    """Pré-ordem da DFS na árvore: a ordem em que o 'dedo' toca cada ponto.

    modo: ``"perto"`` visita primeiro o filho mais próximo, ``"longe"`` o mais
    distante e ``"acaso"`` sorteia.
    """
    visitado = [False] * len(adj)
    ordem = []
    pilha = [inicio]
    while pilha:
        v = pilha.pop()
        if visitado[v]:
            continue
        visitado[v] = True
        ordem.append(v)
        filhos = [u for u in adj[v] if not visitado[u]]
        if modo == "perto":
            filhos.sort(key=lambda u: dist[v, u])
        elif modo == "longe":
            filhos.sort(key=lambda u: -dist[v, u])
        else:
            rng.shuffle(filhos)
        pilha.extend(reversed(filhos))     # o primeiro a visitar fica no topo
    return ordem


def melhorar_2opt(ordem, dist):
    """Melhora um caminho aberto trocando duas arestas por duas, enquanto cair o custo.

    Inverter o trecho ordem[i..j] troca as arestas (i-1,i) e (j,j+1) por
    (i-1,j) e (i,j+1). Cada troca estritamente melhor reduz o custo, e há
    um número finito de ordens, então o laço termina.
    """
    ordem = ordem[:]
    n = len(ordem)
    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                antes = depois = 0.0
                if i > 0:
                    antes += dist[ordem[i - 1], ordem[i]]
                    depois += dist[ordem[i - 1], ordem[j]]
                if j < n - 1:
                    antes += dist[ordem[j], ordem[j + 1]]
                    depois += dist[ordem[i], ordem[j + 1]]
                if depois < antes - 1e-9:
                    ordem[i:j + 1] = ordem[i:j + 1][::-1]
                    melhorou = True
    return ordem


def custo_da_ordem(ordem, dist):
    return custo_linha(ordem, dist)


def melhor_ordem(adj, dist, rng, tentativas=30, usar_2opt=True):
    """DFS a partir de cada folha, com várias ordens de filhos; fica a mais barata.

    Devolve (ordem da DFS, ordem após o 2-opt).
    """
    folhas = [v for v, vizinhos in enumerate(adj) if len(vizinhos) == 1]
    melhor, melhor_custo = None, float("inf")
    for inicio in folhas:
        modos = ["perto", "longe"] + ["acaso"] * tentativas
        for modo in modos:
            ordem = ordem_dfs(adj, inicio, dist, modo, rng)
            custo = custo_da_ordem(ordem, dist)
            if custo < melhor_custo:
                melhor, melhor_custo = ordem, custo
    final = melhorar_2opt(melhor, dist) if usar_2opt else melhor
    return melhor, final


def _grupos_gulosos(pesos, B):
    """Corte guloso: fecha o grupo quando a próxima aresta passaria de B.

    Devolve intervalos [a, b) de índices de arestas.
    """
    grupos = []
    inicio, acumulado = 0, 0.0
    for k, w in enumerate(pesos):
        # a tolerância evita que o erro de ponto flutuante (sum() e a soma em
        # sequência diferem na última casa) feche um grupo à toa quando B = soma
        if acumulado + w > B + 1e-9 and k > inicio:
            grupos.append((inicio, k))
            inicio, acumulado = k, 0.0
        acumulado += w
    grupos.append((inicio, len(pesos)))
    return grupos


def menor_limite(pesos, L):
    """Menor B para o qual o corte guloso usa no máximo L grupos (busca binária)."""
    baixo, alto = max(pesos), sum(pesos)
    for _ in range(100):
        meio = (baixo + alto) / 2
        if len(_grupos_gulosos(pesos, meio)) <= L:
            alto = meio
        else:
            baixo = meio
    return alto


def _equalizar(pesos, grupos):
    """Desloca as divisas entre grupos vizinhos para igualar os custos.

    Move uma aresta de um grupo para o vizinho se isso reduz o maior dos dois.
    A soma dos quadrados dos custos cai a cada movimento, então termina, e o
    maior custo global nunca aumenta.
    """
    grupos = [list(g) for g in grupos]
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(grupos) - 1):
            a = grupos[i]
            b = grupos[i + 1]
            custo_a = sum(pesos[a[0]:a[1]])
            custo_b = sum(pesos[b[0]:b[1]])
            atual = max(custo_a, custo_b)
            # passar a última aresta de a para b
            if a[1] - a[0] >= 2:
                w = pesos[a[1] - 1]
                if max(custo_a - w, custo_b + w) < atual - 1e-9:
                    a[1] -= 1
                    b[0] -= 1
                    mudou = True
                    continue
            # passar a primeira aresta de b para a
            if b[1] - b[0] >= 2:
                w = pesos[b[0]]
                if max(custo_a + w, custo_b - w) < atual - 1e-9:
                    a[1] += 1
                    b[0] += 1
                    mudou = True
    return [tuple(g) for g in grupos]


def cortar_ordem(ordem, dist, L, equalizar=True):
    """Corta a lista em L linhas consecutivas; cada uma começa onde a anterior terminou."""
    pesos = [float(dist[a, b]) for a, b in zip(ordem, ordem[1:])]
    if L > len(pesos):
        raise ValueError("L maior que o número de arestas da lista")

    grupos = _grupos_gulosos(pesos, menor_limite(pesos, L))
    # o guloso pode devolver menos de L grupos: divide o mais pesado
    while len(grupos) < L:
        divisiveis = [i for i, (a, b) in enumerate(grupos) if b - a >= 2]
        i = max(divisiveis, key=lambda j: sum(pesos[grupos[j][0]:grupos[j][1]]))
        a, b = grupos.pop(i)
        k = min(range(a + 1, b),
                key=lambda c: max(sum(pesos[a:c]), sum(pesos[c:b])))
        grupos.insert(i, (k, b))
        grupos.insert(i, (a, k))

    if equalizar:
        grupos = _equalizar(pesos, grupos)
    return [ordem[a:b + 1] for a, b in grupos]


# --------------------------------------------------------------------------
# 5. Verificação, métricas e interface pública
# --------------------------------------------------------------------------
def verificar(linhas, n, L):
    """Confere as exigências; levanta RuntimeError se alguma falhar."""
    if len(linhas) != L:
        raise RuntimeError(f"saíram {len(linhas)} linhas, o pedido era {L}")
    cobertos = set()
    for linha in linhas:
        if len(set(linha)) != len(linha):
            raise RuntimeError("linha com vértice repetido")
        if len(linha) < 2:
            raise RuntimeError("linha sem aresta")
        cobertos |= set(linha)
    if len(cobertos) != n:
        raise RuntimeError("há ponto não atendido")
    if L > 1:
        # o grafo "linhas que se encostam" precisa ser conexo
        alcancadas, fronteira = {0}, [0]
        while fronteira:
            i = fronteira.pop()
            for j in range(L):
                if j not in alcancadas and set(linhas[i]) & set(linhas[j]):
                    alcancadas.add(j)
                    fronteira.append(j)
        if len(alcancadas) != L:
            raise RuntimeError("a rede de linhas não é conexa")


def metricas(linhas, dist, custo_mst):
    custos = [custo_linha(linha, dist) for linha in linhas]
    total = sum(custos)
    L = len(linhas)
    return {
        "custos": custos,
        "total": total,
        "razao_total_mst": total / custo_mst,
        "maior": max(custos),
        "menor": min(custos),
        # 1.0 = perfeito. Para linhas sem aresta em comum, nenhuma solução
        # tem desequilibrio < 1.
        "desequilibrio": max(custos) / (total / L),
    }


def gerar_linhas(sub, L, metodo="auto", semente=0, tentativas=300, usar_2opt=True):
    """Gera L linhas de ônibus sobre o subgrafo dos pontos selecionados.

    Args:
        sub: subgrafo devolvido por ``selecionar_pontos``.
        L: número de linhas desejado.
        metodo: ``"auto"`` usa emparelhamento se L >= m e lista e corte se não;
            ``"emparelhamento"`` ou ``"lista"`` forçam um dos dois.
        semente: semente do sorteio, para reproduzir resultados.
        tentativas: quantos sorteios fazer (emparelhamentos; ordens da DFS).
        usar_2opt: melhora a lista com 2-opt antes de cortar.

    Returns:
        dict com ``linhas`` (listas de nomes de vértices), ``metodo``, ``m``,
        ``custo_mst``, ``metricas`` e ``etapas`` (os dados intermediários, que
        ``plotar_etapas`` desenha: ``arvore`` e, conforme o método,
        ``emparelhamento`` ou ``ordem_dfs`` e ``ordem_2opt``).
    """
    achados = validar(sub)
    if achados["problemas"]:
        raise ValueError("entrada inválida: " + "; ".join(achados["problemas"]))

    nomes, dist = extrair_matriz(sub)
    n = len(nomes)
    if not 1 <= L <= n - 1:
        raise ValueError(f"L deve estar entre 1 e {n - 1}")

    rng = random.Random(semente)
    arvore = prim(dist)
    custo_mst = float(sum(dist[a, b] for a, b in arvore))
    adj = adjacencia(arvore, n)
    m = minimo_de_linhas(adj)

    if metodo == "auto":
        metodo = "emparelhamento" if L >= m else "lista"
    etapas = {"arvore": [(nomes[a], nomes[b]) for a, b in arvore]}
    if metodo == "emparelhamento":
        if L < m:
            raise ValueError(f"emparelhamento exige L >= {m}")
        linhas, base = melhor_emparelhamento(adj, dist, L, rng, tentativas)
        etapas["emparelhamento"] = [[nomes[i] for i in linha] for linha in base]
    elif metodo == "lista":
        ordem_inicial, ordem = melhor_ordem(adj, dist, rng, max(1, tentativas // 10), usar_2opt)
        linhas = cortar_ordem(ordem, dist, L)
        etapas["ordem_dfs"] = [nomes[i] for i in ordem_inicial]
        etapas["ordem_2opt"] = [nomes[i] for i in ordem]
    else:
        raise ValueError("metodo deve ser 'auto', 'emparelhamento' ou 'lista'")

    verificar(linhas, n, L)
    return {
        "linhas": [[nomes[i] for i in linha] for linha in linhas],
        "metodo": metodo,
        "m": m,
        "custo_mst": custo_mst,
        "metricas": metricas(linhas, dist, custo_mst),
        "etapas": etapas,
    }


def plotar_linhas(sub, resultado, arquivo=None):
    """Desenha as linhas sobre os pontos. Sem ``arquivo``, abre a janela."""
    import matplotlib.pyplot as plt

    coords = sub["vertices"]["coordenadas"]
    _, eixo = plt.subplots(figsize=(8, 8))
    paleta = plt.get_cmap("tab10")
    for i, linha in enumerate(resultado["linhas"]):
        xs = [coords[v][0] for v in linha]
        ys = [coords[v][1] for v in linha]
        eixo.plot(xs, ys, "-", color=paleta(i % 10), linewidth=2.5, alpha=0.85,
                  label=f"linha {i + 1}  (custo {resultado['metricas']['custos'][i]:.1f})",
                  zorder=1)
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    eixo.scatter(xs, ys, c="white", s=260, edgecolors="black", zorder=2)
    for nome, (x, y) in coords.items():
        eixo.annotate(nome.split("_")[1], (x, y), ha="center", va="center",
                      fontsize=7, fontweight="bold", zorder=3)
    eixo.set_title(f"Linhas de ônibus ({resultado['metodo']}, L = {len(resultado['linhas'])})")
    eixo.set_aspect("equal")
    eixo.grid(True, linestyle="--", alpha=0.4)
    eixo.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    if arquivo:
        plt.savefig(arquivo, dpi=110)
        plt.close()
    else:
        plt.show()


def _pontos(eixo, coords):
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    eixo.scatter(xs, ys, c="white", s=170, edgecolors="black", zorder=3)
    for nome, (x, y) in coords.items():
        eixo.annotate(nome.split("_")[1], (x, y), ha="center", va="center",
                      fontsize=6, fontweight="bold", zorder=4)
    eixo.set_aspect("equal")
    eixo.grid(True, linestyle="--", alpha=0.3)
    eixo.set_xticks([])
    eixo.set_yticks([])


def _tracar(eixo, coords, caminho, cor, largura=2.2):
    xs = [coords[v][0] for v in caminho]
    ys = [coords[v][1] for v in caminho]
    eixo.plot(xs, ys, "-", color=cor, linewidth=largura, alpha=0.9, zorder=2)


def plotar_etapas(sub, resultado, arquivo=None):
    """Um painel por etapa do algoritmo, todos sobre os mesmos pontos.

    Cada etapa é só um conjunto de segmentos; o que muda de um painel para o
    outro é a lista de caminhos entregue a ``_tracar``. Os dados vêm de
    ``resultado["etapas"]``. Sem ``arquivo``, abre a janela.
    """
    import math
    import matplotlib.pyplot as plt

    coords = sub["vertices"]["coordenadas"]
    pesos = sub["arestas"]["pesos"]
    et = resultado["etapas"]
    L = len(resultado["linhas"])
    paleta = plt.get_cmap("tab10")

    def peso(u, v):
        return pesos[(u, v)] if (u, v) in pesos else pesos[(v, u)]

    def custo(caminho):
        return sum(peso(a, b) for a, b in zip(caminho, caminho[1:]))

    def colorido(linhas):
        def desenhar(eixo):
            for i, linha in enumerate(linhas):
                _tracar(eixo, coords, linha, paleta(i % 10), 3.0)
        return desenhar

    def numerado(ordem):
        def desenhar(eixo):
            _tracar(eixo, coords, ordem, "tab:blue", 2.0)
            for i, v in enumerate(ordem):
                x, y = coords[v]
                eixo.annotate(str(i + 1), (x, y), xytext=(6, 6), textcoords="offset points",
                              fontsize=7, color="tab:red", zorder=5)
        return desenhar

    paineis = [
        (f"1. Subgrafo completo ({len(pesos)} arestas)",
         lambda eixo: [_tracar(eixo, coords, [u, v], "lightgray", 0.6) for u, v in pesos]),
        (f"2. MST (Prim), custo {resultado['custo_mst']:.1f}",
         lambda eixo: [_tracar(eixo, coords, [u, v], "black", 2.0) for u, v in et["arvore"]]),
    ]
    if resultado["metodo"] == "emparelhamento":
        base = et["emparelhamento"]
        paineis.append((f"3. Emparelhamento: {len(base)} linhas (custo = MST)", colorido(base)))
        paineis.append((f"4. Dividido até L = {L}", colorido(resultado["linhas"])))
    else:
        paineis.append((f"3. Lista da DFS com atalhos, custo {custo(et['ordem_dfs']):.1f}",
                        numerado(et["ordem_dfs"])))
        paineis.append((f"4. Após 2-opt, custo {custo(et['ordem_2opt']):.1f}",
                        numerado(et["ordem_2opt"])))
        paineis.append((f"5. Cortada em L = {L} linhas", colorido(resultado["linhas"])))

    colunas = min(3, len(paineis))
    fileiras = math.ceil(len(paineis) / colunas)
    _, eixos = plt.subplots(fileiras, colunas, figsize=(5.2 * colunas, 5.2 * fileiras), squeeze=False)
    eixos = eixos.ravel()
    for eixo, (titulo, desenhar) in zip(eixos, paineis):
        desenhar(eixo)
        _pontos(eixo, coords)
        eixo.set_title(titulo, fontsize=10)
    for eixo in eixos[len(paineis):]:
        eixo.axis("off")
    plt.tight_layout()
    if arquivo:
        plt.savefig(arquivo, dpi=100)
        plt.close()
    else:
        plt.show()


def plotar_selecao(sim, pontos, arquivo=None):
    """Antes e depois da seleção de pontos.

    À esquerda, o mapa original: todos os candidatos, com a cor e o rótulo de
    densidade que o gerador atribuiu, e os escolhidos com anel vermelho. À
    direita, só os escolhidos, que são os vértices que esta parte recebe.

    Args:
        sim: dicionário devolvido por ``gerar_simulacao``.
        pontos: nomes dos vértices escolhidos (``selecionar_pontos(...)["pontos"]``).
        arquivo: se dado, salva a figura; senão, abre a janela.
    """
    import matplotlib.pyplot as plt

    coords = sim["vertices"]["coordenadas"]
    rotulos = sim["vertices"]["rotulos"]
    cores = sim["vertices"]["cores"]
    escolhidos = set(pontos)

    xs_todos = [p[0] for p in coords.values()]
    ys_todos = [p[1] for p in coords.values()]
    limites = ((min(xs_todos) - 1, max(xs_todos) + 1), (min(ys_todos) - 1, max(ys_todos) + 1))

    _, eixos = plt.subplots(1, 2, figsize=(15, 7.5))
    paineis = (
        (eixos[0], list(coords), True, f"Mapa original: {len(coords)} pontos candidatos"),
        (eixos[1], [v for v in coords if v in escolhidos], False,
         f"Pontos selecionados: {len(escolhidos)}"),
    )
    for eixo, nomes, destacar, titulo in paineis:
        xs = [coords[v][0] for v in nomes]
        ys = [coords[v][1] for v in nomes]
        rgb = [tuple(canal / 255 for canal in cores[v]) for v in nomes]
        borda = ["red" if destacar and v in escolhidos else "black" for v in nomes]
        largura = [3.0 if destacar and v in escolhidos else 0.8 for v in nomes]
        eixo.scatter(xs, ys, c=rgb, s=260, edgecolors=borda, linewidths=largura, zorder=2)
        for v in nomes:
            x, y = coords[v]
            eixo.annotate(str(rotulos[v]), (x, y), ha="center", va="center",
                          fontsize=6, fontweight="bold", zorder=3)
        eixo.set_xlim(*limites[0])
        eixo.set_ylim(*limites[1])
        eixo.set_aspect("equal")
        eixo.grid(True, linestyle="--", alpha=0.4)
        eixo.set_title(titulo)
    plt.tight_layout()
    if arquivo:
        plt.savefig(arquivo, dpi=100)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from gerador_de_simulacao import gerar_simulacao
    from selecao_pontos import selecionar_pontos

    sim = gerar_simulacao(80, 20, seed=1, usar_seed=True, n_vertices_aleatorios=10)
    selecao = selecionar_pontos(sim, p=10)
    sub = selecao["subgrafo"]
    plotar_selecao(sim, selecao["pontos"], "selecao.png")
    for L in (1, 2, 3, 4, 6):
        r = gerar_linhas(sub, L)
        mt = r["metricas"]
        print(f"L={L}  método={r['metodo']:<14} m={r['m']}  total/MST={mt['razao_total_mst']:.3f}  "
              f"desequilíbrio={mt['desequilibrio']:.3f}  custos={[round(c, 1) for c in mt['custos']]}")
    plotar_linhas(sub, gerar_linhas(sub, 3), "linhas.png")
    plotar_etapas(sub, gerar_linhas(sub, 3), "etapas_emparelhamento.png")
    plotar_etapas(sub, gerar_linhas(sub, 1), "etapas_lista.png")

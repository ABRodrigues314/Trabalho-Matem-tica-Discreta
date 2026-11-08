"""
Simulador de cidade: definindo pontos como possíveis pontos de ônibus
"""

import random
import matplotlib.pyplot as plt

def _gerar_mapa(numero_de_possiveis_pontos: int, tamanho_do_mapa: int) -> list:
    """Gera a matriz que representa o mapa dos possíveis pontos de ônibus.

    Cada célula da matriz vale 1 quando representa um possível ponto de ônibus
    e 0 quando não representa.

    Args:
        numero_de_possiveis_pontos: Quantidade de pontos de ônibus a
            distribuir no mapa.
        tamanho_do_mapa: Lado da matriz quadrada que representa o mapa.

    Returns:
        Uma matriz quadrada (lista de listas) de tamanho ``tamanho_do_mapa``
        com ``numero_de_possiveis_pontos`` células iguais a 1.
    """
    matriz = []
    for linha in range(tamanho_do_mapa):
        linha_ = []
        for coluna in range(tamanho_do_mapa):
            linha_.append(0)
        matriz.append(linha_[:])

    numero_de_pontos = 0
    while numero_de_pontos != numero_de_possiveis_pontos:
        posicao_x = random.randint(0, tamanho_do_mapa - 1)
        posicao_y = random.randint(0, tamanho_do_mapa - 1)
        if matriz[posicao_y][posicao_x] == 0:
            matriz[posicao_y][posicao_x] = 1
            numero_de_pontos += 1
        else:
            pass

    return matriz


def _set_posicao(matriz: list) -> tuple:
    """Extrai as coordenadas de todos os pontos de ônibus do mapa.

    Args:
        matriz: Matriz do mapa, na qual o valor 1 indica um ponto de ônibus.

    Returns:
        Uma tupla com as coordenadas ``(x, y)`` de cada ponto de ônibus.
    """

    posicao_y = 0
    lista_de_pontos = []

    for linha in matriz:
        posicao_x = 0
        for ponto in linha:
            if ponto == 1:
                lista_de_pontos.append((posicao_x, posicao_y))
            else:
                pass
            posicao_x += 1
        posicao_y += 1

    return tuple(lista_de_pontos)


def _nomear_vertices(lista_de_pontos: tuple) -> dict:
    """Atribui um nome a cada vértice a partir de suas coordenadas.

    Args:
        lista_de_pontos: Tupla com as coordenadas ``(x, y)`` dos pontos.

    Returns:
        Um dicionário que mapeia ``"v_<índice>"`` para a coordenada
        correspondente.
    """
    numero_vertice = 0
    vertices = dict()
    for posicao in lista_de_pontos:
        vertices[f"v_{numero_vertice}"] = posicao
        numero_vertice += 1

    return vertices


def _set_arestas(vertices: dict) -> dict:
    """Calcula todas as arestas do grafo completo e seus pesos.

    O peso de cada aresta é a distância euclidiana entre os dois vértices que
    ela conecta.

    Args:
        vertices: Dicionário que mapeia o nome de cada vértice à sua
            coordenada ``(x, y)``.

    Returns:
        Um dicionário que mapeia cada par de vértices ``(v_i, v_j)`` ao peso da
        aresta que os conecta.
    """
    lista_de_arestas = {}
    vertice_fixo = 0
    while vertice_fixo != len(vertices):
        vertice_corrido = vertice_fixo + 1
        for n_vertices in range(len(vertices) - vertice_fixo - 1):
            lista_de_arestas[(f"v_{vertice_fixo}", f"v_{vertice_corrido}")] = round(((
                vertices[f"v_{vertice_fixo}"][0] - vertices[f"v_{vertice_corrido}"][0]
                ) ** 2 + (
                vertices[f"v_{vertice_fixo}"][1] - vertices[f"v_{vertice_corrido}"][1]
                ) ** 2) ** 0.5, 2)

            vertice_corrido += 1

        vertice_fixo += 1
    return lista_de_arestas


def _set_rotulos(dict_de_vertices: dict, dict_de_arestas: dict, n_vertices_aleatorios: int, limite_inferior: int, limite_superior: int,  variacao=1) -> dict:
    """Define os rótulos dos vértices usando interpolação.

    Alguns vértices recebem rótulos aleatórios e os demais têm seus rótulos
    interpolados a partir dos dois vizinhos já rotulados mais próximos.

    Args:
        dict_de_vertices: Dicionário de vértices (nome -> coordenada).
        dict_de_arestas: Dicionário de arestas (par de vértices -> peso).
        variacao: Fator de variação reservado para ajustes da geração. Não é
            utilizado no momento.

    Returns:
        Um dicionário que mapeia o nome de cada vértice ao seu rótulo.
    """

    dict_rotulos = {}
    vertice_aleatorio = 0

    if len(dict_de_vertices) <= 2:
        raise ValueError("Macho, tem que ter no mínimo 3 vértices né?")
    elif len(dict_de_vertices) < n_vertices_aleatorios*2:
        numero_de_vertices_aleatorios = 2
    else:
        numero_de_vertices_aleatorios = round(len(dict_de_vertices) / n_vertices_aleatorios)

    # Definindo pontos aleatórios
    while len(dict_rotulos) != numero_de_vertices_aleatorios:
        vertice_aleatorio = random.randint(0, len(dict_de_vertices) - 1)
        if f"v_{vertice_aleatorio}" not in dict_rotulos.keys():
            dict_rotulos[f"v_{vertice_aleatorio}"] = random.randint(limite_inferior*10, limite_superior*10) / 10

    # Definindo Interpolação
    while len(dict_rotulos) != len(dict_de_vertices):

        vertice_aleatorio = random.choice(
            list(dict_de_vertices.keys())
        )

        if vertice_aleatorio not in dict_rotulos:

            arestas_locais = []

            for (vertice_1, vertice_2), peso in dict_de_arestas.items():
                if vertice_aleatorio == vertice_1:
                    vizinho = vertice_2

                elif vertice_aleatorio == vertice_2:
                    vizinho = vertice_1

                else:
                    continue

                if vizinho in dict_rotulos:
                    arestas_locais.append((peso, vizinho))

            # Selecionando os dois vizinhos mais próximos
            arestas_locais.sort(key=lambda x: x[0])
            dois_vizinhos = arestas_locais[:2]

            peso_1, vertice1 = dois_vizinhos[0]
            peso_2, vertice2 = dois_vizinhos[1]

            if variacao < 0:
                raise ValueError("A variação não pode ser negativa.")

            influencia_1 = peso_1 ** (-variacao)
            influencia_2 = peso_2 ** (-variacao)

            rotulo_atual = round((
                dict_rotulos[vertice1] * influencia_1 +
                dict_rotulos[vertice2] * influencia_2
            ) / (influencia_1 + influencia_2), 2)

            dict_rotulos[vertice_aleatorio] = rotulo_atual

    return dict_rotulos


def _set_cores(dict_de_vertices: dict, dict_de_arestas: dict, n_vertices_aleatorios: int, limite_inferior: int, limite_superior: int, variacao=1) -> dict:
    """Define as cores RGB dos vértices usando interpolação.

    Alguns vértices recebem cores aleatórias e os demais têm suas cores
    interpoladas, canal a canal, a partir dos dois vizinhos já coloridos mais
    próximos.

    Args:
        dict_de_vertices: Dicionário de vértices (nome -> coordenada).
        dict_de_arestas: Dicionário de arestas (par de vértices -> peso).
        variacao: Fator de variação reservado para ajustes da geração. Não é
            utilizado no momento.

    Returns:
        Um dicionário que mapeia o nome de cada vértice a uma tupla RGB
        ``(r, g, b)`` com valores entre 0 e 255.

    Raises:
        ValueError: Se houver menos de 2 vértices.
    """

    dict_cores = {}

    if len(dict_de_vertices) < 2:
        raise ValueError("É necessário ter pelo menos 2 vértices.")

    if len(dict_de_vertices) < n_vertices_aleatorios*2:
        numero_de_vertices_aleatorios = 2
    else:
        numero_de_vertices_aleatorios = round(
            len(dict_de_vertices) / n_vertices_aleatorios
        )

    # Definindo cores aleatórias
    while len(dict_cores) != numero_de_vertices_aleatorios:

        vertice_aleatorio = random.randint(
            0, len(dict_de_vertices) - 1
        )

        nome_vertice = f"v_{vertice_aleatorio}"

        if nome_vertice not in dict_cores:

            dict_cores[nome_vertice] = (
                random.randint(limite_inferior, limite_superior),
                random.randint(limite_inferior, limite_superior),
                random.randint(limite_inferior, limite_superior)
            )

    # Definindo interpolação de cores
    while len(dict_cores) != len(dict_de_vertices):

        vertice_aleatorio = random.choice(
            list(dict_de_vertices.keys())
        )

        if vertice_aleatorio not in dict_cores:

            arestas_locais = []

            for (vertice_1, vertice_2), peso in dict_de_arestas.items():

                if vertice_aleatorio == vertice_1:
                    vizinho = vertice_2

                elif vertice_aleatorio == vertice_2:
                    vizinho = vertice_1

                else:
                    continue

                if vizinho in dict_cores:
                    arestas_locais.append((peso, vizinho))

            # Selecionando os dois vizinhos mais próximos
            arestas_locais.sort(key=lambda x: x[0])

            if len(arestas_locais) < 2:
                continue

            dois_vizinhos = arestas_locais[:2]

            peso_1, vertice1 = dois_vizinhos[0]
            peso_2, vertice2 = dois_vizinhos[1]

            # Interpolando individualmente os canais RGB
            if variacao < 0:
                raise ValueError("A variação não pode ser negativa.")
            
            influencia_1 = peso_1 ** (-variacao)
            influencia_2 = peso_2 ** (-variacao)

            cor_atual = tuple(
                round(
                    (
                        dict_cores[vertice1][i] * influencia_1 +
                        dict_cores[vertice2][i] * influencia_2
                    ) / (influencia_1 + influencia_2)
                )

                for i in range(3)
            )

            dict_cores[vertice_aleatorio] = cor_atual

    return dict_cores


def gerar_simulacao(numero_de_possiveis_pontos: int, tamanho_do_mapa: int, seed = 0, n_vertices_aleatorios = 6, limites_rotulos = (6, 10), limites_cores = (50,255), contraste=1, imprimir=False, plotar=False, usar_seed=False) -> dict:
    """Gera uma simulação completa de uma cidade com pontos de ônibus.

    Executa todas as etapas da geração (mapa, posições, vértices, arestas,
    rótulos e cores) e, opcionalmente, imprime o resultado no terminal e exibe
    a plotagem do grafo.

    Args:
        numero_de_possiveis_pontos: Quantidade de pontos de ônibus a
            distribuir no mapa.
        tamanho_do_mapa: Lado da matriz quadrada que representa o mapa.
        variacao: Fator de variação repassado às etapas de rótulos e cores.
        imprimir: Se ``True``, imprime os mapas de pontos, rótulos e cores.
        plotar: Se ``True``, exibe a plotagem do grafo com matplotlib.

    Returns:
        Um dicionário com a simulação, contendo a chave ``"vertices"`` (com
        ``"coordenadas"``, ``"rotulos"`` e ``"cores"``) e a chave ``"arestas"``
        (com ``"pesos"``).

    Raises:
        ValueError: Se ``numero_de_possiveis_pontos`` for maior que o número
            de células do mapa.
    """
    if usar_seed:
        random.seed(seed)

    if numero_de_possiveis_pontos > tamanho_do_mapa ** 2:
        raise ValueError("O quadrado do tamanho do mapa deve ser maior que o numero de possiveis pontos")

    mapa = _gerar_mapa(numero_de_possiveis_pontos, tamanho_do_mapa)
    posicoes = _set_posicao(mapa)
    vertices = _nomear_vertices(posicoes)
    arestas = _set_arestas(vertices)
    rotulos = _set_rotulos(vertices, arestas, n_vertices_aleatorios, limites_rotulos[0], limites_rotulos[1], variacao=contraste)
    cores = _set_cores(vertices, arestas, n_vertices_aleatorios, limites_cores[0], limites_cores[1], variacao=contraste)

    if imprimir:
        print("_" * 80)
        print("Mapa de pontos possíveis")
        for linha in mapa:
            print(linha)
        print("_" * 80)
        print("Mapa de rótulos")

        indice_vertice = 0

        mapa_novo = []
        for linha in mapa:
            linha_ = []
            for ponto in linha:
                if ponto == 1:
                    linha_.append(rotulos[f"v_{indice_vertice}"])
                    indice_vertice += 1
                elif ponto == 0:
                    linha_.append(0)
            mapa_novo.append(linha_[:])

        for linha in mapa_novo:
            print(linha)

        print("_" * 80)
        indice_vertice = 0

        mapa_novo = []
        for linha in mapa:
            linha_ = []
            for ponto in linha:
                if ponto == 1:
                    linha_.append(cores[f"v_{indice_vertice}"])
                    indice_vertice += 1
                elif ponto == 0:
                    linha_.append(0)
            mapa_novo.append(linha_[:])

        for linha in mapa_novo:
            print(linha)

    dict_vertices = {"coordenadas": vertices,
                     "rotulos": rotulos,
                     "cores": cores}

    dict_arestas = {"pesos": arestas}

    simulacao = {"vertices": dict_vertices,
                 "arestas": dict_arestas}

    if plotar:
        _gerar_plotagem(simulacao)

    return simulacao


def _gerar_plotagem(dict_simulacao: dict) -> None:
    """Desenha a simulação do grafo com matplotlib.

    Desenha as arestas, os vértices (coloridos conforme as cores geradas) e os
    rótulos de cada vértice.

    Args:
        dict_simulacao: Dicionário retornado por :func:`gerar_simulacao`.

    Returns:
        None. A figura é exibida na tela.
    """
    vertices = dict_simulacao["vertices"]["coordenadas"]
    rotulos = dict_simulacao["vertices"]["rotulos"]
    cores = dict_simulacao["vertices"]["cores"]
    arestas = dict_simulacao["arestas"]["pesos"]

    _, eixo = plt.subplots()

    for vertice_1, vertice_2 in arestas:
        x1, y1 = vertices[vertice_1]
        x2, y2 = vertices[vertice_2]
        eixo.plot(
            [x1, x2],
            [y1, y2],
            color="lightgray",
            linewidth=0.8,
            zorder=1,
        )

    xs = [posicao[0] for posicao in vertices.values()]
    ys = [posicao[1] for posicao in vertices.values()]
    cores_rgb = [
        tuple(canal / 255 for canal in cores[nome]) for nome in vertices
    ]

    eixo.scatter(xs, ys, c=cores_rgb, s=300, edgecolors="black", zorder=2)

    for nome, (x, y) in vertices.items():
        eixo.annotate(
            str(rotulos[nome]),
            (x, y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=3,
        )

    eixo.set_title("Simulação de pontos de ônibus")
    eixo.set_xlabel("x")
    eixo.set_ylabel("y")
    eixo.set_aspect("equal")
    eixo.grid(True, linestyle="--", alpha=0.4)

    plt.show()


if __name__ == "__main__":
    simulacao = gerar_simulacao(35, 15, plotar=True, n_vertices_aleatorios=35)
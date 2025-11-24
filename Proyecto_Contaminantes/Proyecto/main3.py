import heapq
from typing import Dict, Tuple
import networkx as nx
from itertools import combinations as cb
from main import grafo
import matplotlib.pyplot as plt

#Clase del Union-Find

class UnionFind:
    def __init__(self, elementos):
        self.padre = {x: x for x in elementos}
        self.rank = {x: 0 for x in elementos}
        self.birth = {x: float(0) for x in elementos}  # etiqueta de nacimiento del nodo

    def find(self, x):
        if self.padre[x] != x:
            self.padre[x] = self.find(self.padre[x])
        return self.padre[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # ya estaban conectados

        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            self.padre[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.padre[ry] = rx
        else:
            self.padre[ry] = rx
            self.rank[rx] += 1
        return True


# Grafo dirigido con pesos en aristas y nodos
G = nx.DiGraph()
G.add_edge("A", "B", w=0.45)
G.add_edge("A", "C", w=0.08)
G.add_edge("B", "D", w=0.34)
G.add_edge("C", "B", w=0.56)
G.add_edge("C", "D", w=0.75)
G.add_edge("D", "B", w=0.45)
G.add_edge("D", "A", w=0.98)
G.add_edge("A", "E", w=0.76)
G.add_edge("B", "E", w=0.97)

# Agregar pesos a los nodos directamente
G.nodes["A"]["w"] = 0.2
G.nodes["B"]["w"] = 0.3
G.nodes["C"]["w"] = 0.1
G.nodes["D"]["w"] = 0.5
G.nodes["E"]["w"] = 0.4



def dijkstra_filtrado(
    grafo: Dict[str, Dict[str, Dict[str, float]]],
    inicio: str,
    G: nx.DiGraph
) -> Dict[str, float]:
    """
    Dijkstra con métrica M(P) = max_{(s,t) in P} { 
        inf si min(w(s,t)-w(s), w(s,t), w(s,t)-w(t)) < 0 
        w(s,t) en otro caso }
    """

    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0.0

    heap = [(0.0, inicio)]

    while heap:
        distancia_actual, nodo_actual = heapq.heappop(heap)
        if distancia_actual > distancias[nodo_actual]:
            continue

        for vecino, atributos in grafo[nodo_actual].items():
            w_st = atributos['w']
            w_s = G.nodes[nodo_actual]['w']
            w_t = G.nodes[vecino]['w']

            # Condición de filtración: si no se cumple, descarta la arista
            if min(w_st - w_s, w_st, w_st - w_t) < 0:
                continue

            distancia_nueva = max(distancia_actual, w_st)

            if distancia_nueva < distancias[vecino]:
                distancias[vecino] = distancia_nueva
                heapq.heappush(heap, (distancia_nueva, vecino))

    return distancias


class DiagramaPersistencia:
    def __init__(self, G: nx.DiGraph):
        self.digrafo: nx.DiGraph = G
        self.MatrizBottleNeck: Dict[str, Dict[str, float]] = self.CalcularMatriz()
        self.filtracion: Dict[Tuple[str, str]|str, float] = self.CrearFiltracion()
        self.puntos = self.GenerarDiagrama()

    def CalcularNacimientoArista(self, u: str, v: str) -> float:
        minimo = float('inf')
        for t in self.digrafo.nodes():
            maximo = max(self.MatrizBottleNeck[u][t], self.MatrizBottleNeck[v][t])
            minimo = min(minimo, maximo)
        return minimo

    def CrearFiltracion(self) -> Dict[Tuple[str, str]|str, float]:
        filtracion = {}
        for nodo in self.digrafo.nodes():
            filtracion[nodo] = self.digrafo.nodes[nodo]['w']
        
        for u, v in cb(self.digrafo.nodes(), 2):
            filtracion[(u, v)] = self.CalcularNacimientoArista(u, v)
        return filtracion

    def CalcularMatriz(self) -> Dict[str, Dict[str, float]]:
        adjacency_dict = nx.to_dict_of_dicts(self.digrafo)
        matriz = {}
        for nodo in self.digrafo.nodes():
            matriz[nodo] = dijkstra_filtrado(adjacency_dict, nodo, self.digrafo)
        return matriz

    def VerificarFiltración(self):
        bool = True
        for clave in self.filtracion.keys():
            if(clave is Tuple[str,str]):
                n_inicio = self.filtracion[clave[0]]
                n_final = self.filtracion[clave[1]]
                n_arista = self.filtracion[clave]
                if(n_inicio <= n_arista and n_final <= n_inicio):
                    continue
                else:
                    bool = False

        if(bool == True):
            print("La filtración cumple las condiciones mínimas")
        else:
            print("La condicion INCUMPLE las condiciones necesarias")
        return
    
    def GenerarDiagrama(self):
        """Construye el diagrama de persistencia H0 usando Union-Find."""
        # Ordenar filtración por valor
        elementos_ordenados = sorted(self.filtracion.items(), key=lambda x: x[1])
        uf = UnionFind(self.digrafo.nodes())

        # Asignar nacimiento de cada nodo
        for nodo in self.digrafo.nodes():
            uf.birth[nodo] = self.filtracion[nodo]

        puntos = []

        for clave, valor in elementos_ordenados:
            if isinstance(clave, tuple):
                u, v = clave
                ru, rv = uf.find(u), uf.find(v)
                if ru != rv:
                    # Componente con menor nacimiento muere
                    if uf.birth[ru] < uf.birth[rv]:
                        puntos.append((uf.birth[rv], valor))
                        uf.union(u, v)
                        uf.birth[uf.find(u)] = uf.birth[ru]
                    else:
                        puntos.append((uf.birth[ru], valor))
                        uf.union(u, v)
                        uf.birth[uf.find(u)] = uf.birth[rv]

        return puntos

    def VisualizarDiagrama(self):
        """Visualiza los puntos del diagrama de persistencia."""
        if not self.puntos:
            print("Primero genera el diagrama con `GenerarDiagrama()`.")
            return

        x, y = zip(*self.puntos)
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c="blue", s=50, label="Componentes H0")
        plt.plot([0, max(y)], [0, max(y)], "r--", label="Diagonal")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.title("Diagrama de Persistencia H₀ (Significancia)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def VisualizarDiagramaConfianza(self):
        """Visualiza los puntos del diagrama de persistencia."""
        if not self.puntos:
            print("Primero genera el diagrama con `GenerarDiagrama()`.")
            return
        
        puntos_confianza = []
        for birth, death in self.puntos:
            new_birth =1- death
            new_death = 1- birth
            puntos_confianza.append((new_birth,new_death))


        x, y = zip(*puntos_confianza)
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c="blue", s=50, label="Componentes H0")
        plt.plot([0, max(y)], [0, max(y)], "r--", label="Diagonal")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.title("Diagrama de Persistencia H₀ (Confianza)")
        plt.legend()
        plt.grid(True)
        plt.show()


# Instancia
D = DiagramaPersistencia(grafo)
D.VerificarFiltración()
D.VisualizarDiagramaConfianza()


for punto in D.puntos:
    print(f"birth: {punto[0]} death: {punto[1]}")
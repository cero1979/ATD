import heapq
from typing import Dict, Tuple
import networkx as nx
from itertools import combinations as cb

G = nx.DiGraph()
G.add_edge("A", "B", w=3.0)
G.add_edge("A", "C", w=1.0)
G.add_edge("B", "D", w=2.0)
G.add_edge("C", "B", w=1.0)
G.add_edge("C", "D", w=4.0)
G.add_edge("D", "B", w=3.0)
G.add_edge("D", "A", w=0.3)
G.add_edge("A", "E", w=1.2)
G.add_edge("B", "E", w=1.3)

diccionario = nx.to_dict_of_dicts(G)
print(diccionario)

def dijkstra(grafo: Dict[str, Dict[str, Dict[str, float]]], inicio: str)->Dict[str, float]:
    # Diccionario para almacenar la distancia mínima desde inicio
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = float(0)
    
    #Diccionario para almacenar el camino más corto
    caminos = {nodo: [] for nodo in grafo}
    caminos[inicio] = [inicio]
    
    #Cola de prioridad para explorar los nodos más cercanos
    heap = [(float(0), inicio)]
    
    while heap:
        distancia_actual, nodo_actual = heapq.heappop(heap)
        
        # Saltar si ya hay una distancia menor registrada
        if distancia_actual > distancias[nodo_actual]:
            continue
        
        # Revisar vecinos del nodo actual
        for vecino, peso in grafo[nodo_actual].items():
            distancia_nueva = max(distancia_actual, peso['w'])
            if distancia_nueva < distancias[vecino]:
                distancias[vecino] = distancia_nueva
                # Construir el camino más corto hasta el vecino
                caminos[vecino] = caminos[nodo_actual] + [vecino]
                heapq.heappush(heap, (distancia_nueva, vecino))
    
    return distancias


"""
El grafo que se le pasa al clase del DiagramaPersistencia, 
es el grafo más grande( Es decir al asociado a el nivel de la filtracion más grande que en este caso es 1)
"""

class DiagramaPersistencia:
    def __init__(self, G: nx.DiGraph):
        self.digrafo: nx.DiGraph = G
        self.MatrizBottleNeck: Dict[str, Dict[str, float]]  = self.CalcularMatriz()
        self.filtracion: Dict[Tuple[str, str], float] = self.CrearFiltracion()
        return
    
    def CalcularNacimientoArista(self, u: str, v: str) -> float:
        minimo = float('inf')
        for t in self.digrafo.nodes():
            maximo = max(self.MatrizBottleNeck[u][t], self.MatrizBottleNeck[v][t])

            if maximo < minimo:
                minimo = maximo
        
        if(u == "A" and v == "E"):
            print(minimo)
        return minimo    
    
    def CrearFiltracion(self) -> Dict[Tuple[str, str], float]:
        filtracion = dict()
        for u, v in cb(self.digrafo.nodes(), 2):           
            filtracion[(u,v)] = self.CalcularNacimientoArista(u,v)
        return filtracion
    
    def CalcularMatriz(self) -> Dict[str, Dict[str, float]]:
        adjacency_dict = nx.to_dict_of_dicts(self.digrafo)
        diccionario = dict()
        for nodo in self.digrafo.nodes():
            diccionario[nodo] = dijkstra(adjacency_dict, nodo)
        return diccionario
    
    def GenerarDiagrama(self):
        #Aqui se implemneta el algoritmo Union Find para poder generar el diagrama de Persistencia
        return
    

"""
Creemos una instancia de la clase
"""
"""D = DiagramaPersistencia(G)
#print(D.MatrizBottleNeck)
print(D.filtracion)
"""

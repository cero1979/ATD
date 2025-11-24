import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import gudhi as gd
import mlcausality
from statsmodels.tsa.stattools import adfuller
from itertools import chain, combinations


path: str = "C:/Users/aleja/OneDrive/Documents/semestre 8/TDA/Proyecto/datosAire.xlsx"

def get_powerset(iterable):
    # Genera todos los subconjuntos, incluyendo el vacío.
    s = list(iterable)
    combinations_generator = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    for subset_tuple in combinations_generator:
        yield frozenset(subset_tuple)

class TdaSeries:
    loaders = {"csv": pd.read_csv, "xlsx": pd.read_excel}  # extensiones permitidas
    
    def __init__(self, link: str, metodo: str, numero_lags: int, prefix: str, ColumnasIgnorar: list = None):
        extension = TdaSeries.extraer_tipo(link)
        if extension not in TdaSeries.loaders:
            raise ValueError(f"Extensión no soportada: {extension}")
        
        # Cargar dataframe
        self.dataframe: pd.DataFrame = TdaSeries.loaders[extension](link)  


        # Construir matriz de las aristas
        self.etiquetas_aristas: pd.DataFrame = self.construir_matriz(metodo, numero_lags, prefix, ColumnasIgnorar)

        # Construir la lista de Estacionaridad de Nodos
        self.etiquetas_nodos: dict = self.construir_nodos(prefix, ColumnasIgnorar)

        # crear el grafo bajo el nivel de Confianza estandar que es 0%
        self.G: nx.DiGraph = self.crear_grafo()

        # creemos el complejo de Dowker bajo el nivel de significancia de Confianza estandar que es de 0%
        self.DowkerComplex: set = self.crear_complejo()

    def construir_matriz(self, metodo: str, numero_lags: list, prefix: str, ColumnasIgnorar: list = None)-> pd.DataFrame:
        df_filter = TdaSeries.select_by_prefix(self.dataframe, prefix)
        ##Eliminar las columnas a Ignorar
        if(ColumnasIgnorar != None):
            columnasEliminar = [c for c in ColumnasIgnorar if c in df_filter.columns]
            df_filter = df_filter.drop(columns=columnasEliminar, axis = 1)


        ##Calcular la matriz de p-valores
        results = mlcausality.multiloco_mlcausality(df_filter.iloc[:, 0:], lags = numero_lags) ##Me devuelve un dataFrame
        ##Miremos sí el Multiloco ya tiene NaN

        names = [c for c in df_filter.columns if c != "date"]
        df = pd.DataFrame(index=names, columns=names) #Inicializacion del DataFrame

        ##Ahora vamos a recorrer el Dataframe

        if metodo == "w":
            col_pval = "wilcoxon.pvalue"
        elif metodo == "s":
            col_pval = "sign_test.pvalue"
        else:
            raise ValueError("Método no reconocido, use 'w' o 's'.")
        

        for fila in df.index:
            for col in df.columns:
                match = results[(results["X"] == fila) & (results["y"] == col)]  ##El match necesario
                if(not match.empty):
                    df.loc[fila, col] = match[col_pval].iloc[0]
                else:
                    df.loc[fila, col] = -1
        return df

                
    def construir_nodos(self, prefix: str, ColumnasIgnorar: list = None) -> dict:
        ##Realizar la prueba estadistica de Estacionaridad de las series
        p_valores = {}
        df_filter = TdaSeries.select_by_prefix(self.dataframe, prefix)

        ##Eliminar las columnas a Ignorar
        if(ColumnasIgnorar != None):
            columnasEliminar = [c for c in ColumnasIgnorar if c in df_filter.columns]
            df_filter = df_filter.drop(columns=columnasEliminar)

        df_filter.columns
        ##Calcular la prueba de hipotesis por columna
        for col in df_filter.columns:
            serie = df_filter.loc[:, col]
            p_valor = adfuller(serie)[1]
            p_valores[col] = p_valor

        return p_valores
    
    def crear_grafo(self) -> nx.DiGraph:
        G = nx.DiGraph()
        df = self.etiquetas_aristas

        # Agreguemos los nodos
        for col_name in df.columns:
            G.add_node(col_name, w=self.etiquetas_nodos[col_name])

        # Agreguemos las aristas
        for fila in G.nodes():
            for col in G.nodes():
                if fila != col:
                    if not pd.isna(df.loc[fila, col]):  # evita NaN
                        G.add_edge(fila, col, w=df.loc[fila, col])

        return G
    
    def crear_complejo(self)->set: #Esta función se encarga de crear el complejo de Dowker 

        #Creamos un diccionario vacío
        Alcanzables = {}  

        #Rellenamos el diccionario vació

        for nodo in self.G.nodes():
            Alcanzables[nodo] = nx.descendants(self.G, nodo) | {nodo}

        #Imprimimos el diccionario

        ##Ahora construyamos la relación de alcanzabilidad entre nodos
        relacion = []
        
        for nodo1 in self.G.nodes():
            for nodo2 in self.G.nodes():
                if(Alcanzables[nodo1].intersection(Alcanzables[nodo2])):
                    relacion.append((nodo1, nodo2))
        

        #Ahora construyamos el complejo de Dowker de la relación
        K_x = set()
                

        #Rellena el Complejo Simplicial con los 0-simplices
        for nodo in self.G.nodes():
            K_x.add(frozenset({nodo}))
        
        #Terminemos de rellenar el complejo simplicial con todos los subconjuntos de los complejos maximales
        for nodo in self.G.nodes():
            max_simplex = set()
            for x, y in relacion:
                if(y == nodo):
                    max_simplex.add(x)    
            K_x.add(frozenset(max_simplex))

        #Ahora toca terminar de rellenar el complejo
        simplices_a_añadir = []
        for simplex in K_x:
            if(len(simplex) > 1):
                for nuevo_simplex in get_powerset(simplex):
                    simplices_a_añadir.append(nuevo_simplex)

        K_x.update(simplices_a_añadir)
        return K_x

    def visualizar_complejo(self, titulo: str):
        ##Aqui guardamos los simplices
        cero_simplices = set()
        uno_simplices = []
        dos_simplices = []
        for simplex in z.DowkerComplex:
            if(len(simplex) - 1 == 0):
                cero_simplices = cero_simplices.union(simplex)
            elif(len(simplex) - 1 == 1):
                uno_simplices.append(tuple(simplex))
            elif(len(simplex) - 1 == 2):
                dos_simplices.append(tuple(simplex))  
        cero_simplices = list(cero_simplices)

                
        G = nx.Graph()
        G.add_nodes_from(cero_simplices) 
        G.add_edges_from(uno_simplices)
        
        # Layout
        pos = nx.spring_layout(G, seed=42)

        #Asignar colores a nodos (ejemplo: colormap)
        cmap = plt.cm.viridis
        node_list = G.nodes()
        node_colors = {node: cmap(i / (len(node_list)-1 if len(node_list)>1 else 1)) for i, node in enumerate(cero_simplices)}

        # Dibujar los 2-símplices primero (para que no tapen aristas/nodos)
        for simplex in dos_simplices:
            xs = [pos[v][0] for v in simplex]
            ys = [pos[v][1] for v in simplex]
            plt.fill(xs, ys, alpha=0.3, color="orange")

        # Dibujar aristas encima
        nx.draw_networkx_edges(G, pos, edge_color="black", width=1.5)

        # Dibujar nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_size=100,
            node_color= [node_colors[node] for node in G.nodes()]
        )

        # Crear la leyenda
        lista_nodos = G.nodes()
        patches = [mpatches.Patch(color=node_colors[n], label=str(n)) for n in lista_nodos]
        plt.legend(handles=patches, title="Nodos", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.title(titulo)
        plt.axis("off")
        plt.show()
        return
    
    def generar_filtración(nivel_Confianza: int):
        if(nivel_Confianza > 1 or nivel_Confianza < 0):
            print("Haz Cometido un error, ingresa un número entre [0,1]")
        else:
            nivel_significancia = 1 - nivel_Confianza
        return

    @staticmethod
    def extraer_tipo(link: str) -> str:
        pos = link.rfind(".")
        return link[pos + 1:]
    

    @staticmethod
    def select_by_prefix(df: pd.DataFrame, prefix: str):
        cols = [c for c in df.columns if c.startswith(prefix)]
        return df.loc[:, cols]



z = TdaSeries(path, "s", [5], "median", ["median_NEsperanza"])

complejoSimplicial = z.DowkerComplex
#print(len(complejoSimplicial))
#z.visualizar_complejo("Complejo Simplicial")
grafo = z.G
print(grafo)

                    
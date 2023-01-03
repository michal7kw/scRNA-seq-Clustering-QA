# ------ Import necessary packages ----
import pandas as pd
import networkx as nx

def create_graph(dir):
    G = nx.read_gexf(dir)
    pos = nx.spring_layout(G)
    return G, pos

def create_graph_csv(dirs):
    input_data = pd.read_csv(dirs["graph_in_csv"], header=0, usecols={1,2,3})
    records = input_data.to_records(index=False)
    result = list(records)
    len(result)
    G = nx.Graph()
    G.add_weighted_edges_from(result)
    pos = nx.spring_layout(G)
    return G, pos

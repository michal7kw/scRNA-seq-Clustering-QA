import random
import itertools

import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import dimod
import hybrid 
# import dwavebinarycsp
import dwave.inspector
import dwave_networkx as dnx
# from minorminer import find_embedding
# from dwave.embedding import embed_ising
from dimod import BinaryQuadraticModel
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler, LeapHybridDQMSampler, LeapHybridCQMSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite, FixedEmbeddingComposite
from dwave.cloud.client import Client

import json

def graph_subsampling(G, gamma, solver):
    # P = 1 * len(G.nodes)
    P = 1
    # Initialize our Q matrix
    Q = defaultdict(int)
    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += -P*(1- G.get_edge_data(u, v)["weight"])
        Q[(v,v)] += -P*(1- G.get_edge_data(u, v)["weight"])
        Q[(u,v)] +=  P*(1- G.get_edge_data(u, v)["weight"])
    for i in G.nodes:
        Q[(i,i)] += gamma

    num_reads = 100
    chain_strength = 4

    if solver == "hybrid":
        sampler = LeapHybridSampler()
        response = sampler.sample_qubo(Q, label="prun_data")
    elif solver == "fixed_embedding":
        save = False
        try:
            a_file = open(dirs["embedding_pru"])
            embedding = json.load(a_file)
            a_file.close()
            sampler = FixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'), embedding)
            print("found embedding")
        except IOError:
            save = True
            print("generate new embedding")
            sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))

        response = sampler.sample_qubo(Q, label="prun_data", chain_strength=chain_strength, num_reads=num_reads)    
        
        if save:
            embedding = sampler.properties['embedding']
            a_file = open(dirs["embedding_pru"], "w")
            json.dump(embedding, a_file)
            a_file.close()   
    elif solver == "embedding_composite":
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(Q, label="cellpath", chain_strength=chain_strength, num_reads=num_reads)

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Num. of occurrences'))
    print('-' * 60)

    i=0
    for sample, E, occur in response.data(fields=['sample','energy', "num_occurrences"]):
        # select clusters
        S0 = [k for k,v in sample.items() if v == 0]
        S1 = [k for k,v in sample.items() if v == 1]

        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(occur)))

    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    print("S0 length: ", len(S0))
    print("S1 length: ", len(S1))

    label = "label1"

    for i in S0:
        G.nodes(data=True)[i][label] = 0
        
    for i in S1:
        G.nodes(data=True)[i][label] = 1
    
    return response

def graph_subsampling_2(G, gamma):
    sampler = EmbeddingComposite(DWaveSampler())
    # sampler = LeapHybridSampler()
    S = dnx.maximum_independent_set(G, sampler=sampler, num_reads=10, label='graph_subsampling_2', time_limit=3.0)
    
    # Visualize the results
    # k = G.subgraph(S)
    notS = list(set(G.nodes()) - set(S))
    # othersubgraph = G.subgraph(notS)
    # pos = nx.spring_layout(G)

    label = "label1"
    for i in S:
        G.nodes(data=True)[i][label] = 1

    for i in notS:
        G.nodes(data=True)[i][label] = 0

    return S

def prune_graph(G, pos, dirs):
    prun_nodes = [x for x,y in G.nodes(data=True) if y['label1']==1]
    H = G.subgraph(prun_nodes)
    nx.write_gexf(H, dirs["graph_out_pru2"])

    plt.cla()
    nx.draw_networkx_nodes(H, pos, node_size=20, nodelist=H.nodes)
    nx.draw_networkx_edges(H, pos, edgelist=H.edges, style='solid', width=1)
    plt.savefig(dirs["img_out_p2"], bbox_inches='tight')

    return H
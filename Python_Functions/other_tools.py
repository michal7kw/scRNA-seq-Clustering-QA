import random
import itertools

import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations

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

def check_embedding_inspector(G, gamma_factor):
    print("starting")
    name = "for_inspection"

    edges_weights = G.size(weight="weight")
    nodes_len = len(G.nodes)
    gamma = gamma_factor * edges_weights/nodes_len
    print("gama: ", gamma)
    k = 8

    # Initialize our Q matrix
    Q = defaultdict(int)
    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += k*G.get_edge_data(u, v)["weight"]
        Q[(v,v)] += k*G.get_edge_data(u, v)["weight"]
        Q[(u,v)] += k *-2*G.get_edge_data(u, v)["weight"]

    for i in G.nodes:
        Q[(i,i)] += gamma*(1-len(G.nodes))

    for i, j in combinations(G.nodes, 2): # do you need this term ???
        Q[(i,j)] += 2*gamma
    
    num_reads = 5
    chain_strength = 4
    print("Looking for a file")
    try:
        a_file = open(dirs["embedding"])
        embedding = json.load(a_file)
        a_file.close()
        print("found embedding")
        sampler = FixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'), embedding)
    except IOError:
        print("embedding not found")
        return

    print("Sampling")
    response = sampler.sample_qubo(Q, label=name, chain_strength=chain_strength, num_reads=num_reads)    
    dwave.inspector.show(response) # , block='never'

def retrive_response(problem_id, token):
    client = Client(token=token)
    future = client.retrieve_answer(problem_id)
    sampleset = future.sampleset
    return sampleset

def disconnected_components(G):
    lengths  = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    print(lengths)
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for s in S:
        print(len(s.nodes()))
        if len(s.nodes()) > 15:
            subindex = 0
            for n in s.nodes():
                G.nodes(data=True)[n]["subindex"] = subindex
                G.nodes(data=True)[n]["valid"] = 1
                subindex = subindex + 1            
        else:
            for n in s.nodes():
                G.nodes(data=True)[n]["valid"] = 0
    return G, S, lengths
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

def clustering_dqm(G, num_of_clusters, gamma):
    nodes = G.nodes
    edges = G.edges
    clusters = [i for i  in range(0, num_of_clusters)]

    dqm = dimod.DiscreteQuadraticModel()
    for node in nodes:
        dqm.add_variable(num_of_clusters, label=node)
    
    for node in nodes:
        dqm.set_linear(node, [gamma*(1-len(G.nodes)/num_of_clusters) for cluster in clusters])

    for i, j in combinations(nodes, 2):
        dqm.set_quadratic(i, j, {(cluster, cluster) : 2*gamma for cluster in clusters})

    # ? wrong: add_quadratic instead of set_quadratic
    for u, v in edges:
        dqm.set_quadratic(u, v, {(cluster, cluster) : -2*G.get_edge_data(u, v)["weight"] for cluster in clusters})
        dqm.set_linear(u, [G.get_edge_data(u, v)["weight"] for cluster in clusters])
        dqm.set_linear(v, [G.get_edge_data(u, v)["weight"] for cluster in clusters])

    sampleset = LeapHybridDQMSampler().sample_dqm(dqm, label='DQM - scRAN-seq') 
    print("Energy: {}\nSolution: {}".format(sampleset.first.energy, sampleset.first.sample)) 
    return sampleset

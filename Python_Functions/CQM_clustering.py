# ------ Import necessary packages ----
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

def clustering_cqm(G, num_of_clusters, cluster_size_constraint, weights_amplification):
    nodes = G.nodes
    print("CQM nomber of nodes: ", G.number_of_nodes())
    edges = G.edges
    print("CQM nomber of nodes: ", G.number_of_edges())
    clusters = range(num_of_clusters)

    cqm = dimod.ConstrainedQuadraticModel()

    print("\nAdding variables....")
    v = [[dimod.Binary(f'v_{i},{k}') for k in clusters] for i in nodes]

    print("\nAdding one-hot constraints...")
    for i in nodes:    
        cqm.add_discrete([f'v_{i},{k}' for k in clusters], label=f"one-hot-node-{i}")

    print("\nAdding objective...")
    min_edges = []
    for i,j in edges:
        for p in clusters:
            min_edges.append(v[int(i)][p]+v[int(j)][p] - 2*weights_amplification*G.get_edge_data(i, j)["weight"]*v[int(i)][p]*v[int(j)][p])
    cqm.set_objective(sum(min_edges))

    print("\nAdding partition size constraint...")
    for j in clusters:
        cqm.add_constraint(sum(v[int(i)][j] for i in nodes) >= cluster_size_constraint, label=f'cluster_size{j}')

    print("\nSending to the solver...")

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm, label='CQM - scRAN-seq')
    print("Energy: {}\nSolution: {}".format(sampleset.first.energy, sampleset.first.sample)) 
    return sampleset

def clustering_cqm_2(G, num_of_clusters):
    nodes = G.nodes()
    edges = G.edges()
    clusters = range(num_of_clusters)

    cqm = dimod.ConstrainedQuadraticModel()

    print("\nAdding variables....")
    v = [[dimod.Binary(f'v_{G.nodes(data=True)[i]["subindex"]},{k}') for k in clusters] for i in nodes]

    print("\nAdding one-hot constraints...")
    for i in nodes:    
        c = G.nodes(data=True)[i]["subindex"]
        # c = [x for x,y in G.nodes(data=True) if y['label']==i][0]
        cqm.add_discrete([f'v_{c},{k}' for k in clusters], label=f"one-hot-node-{i}")

    print("\nAdding objective...")
    min_edges = []
    for i,j in edges:
        c = G.nodes(data=True)[i]["subindex"]
        d = G.nodes(data=True)[j]["subindex"]
        for p in clusters:
            min_edges.append(v[int(c)][p]+v[int(d)][p] - 2*G.get_edge_data(i, j)["weight"]*v[int(c)][p]*v[int(d)][p])
    cqm.set_objective(sum(min_edges))

    print("\nAdding partition size constraint...")
    for j in clusters:
        cqm.add_constraint(sum(v[int(G.nodes(data=True)[i]["subindex"])][j] for i in nodes) >= 20, label=f'cluster_size{j}')

    print("\nSending to the solver...")

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm, label='CQM - scRAN-seq')
    print("Energy: {}\nSolution: {}".format(sampleset.first.energy, sampleset.first.sample)) 
    return sampleset
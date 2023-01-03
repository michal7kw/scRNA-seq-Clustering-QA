from asyncio import futures
from cProfile import label
from email import iterators
from http import client
from lib2to3.pgen2 import tokenize
import math
import random
import types
import itertools

from matplotlib.colors import same_color
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations, count
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import dimod
import hybrid 
import dwave.inspector
import dwave_networkx as dnx
from dimod import BinaryQuadraticModel
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler, LeapHybridDQMSampler, LeapHybridCQMSampler
from dwave.system.composites import EmbeddingComposite, LazyFixedEmbeddingComposite, FixedEmbeddingComposite
from dwave.cloud.client import Client

import json
import pickle

from Python_Functions.create_graphs import *
from Python_Functions.plot_and_save import *
from Python_Functions.other_tools import *
from Python_Functions.QA_subsampling import *
from Python_Functions.BQM_clustering import *
from Python_Functions.DQM_clustering import *
from Python_Functions.CQM_clustering import *

def define_dirs(n, k, dim, ord, g, gf, custom,type, experiment):
    # n-size, k-k_nn, dim-dimensions, ord-max_degree, g-gamma, custom-for one's needs, type-type of graph
    type_names = ["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"]
    g = str(g).replace( ".", "")
    gf = str(gf).replace( ".", "")

    dirs = {
        "name"              : ''.join([str(n), "_graph_snn", "_k", str(k), "_dim", str(dim), type_names[type], str(ord)]),
        
        "graph_in"          : ''.join(["./graphs/R_generated/",      str(experiment), str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".gexf"       ]),
        "graph_in_csv"      : ''.join(["./graphs/R_generated/",      str(experiment), str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".csv"        ]),
        "graph_in_pru"      : ''.join(["./graphs/R_generated/",      str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".gexf"       ]),
        
        "graph_out"         : ''.join(["./graphs/Python_generated/"  ]),
        "graph_out_bqm"     : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim), "_gf", str(gf), type_names[type], str(ord), custom, "_out.gexf"   ]),
        "graph_out_dqm"     : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_dqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, ".gexf"       ]),
        "graph_out_cqm"     : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_cqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, ".gexf"       ]),
        "graph_out_pru1"    : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".gexf"       ]),
        "graph_out_pru2"    : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "2.gexf"      ]),

        "img_in"            : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, ".png"        ]),
        "img_out_bqm"       : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_bqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_gf", str(gf), type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_dqm"       : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_dqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_cqm"       : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_cqm_graph_snn" , "_k", str(k), "_dim", str(dim), "_g", str(g)  , type_names[type], str(ord), custom, "_out.png"    ]),
        "img_out_p1"        : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "_out1.png"   ]),
        "img_out_p2"        : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "_out2.png"   ]),
        "img_out_p3"        : ''.join(["./graphs/Python_generated/", str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord), custom, "_out3.png"   ]),
        
        "embedding"         : ''.join(["./Embedding/", str(experiment), str(n), "_graph_snn"     , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".json"       ]),
        "embedding_pru"     : ''.join(["./Embedding/", str(experiment), str(n), "_pru_graph_snn" , "_k", str(k), "_dim", str(dim),                 type_names[type], str(ord),         ".json"       ])
    }
    return dirs
  
solvers = {
    "h"     : "hybrid",
    "fe"    : "fixed_embedding",
    "ec"    : "embedding_composite"
}
solver = solvers["h"] # type of used solver

n = 961     # size of the graph
k = 8       # k_nn used for SNN
ord = 8     # maximum order of node degree when "trimmed" mode is enabled
dim = 20    # number of dimensions used for SNN
g_type = 1  # ["_", "_trimmed_", "_negedges_", "_trimmed_negedges_"], where "_" -> unaltered SNN output
color = 0   # initial value of clusters coloring fro bqm
gamma_factor = 0.05         # to be used with dqm, weights the clusters' sizes constraint
gamma = 0.005               # to be used with bqm
custom = "_pruned"          # additional metadata for file names
terminate_on = "conf"       # other options: "conf", "min_size"
size_limit = 40             # may be used in both bqm and dqm // to finish
num_of_clusters = 5         # may be used in both bqm and dqm // to finish
iter_limit = 2              # limit of iteration 
chain_strength = 20
experiment = "kidney_endo/"

# define local directories
dirs = define_dirs(n, k, dim, ord, gamma, gamma_factor, custom, g_type, experiment)

# --------- import graph automatic name --------
print(dirs["graph_in"])
G, pos = create_graph(dirs["graph_in"])

# --------- save input graph ---------
print(dirs["img_in"])
plot_and_save_graph_in(G, pos, dirs)

# --------- CQM ---------
sampleset_cqm = clustering_cqm(G, num_of_clusters=5, cluster_size_constraint=50, weights_amplification=1.2)
print(dirs['graph_out_cqm'] + "\n" + dirs['img_out_cqm']) 
plot_and_save_graph_out_cqm(G, pos, dirs, sampleset_cqm, num_of_clusters)


################ Experimental ################

# --------- look for disconnected components ---------
G, S, lengths = disconnected_components(G)
print(lengths)

#  --------- Graph Subsampling ---------
response = graph_subsampling(G, 3, solver)
# S = graph_subsampling_2(G, 10)
# dwave.inspector.show(response)
plot_and_save_graph_out_mvc(G, pos, dirs)
H = prune_graph(G, pos, dirs)

# --------- DQM -----------
sampleset_dqm = clustering_dqm(G, num_of_clusters, gamma)       
plot_and_save_graph_out_dqm(G, pos, dirs, sampleset_dqm)

# --------- CQM 2 ---------
sampleset_cqm = clustering_cqm_2(G, num_of_clusters)
pos = nx.spring_layout(G)
plot_and_save_graph_out_cqm_2(G, pos, dirs, sampleset_cqm, num_of_clusters)

#  --------- BQM recursive -----------
iteration = 1
response = clustering_bqm(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit, iter_limit, chain_strength)
plot_and_save_graph_out_bqm(G, pos, dirs)

#  --------- BQM_2 recursive, lessened constraints  -----------
iteration = 1
response = clustering_bqm_2(G, iteration, dirs, solver, 0.01, color, terminate_on, size_limit, 1, 1)
plot_and_save_graph_out_bqm(G, pos, dirs)
# dwave.inspector.show(response)

#  --------- BQM_3 recursive, lessened constraints-----------
iteration = 1
clustering_bqm_3(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit)
plot_and_save_graph_out_bqm(G, pos, dirs)

################ Results evaluation 1 ################

token = "" 
p_id = ""

#  --------- Check graph embedding in the inspector -----------
check_embedding_inspector(G, gamma_factor)

#  --------- Retrive response -----------
sampleset = retrive_response(problem_id=p_id, token=token)
# dwave.inspector.show(sampleset)
print(sampleset)
dir = dirs["graph_out"] + experiment
print(dir)
plot_and_save_graph_out_cqm_multi(G, pos, dir, sampleset, num_of_clusters, 16)

################ Results evaluation 2 ################
response = sampleset
response.first.num_occurrences

lut = response.first.sample
# Interpret best result in terms of nodes and edges
S0 = [node for node in G.nodes if not lut[node]]
S1 = [node for node in G.nodes if lut[node]]

print("S0 length: ", len(S0))
print("S1 length: ", len(S1))

# Assign nodes' labels
label = "cluster"
col = random.randint(0, 100)
for i in S0:
    # G.nodes(data=True)[i][label] = 100 - color
    G.nodes(data=True)[i][label] = col

col = random.randint(120, 220)    
for i in S1:
    # G.nodes(data=True)[i][label] = color - 100
    G.nodes(data=True)[i][label] = col


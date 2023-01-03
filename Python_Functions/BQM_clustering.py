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

def clustering_bqm(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit, iter_limit, chain_strength):

    name_spec = ''.join([dirs["name"], "_", solver]) 
    
    edges_weights = G.size(weight="weight")
    nodes_len = len(G.nodes)
    gamma = gamma_factor * edges_weights/nodes_len
    print("gamma: ", gamma)
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

    for i, j in combinations(G.nodes, 2):
        Q[(i,j)] += 2*gamma

    # --------------
    print("... Running on QPU ...")
    
    num_reads = 500
    chain_strength = chain_strength

    if solver == "hybrid":
        sampler = LeapHybridSampler()
        response = sampler.sample_qubo(Q, label=name_spec)
    elif solver == "fixed_embedding":
        save = False
        try:
            a_file = open(dirs["embedding"])
            embedding = json.load(a_file)
            a_file.close()

            sub_embedding = dict((k, embedding[k]) for k in G.nodes if k in embedding)
            
            sampler = FixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'), sub_embedding)
            print("found embedding")
        except IOError:
            save = True
            print("generate new embedding")
            # sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))
            sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))

        response = sampler.sample_qubo(Q, label=name_spec, chain_strength=chain_strength, num_reads=num_reads, return_embedding=True)    
        
        if save:
            # embedding = sampler.properties['embedding']
            embedding = response.info['embedding_context']['embedding']
            a_file = open(dirs["embedding"], "w")
            json.dump(embedding, a_file)
            a_file.close()   
    elif solver == "embedding_composite":
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(Q, label=name_spec, chain_strength=chain_strength, num_reads=num_reads)

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
        
        if (i > 3):
            break
        i = i + 1
    
    label = "label" + str(iteration)
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    print("S0 length: ", len(S0))
    print("S1 length: ", len(S1))
    if terminate_on == "min_size":
        if(len(S0)>size_limit and len(S1)>size_limit and iteration < iter_limit):
            # Assign nodes' labels
            col = random.randint(0, 100)
            for i in S0:
                # G.nodes(data=True)[i][label] = 100 - color
                G.nodes(data=True)[i][label] = col
            
            col = random.randint(120, 220)    
            for i in S1:
                # G.nodes(data=True)[i][label] = color - 100
                G.nodes(data=True)[i][label] = col
            # write to the graph file
            # file_name = "clustring_" + str(iteration) + ".gexf"
            # nx.write_gexf(G, file_name)

            clustering_bqm(G.subgraph(S0), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
            clustering_bqm(G.subgraph(S1), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
    #to-do
    elif terminate_on == "conf":
        if len(response.record.energy) > 3:
            print("energies", response.record.energy[:3])
            if response.record.energy[3]>0.1 or response.record.energy[3]<-0.1: # this is to avoid dicivion by 0
                ratio = response.record.energy[0]/response.record.energy[3]
            else:
                print("error: 3rd lowest energy too close to zero; check your results")
                col = random.randint(0, 100)
                for i in G.nodes:
                    G.nodes(data=True)[i][label] = col
                return response
            difference = np.abs(response.record.energy[0]-response.record.energy[3])
            print("ratio:", ratio)
            print("difference:", difference)
            if ratio > 1.5 and min(len(S0), len(S1)) > 5 and iteration < iter_limit:
                # Assign nodes' labels
                col = random.randint(0, 100)
                for i in S0:
                    # G.nodes(data=True)[i][label] = 100 - color
                    G.nodes(data=True)[i][label] = col
                
                col = random.randint(120, 220)    
                for i in S1:
                    # G.nodes(data=True)[i][label] = color - 100
                    G.nodes(data=True)[i][label] = col
                
                clustering_bqm(G.subgraph(S0), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
                clustering_bqm(G.subgraph(S1), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
            col = random.randint(0, 100)
            for i in G.nodes:
                G.nodes(data=True)[i][label] = col
            return response
        elif min(len(S0), len(S1)) > 5 and iteration < iter_limit:
            # Assign nodes' labels
            col = random.randint(0, 100)
            for i in S0:
                # G.nodes(data=True)[i][label] = 100 - color
                G.nodes(data=True)[i][label] = col
            
            col = random.randint(120, 220)    
            for i in S1:
                # G.nodes(data=True)[i][label] = color - 100
                G.nodes(data=True)[i][label] = col
            clustering_bqm(G.subgraph(S0), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
            clustering_bqm(G.subgraph(S1), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
        else:
            col = random.randint(0, 100)
            for i in G.nodes:
                G.nodes(data=True)[i][label] = col
            return response
    
    elif terminate_on == "once":
        col = random.randint(0, 100)
        for i in S0:
            G.nodes(data=True)[i][label] = col
        
        col = random.randint(120, 220)    
        for i in S1:
            G.nodes(data=True)[i][label] = col

    elif terminate_on == "iter_limit":
        if iteration < iter_limit:
            col = random.randint(0, 100)
            for i in S0:
                G.nodes(data=True)[i][label] = col
            
            col = random.randint(120, 220)    
            for i in S1:
                G.nodes(data=True)[i][label] = col
            
            clustering_bqm(G.subgraph(S0), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
            clustering_bqm(G.subgraph(S1), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, iter_limit)
    return

def clustering_bqm_2(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit, k, chain_strength):

    name_spec = ''.join([dirs["name"], "_", solver])   

    nodes_len = len(G.nodes)

    degrees = [val for (node, val) in G.degree()]
    degrees_sum = sum(degrees)
    degrees_mean = np.mean(degrees)
    
    weights = list(nx.get_edge_attributes(G, "weight").values())
    weights_sum = G.size(weight="weight")
    weights_mean = np.mean(weights)

    chain_strength = weights_mean * degrees_mean * 2
    # chain_strength = gamma_factor * edges_weights
    gamma = (weights_sum/nodes_len) * gamma_factor
    # gamma = gamma_factor/nodes_len
    print("gamma: ", gamma)
    print("chain_strength: ", chain_strength)

    # Initialize our Q matrix
    Q = defaultdict(int)
    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += k*G.get_edge_data(u, v)["weight"]
        Q[(v,v)] += k*G.get_edge_data(u, v)["weight"]
        Q[(u,v)] += k *-2*G.get_edge_data(u, v)["weight"]

    for i in G.nodes:
        Q[(i,i)] += gamma

    print("... Running on QPU ...")
    
    num_reads = 5000
    # chain_strength = 20

    if solver == "hybrid":
        sampler = LeapHybridSampler()
        response = sampler.sample_qubo(Q, label=name_spec)
    elif solver == "fixed_embedding":
        save = False
        try:
            a_file = open(dirs["embedding"])
            embedding = json.load(a_file)
            a_file.close()

            sub_embedding = dict((k, embedding[k]) for k in G.nodes if k in embedding)
            
            sampler = FixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'), sub_embedding)
            print("found embedding")
        except IOError:
            save = True
            print("generate new embedding")
            # sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))
            sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))

        response = sampler.sample_qubo(Q, label=name_spec, chain_strength=chain_strength, num_reads=num_reads, return_embedding=True)    
        
        if save:
            # embedding = sampler.properties['embedding']
            embedding = response.info['embedding_context']['embedding']
            a_file = open(dirs["embedding"], "w")
            json.dump(embedding, a_file)
            a_file.close()   
    elif solver == "embedding_composite":
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(Q, label=name_spec, chain_strength=chain_strength, num_reads=num_reads)

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
        
        if (i > 3):
            break
        i = i + 1

    
    label = "label" + str(iteration)
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    print("S0 length: ", len(S0))
    print("S1 length: ", len(S1))
    if terminate_on == "min_size":
        # Assign nodes' labels
        # col = random.randint(0, 100)
        for i in S0:
            G.nodes(data=True)[i][label] = 100 - color
            # G.nodes(data=True)[i][label] = col
        
        # col = random.randint(120, 220)    
        for i in S1:
            G.nodes(data=True)[i][label] = color - 100
            # G.nodes(data=True)[i][label] = col
        # write to the graph file
        # file_name = "clustring_" + str(iteration) + ".gexf"
        # nx.write_gexf(G, file_name)
        if(len(S0)>size_limit and len(S1)>size_limit):
            clustering_bqm_2(G.subgraph(S0), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, k, chain_strength)
            clustering_bqm_2(G.subgraph(S1), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, k, chain_strength)
    #to-do
    elif terminate_on == "conf":
        print("energies", response.record.energy[:10])
        # ratio = response.record.energy[0]/response.record.energy[3]
        difference = np.abs(response.record.energy[0]-response.record.energy[3])
        # print("ratio:", ratio)
        print("difference:", difference)
        if difference > 10 and min(len(S0), len(S1)) > 5:
            # Assign nodes' labels
            col = random.randint(0, 100)
            for i in S0:
                # G.nodes(data=True)[i][label] = 100 - color
                G.nodes(data=True)[i][label] = col
            
            col = random.randint(120, 220)    
            for i in S1:
                # G.nodes(data=True)[i][label] = color - 100
                G.nodes(data=True)[i][label] = col

            clustering_bqm_2(G.subgraph(S0), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, k, chain_strength)
            clustering_bqm_2(G.subgraph(S1), iteration+1, dirs, solver, gamma_factor, color+20, terminate_on, size_limit, k, chain_strength)

    elif terminate_on == "once":
        col = random.randint(0, 100)
        for i in S0:
            G.nodes(data=True)[i][label] = col
        
        col = random.randint(120, 220)    
        for i in S1:
            G.nodes(data=True)[i][label] = col
        
        return response
    return

def clustering_bqm_3(G, iteration, dirs, solver, gamma_factor, color, terminate_on, size_limit):

    name_spec = ''.join([dirs["name"], "_", solver]) 
    
    edges_weights = G.size(weight="weight")
    nodes_len = len(G.nodes)
    gamma = gamma_factor * edges_weights/nodes_len
    print("gamma: ", gamma)
    k = 8

    # Initialize our Q matrix
    Q = defaultdict(int)
    # Fill in Q matrix
    for u, v in G.edges:
        Q[(u,u)] += k*G.get_edge_data(u, v)["weight"]
        Q[(v,v)] += k*G.get_edge_data(u, v)["weight"]
        Q[(u,v)] += k *-2*G.get_edge_data(u, v)["weight"]

    bqm = BinaryQuadraticModel.from_qubo(Q)
    
    x = [str(n) for n in G.nodes()]
    
    c1 = [(x[int(n)], 1) for n in G.nodes()]
    bqm.add_linear_inequality_constraint(c1,
            lb = size_limit,
            ub = len(G.nodes)/6,
            lagrange_multiplier = gamma,
            label = 'c1_constraint')
    
    print("... Running on QPU ...")

    # sampler = EmbeddingComposite(DWaveSampler())
    # response = sampler.sample(bqm)
    response = hybrid.KerberosSampler().sample(bqm, max_iter=1, num_reads=1, qpu_reads=100, tabu_timeout=200, qpu_params={'label': 'Notebook - Hybrid Computing 1'})
    # if solver == "hybrid":
    #     sampler = LeapHybridSampler()
    #     response = sampler.sample(bqm, label=name_spec, time_limit=3.0)
    
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
        
        if (i > 3):
            break
        i = i + 1

    
    label = "label" + str(iteration)
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    print("S0 length: ", len(S0))
    print("S1 length: ", len(S1))

    col = random.randint(0, 100)
    for i in S0:
        G.nodes(data=True)[i][label] = col
    
    col = random.randint(120, 220)    
    for i in S1:
        G.nodes(data=True)[i][label] = col
    
    return response
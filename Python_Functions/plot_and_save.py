from unittest import result
import networkx as nx
from collections import defaultdict
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# dirs["img_in"]
def plot_and_save_graph_in(G, pos, dirs):
    plt.cla()
    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='solid', alpha=0.5, width=1)
    plt.savefig(dirs["img_in"], bbox_inches='tight')
    print("graph saved as: ", dirs["img_in"])

# dirs["img_out_bqm"] & dirs["graph_out_bqm"]
def plot_and_save_graph_out_bqm(G, pos, dirs):
    cut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]!=list(G.nodes[v].values())[-1]]
    uncut_edges = [(u, v) for u, v in G.edges if list(G.nodes[u].values())[-1]==list(G.nodes[v].values())[-1]]

    len(cut_edges)
    len(uncut_edges)

    # colors = [sum(list(y.values())) for x,y in G.nodes(data=True)]
    colors = [int(list(y.values())[-1]) for x,y in G.nodes(data=True)]

    # ------- plot and & output graph -------
    plt.cla()

    nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=G.nodes,  node_color=colors)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=1)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=1)

    plt.savefig(dirs["img_out_bqm"], bbox_inches='tight')

    nx.write_gexf(G, dirs["graph_out_bqm"])

# dirs["img_out_dqm"] & dirs["graph_out_dqm"]
def plot_and_save_graph_out_dqm(G, pos, dirs, sampleset):
    plt.cla()
    nx.draw(G, pos=pos, with_labels=False, node_color=list(sampleset.first.sample.values()), node_size=10, cmap=plt.cm.rainbow)                 
    plt.savefig(dirs["img_out_dqm"], bbox_inches='tight')

    lut = sampleset.first.sample
    nx.set_node_attributes(G, lut, name="label1")

    nx.write_gexf(G, dirs["graph_out_dqm"])

# dirs["img_out_cqm"] & dirs["graph_out_cqm"]
def plot_and_save_graph_out_cqm(G, pos, dirs, sampleset_cqm, num_of_clusters):
    sample = sampleset_cqm.first.sample

    clusters = [-1]*G.number_of_nodes()
    labels = defaultdict(int)

    for node in G.nodes:
        for p in range(num_of_clusters):
            if sample[f'v_{int(node)},{p}'] == 1:
                clusters[int(node)] = p
                labels[node] = p

    plt.cla()
    nx.draw(G, pos=pos, with_labels=False, node_color=clusters, node_size=10, cmap=plt.cm.rainbow)                 
    plt.savefig(dirs["img_out_cqm"], bbox_inches='tight')

    nx.set_node_attributes(G, labels, name="label1")
    nx.write_gexf(G, dirs["graph_out_cqm"])

# dirs["img_out_cqm"] & dirs["graph_out_cqm"]
def plot_and_save_graph_out_cqm_2(G, pos, dirs, sampleset_cqm, num_of_clusters):
    sample = sampleset_cqm.first.sample

    clusters = [-1]*G.number_of_nodes()
    labels = defaultdict(int)

    for node in G.nodes:
        for p in range(num_of_clusters):
            if sample[f'v_{int(G.nodes(data=True)[node]["subindex"])},{p}'] == 1:
                G.nodes()[node]["z_cluster"] = p
                clusters[int(G.nodes(data=True)[node]["subindex"])] = p
                labels[G.nodes(data=True)[node]["subindex"]] = p

    plt.cla()
    nx.draw(G, pos=pos, with_labels=False, node_color=clusters, node_size=10, cmap=plt.cm.rainbow)                 
    plt.savefig(dirs["img_out_cqm"], bbox_inches='tight')

    nx.set_node_attributes(G, labels, name="label1")
    nx.write_gexf(G, dirs["graph_out_cqm"])

# dirs["img_out_p1"] & dirs["graph_out_pru1"]
def plot_and_save_graph_out_mvc(G, pos, dirs):
    included_edges = [(u, v) for u, v in G.edges if (G.nodes[u]["label1"]==1 or G.nodes[v]["label1"]==1)]
    excluded_edges = [(u, v) for u, v in G.edges if (u, v) not in included_edges]
    
    # ------- plot and & output graph -------
    colors = [y["label1"] for x, y in list(G.nodes(data=True))]
    labels = [(x, y["label1"]) for x, y in list(G.nodes(data=True))]
    labels = dict(labels)

    plt.cla()
    nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=G.nodes, node_color=colors)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=5, font_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=excluded_edges, style='dashdot', alpha=0.5, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=included_edges, style='solid', width=1)

    plt.savefig(dirs["img_out_p1"], bbox_inches='tight')

    nx.write_gexf(G, dirs["graph_out_pru1"])

# directory: dirs["graph_out"] + <experiment> + /multi_eng/
def plot_and_save_graph_out_cqm_multi(G, pos, dir, sampleset_cqm, num_of_clusters, number_of_samples):
    results = [sample for sample in sampleset_cqm.samples()[:number_of_samples-1]]
    
    for i in range(len(results)):
        graph_name = dir + "/multi_eng/sample_number" + str(i)

        sample = results[i]
        clusters = [-1]*G.number_of_nodes()
        labels = defaultdict(int)

        for node in G.nodes:
            for p in range(num_of_clusters):
                if sample[f'v_{int(node)},{p}'] == 1:
                    clusters[int(node)] = p
                    labels[node] = p

        plt.cla()
        nx.draw(G, pos=pos, with_labels=False, node_color=clusters, node_size=10, cmap=plt.cm.rainbow)                 
        plt.savefig(graph_name + ".png", bbox_inches='tight')

        nx.set_node_attributes(G, labels, name="label1")
        nx.write_gexf(G, graph_name + ".gexf")

import networkx as nx

def read_graph(map_file_name):
    print(f"\n************Read {map_file_name}*************")
    G = nx.DiGraph()
    f = open("dataset/map_file/" + map_file_name, 'r')
    data = f.readlines()
    f.close()
    for idx, lines in enumerate(data):
        if (idx < 2):
            continue
        src, dest, dist, is_one_way = lines.split(" ")
        G.add_weighted_edges_from([(int(src), int(dest), float(dist))])
        if (int(is_one_way) == 0):
            G.add_weighted_edges_from([(int(dest), int(src), float(dist))])
    # 输出节点和边的个数
    print("Finish Reading Graph!")
    print(f"nodes:{len(G.nodes())}   edges:{len(G.edges())}")
    return G

G  = read_graph('test')
print(G[0][2]['weight'])
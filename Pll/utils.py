import time
import networkx as nx
import queue as Q
import numpy as np
from random import randint
from numba import njit
import pandas as pd
max_length = 999999999

def write_2_hop_count(map_file_name, hop_count):
    fileName = "dataset/hop_count/"+ map_file_name+"_hop_count.txt"
    f = open(fileName, 'w')
    f.write(str(list(hop_count)))
    f.close()

def write_order(map_file_name, vertex_order, mode):
    # order写入单独文件
    write_data = {}
    write_data[mode] = list(vertex_order)
    fileName = "dataset/order/"+map_file_name+"_order.txt"
    f = open(fileName,"a")
    f.write(str(write_data))
    # json.dumps()
    f.write("\n")
    f.close()
        
def load_order(map_file_name, mode):
    fileName = "dataset/order/"+map_file_name+"_order.txt"
    f = open(fileName,"r")
    data = f.readlines()
    f.close()
    if mode == 'degree' :
        order = np.array((eval(data[0]))[mode], dtype= np.int64)
    elif mode == 'betweenness':
        order = np.array((eval(data[1]))[mode], dtype= np.int64)
    elif mode == "hop_count":
        order = np.array((eval(data[2]))[mode], dtype= np.int64)
    elif mode == "user_define":
        fileName = "dataset/order/"+map_file_name+"_user_define_order.txt"
        f = open(fileName,"r")
        data = f.readline()
        order = np.array(eval(data), dtype= np.int64)
    return order

# 使用networkx读入图
def read_graph(map_file_name):
    print("\n************Read Graph*************")
    G = nx.DiGraph()
    f = open("dataset/map_file/" + map_file_name, 'r')
    data = f.readlines()
    f.close()
    for idx, lines in enumerate(data):
        if (idx < 2):
            continue
        src, dest, dist, is_one_way = lines.split(" ")
        G.add_weighted_edges_from([(int(src), int(dest), int(dist))])
        if (int(is_one_way) == 0):
            G.add_weighted_edges_from([(int(dest), int(src), int(dist))])
    # 输出节点和边的个数
    print("Finish Reading Graph!")
    print(f"nodes:{len(G.nodes())}   edges:{len(G.edges())}")
    return G

@njit
def query_for_BFS(src, dest, order2index, index):
        # print(src)
        # print(src,dest)
        shortest_dist = max_length
        src_index = order2index[src] # 3
        dest_index = order2index[dest] # 1
        # print(f"order2index:{self.order2index}")
        # print(f"vertex_order:{self.vertex_order}")
        
        if (src != dest):
            tmp = index[src_index,dest_index]
            if (tmp !=0):
                shortest_dist = tmp
            else:
                high_prior = min(src_index,dest_index)
                src_list = index[src_index,0:high_prior]   
                dest_list = index[0:high_prior,dest_index]
                src_list_non_zero = np.nonzero(src_list)[0]
                for hop_index in src_list_non_zero:
                    if (dest_list[hop_index]!=0):
                        curr_dist = src_list[hop_index] + dest_list[hop_index]
                        if curr_dist < shortest_dist:
                            shortest_dist = curr_dist
        return shortest_dist

@njit
def query(src, dest, order2index, index):
    shortest_dist = max_length
    src_index = order2index[src] # 3
    dest_index = order2index[dest] # 1
    if (src != dest):
        tmp = index[src_index,dest_index]
        if (tmp !=0):
            shortest_dist = tmp
        else:
            high_prior = min(src_index,dest_index)
            src_list = index[src_index,0:high_prior]   
            dest_list = index[0:high_prior,dest_index]
            src_list_non_zero = np.nonzero(src_list)[0]
            for hop_index in src_list_non_zero:
                if (dest_list[hop_index]!=0):
                    curr_dist = src_list[hop_index] + dest_list[hop_index]
                    if curr_dist < shortest_dist:
                        shortest_dist = curr_dist
    else:
        shortest_dist = 0
    return shortest_dist

def gen_degree_base_order(G):
    nodes_list = np.array(sorted(G.degree, key=lambda x: x[1], reverse=True))
    vertex_order = generate_order_for_BFS(nodes_list, G)
    return vertex_order

def gen_betweeness_base_order(G):
    nodes_list = nx.betweenness_centrality(G, weight="weight")
    nodes_list = np.array(sorted(nodes_list.items(), key=lambda item:item[1], reverse = True))
    # print(nodes_list)
    vertex_order = generate_order_for_BFS(nodes_list, G)
    return vertex_order

@njit
def gen_2_hop_base_order(nNodes, order2index, index: np.ndarray, vertex_order):
    hop_count = np.zeros(nNodes, dtype= np.int64)
    for src in vertex_order:
        tmp_count = cal_2_hop(nNodes, src, order2index, index, vertex_order)
        hop_count = hop_count + tmp_count
    vertex_order = np.argsort(hop_count)[::-1]
    return vertex_order, hop_count

@njit
def cal_2_hop(nNodes, src, order2index:np.ndarray, index: np.ndarray, vertex_order):
    hop_count = np.zeros(nNodes, dtype= np.int64)
    # hop_node_list = np.empty(0, dtype= np.int64)
    src_index = order2index[src]
    src_to_others = index[src_index, :src_index]
    non_zero_src_indices = np.nonzero(src_to_others)[0]
    src_to_others = src_to_others[non_zero_src_indices]
    for dest_index in range(nNodes):
        if src_index != dest_index:
        # 判断是否存在src到dest的一跳
            if index[src_index, dest_index] > 0:
                hop_count[src] += 1
                hop_count[vertex_order[dest_index]] += 1
            else:
                #判断是否存在src到dest的二跳
                others_to_dest = index[non_zero_src_indices, dest_index]

                others_to_dest_non_zero_indices = np.nonzero(others_to_dest)[0]
                others_to_dest_new = others_to_dest[others_to_dest_non_zero_indices]
                src_to_others_new = src_to_others[others_to_dest_non_zero_indices]

                two_hops = src_to_others_new + others_to_dest_new
                if len(two_hops) != 0:
                    min_indices = np.argwhere(two_hops == np.min(two_hops)).flatten()
                    for item in non_zero_src_indices[others_to_dest_non_zero_indices[min_indices]]:
                        hop_count[vertex_order[item]] += 1 
    return hop_count
    
def generate_order_for_BFS(nodes_list, G):
    result = np.empty((0,), dtype=np.int64)
    nNodes = len(G.nodes())
    for i in range(nNodes):
        result = np.append(result, int(nodes_list[i][0]))
    return result
    
# 判断是否需要剪枝
@njit
def need_to_expand(src, dest, dist = -1, order2index = [], index = []):
    our_result = query_for_BFS(src, dest, order2index, index)
    if (our_result <= dist):
        return False
    return True

def build_index(G, vertex_order):
    start = time.perf_counter()
    nNodes = len(G.nodes())
    order2index = np.zeros(nNodes,dtype= np.int64)
    for i in range(nNodes):
        order2index[vertex_order[i]] = i
    order2index = order2index
   
    index = np.zeros((nNodes,nNodes),dtype= np.int64)
    pq = Q.PriorityQueue()
    has_process = np.zeros(nNodes)
    i = 0
    for cur_node in vertex_order:
        i += 1
        # Calculate Forward
        if (i%1000 == 0) :
            print("Caculating %s (%d/%d) forward ... " % (cur_node, i, nNodes))
        pq.put((0, cur_node))
        # 把所有点是否剪枝记为0
        has_process[:] = 0
        while (not pq.empty()):
            cur_dist, src = pq.get()
            if (has_process[src] or order2index[cur_node] > order2index[src] or not need_to_expand(cur_node, src, cur_dist, order2index, index)):
                has_process[src] = 1
                continue
            has_process[src] = 1
            index[order2index[cur_node], order2index[src]] = cur_dist
            edges = G.out_edges(src)   
            for _, dest in edges:
                weight = G.get_edge_data(src, dest)['weight']
                if (has_process[dest]):
                    continue
                pq.put((cur_dist + weight, dest))

        # Calculate Backward
        pq.put((0, cur_node))
        has_process[:] = 0
        # print(has_process)
        while (not pq.empty()):
            cur_dist, src = pq.get()
            # print("Pop: (%s %d)"%(src,cur_dist))
            if (has_process[src] or order2index[cur_node] > order2index[src] or not need_to_expand(src, cur_node, cur_dist, order2index, index)):
                has_process[src] = 1
                continue
            has_process[src] = 1
            index[order2index[src], order2index[cur_node]] = cur_dist
            edges = G.in_edges(src)
            for dest, _ in edges:
                weight = G.get_edge_data(dest, src)['weight']
                if (has_process[dest]):
                    continue
                pq.put((cur_dist + weight, dest))
    end = time.perf_counter()
    print(f'finish building index, time cost: {end-start:.4f}')

    return order2index, index, (end-start)

def check(order2index, index, G):
    nNodes = len(G.nodes())
    for i in range(100):
        src = randint(0,nNodes-1)
        dest = randint(0,nNodes-1)
        dist1 = query(src,dest,order2index, index)
        if nx.has_path(G, src, dest):
            dist2 = nx.dijkstra_path_length(G, src, dest)
        else:
            dist2 = 999999999
        if (dist1!=dist2):
            print("error!")
            exit()
        # print(f"src:{src_index}->dest:{dest_index} : {dist1} {dist2}")
    print("succeed!")

def query_100K(order2index, index, G):
    start = time.perf_counter()
    nNodes = len(G.nodes())
    for i in range(100000):
        src = randint(0,nNodes-1)
        dest = randint(0,nNodes-1)
        _ = query(src,dest,order2index, index)
    end = time.perf_counter()
    print(f'finish query_100K, time cost: {(end-start):.4f}')
    return (end-start)

def count_avg_label_size(nNodes, index):
    nonzero_count = np.count_nonzero(index)
    label_size = nonzero_count / nNodes
    return label_size

def output_to_excel(map_file_name, gen_order_time_list, BFS_time_list, query_time_100K_list, avg_label_size_list):
    total_time_list = [BFS_time_list[i] + gen_order_time_list[i] for i in range(len(BFS_time_list))]
    df = pd.DataFrame(columns=["Degree","betweenness",'betweenness-2-hop-count'])
    df.loc[len(df.index)] = gen_order_time_list
    df.loc[len(df.index)] = BFS_time_list
    df.loc[len(df.index)] = query_time_100K_list
    df.loc[len(df.index)] = avg_label_size_list
    df.loc[len(df.index)] = total_time_list
    df.rename(index = {0:"gen_order_time",1:"BFS_time",2:"query_time_100K",3:"avg_label_size",4:"each_total_time"},inplace=True)
    print(df)

    excel_file_name = "dataset/excel/"+ map_file_name+".xlsx"
    writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
    df.to_excel(writer,'Sheet1')
    worksheet = writer.sheets['Sheet1']
    worksheet.set_column('A:G', 25)
    total_time = df.loc['each_total_time',"Degree"] + df.loc['each_total_time',"betweenness"] + df.loc['each_total_time',"betweenness-2-hop-count"]
    print(f"total_time: {total_time}")
    worksheet.write(7,0,total_time)
    writer.save()
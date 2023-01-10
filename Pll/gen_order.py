from utils import *
import time
import getopt
import sys

# 根据输入的mode采取不同的策略构建节点的order
def gen_order( G, mode = "degree", nNodes = 0, order2index = [], index = [], vertex_order_bet = [], map_file_name = ''):
    start = time.perf_counter()
    if (mode == "degree"):
        print("\n*************Degree****************")
        vertex_order = gen_degree_base_order(G)
    if (mode == "betweenness"):
        print("\n*************Betweenness***********")
        vertex_order = gen_betweeness_base_order(G, map_file_name)
    if (mode == "hop_count"):
        print("\n*************2-hop*****************")
        vertex_order, hop_count = gen_2_hop_base_order(nNodes, order2index, index, vertex_order_bet)
        end = time.perf_counter()
        print(f"finish generating order, time cost: {end-start:.4f}")
        return vertex_order,end-start, hop_count
    end = time.perf_counter()
    print(f"finish generating order, time cost: {end-start:.4f}")
    return vertex_order,end-start

# 从命令行提取参数
def fetch_map_name():
    help_msg = "python gen_order.py -i [input_file] -m mode"
    try:
        options, _ = getopt.getopt(sys.argv[1:], "-i:-m:", ["input=","mode="])
        map_file_name = options[0][1]
        mode = options[1][1]

        return map_file_name, mode
    except:
        print(help_msg)
        exit()

if __name__== "__main__":
    map_file_name, mode = fetch_map_name()
    G = read_graph(map_file_name)
    nNodes = len(G.nodes())
    if mode == 'hop_count':
        vertex_order, _ = gen_order(G, mode="betweenness")
        order2index, index, _ = build_index(G, vertex_order)
        vertex_order, gen_order_time, hop_count = gen_order(G, 'hop_count', nNodes, order2index, index, vertex_order)
        # print(hop_count)
    else:
        vertex_order, gen_order_time = gen_order(G, mode= mode)   
    write_order(map_file_name, vertex_order, mode)
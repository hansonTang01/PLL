from utils import *
from gen_order import gen_order
import getopt
import sys

modes = ['degree', 'betweenness', 'hop_count']

# 从命令行提取参数
def fetch_map_name():
    help_msg = "python script.py -i [input_file]"
    try:
        options, _ = getopt.getopt(sys.argv[1:], "-i:", ["input="])
        map_file_name = options[0][1]
        return map_file_name
    except:
        print(help_msg)
        exit()

for i in range(100):
    BFS_time_list = []
    query_time_100K_list = []
    gen_order_time_list = []
    avg_label_size_list = []
    # map_file_name = fetch_map_name()
    map_file_name = str(i+1) + '.map'
    G = read_graph(map_file_name)
    nNodes = len(G.nodes())

    for mode in modes:
        if mode == 'hop_count':
            vertex_order = load_order(map_file_name, mode ="betweenness")
            vertex_order, gen_order_time, hop_count = gen_order(G, 'hop_count', nNodes, order2index, index, vertex_order)
            write_2_hop_count(map_file_name, hop_count)
        else:
            vertex_order, gen_order_time = gen_order(G, mode= mode, map_file_name= map_file_name)  

        write_order(map_file_name, vertex_order, mode)
        order2index, index, BFS_time = build_index(G, vertex_order)
        query_time_100K = query_100K(order2index, index, G)
        avg_label_size = count_avg_label_size(nNodes, index)

        BFS_time_list.append(float("%.2f"%BFS_time))
        query_time_100K_list.append(float("%.2f"%query_time_100K))
        gen_order_time_list.append(float("%.2f"%gen_order_time))
        avg_label_size_list.append(float("%.2f"%avg_label_size))
    output_to_excel(map_file_name,gen_order_time_list, BFS_time_list, query_time_100K_list, avg_label_size_list)
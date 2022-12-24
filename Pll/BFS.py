from utils import build_index
from utils import read_graph
from utils import load_order
import numpy as np
import getopt
import sys

# def BFS(G,vertex_order):
#     order2index, index = build_index(G,vertex_order)
#     return order2index, index

# 从命令行提取参数
def fetch_map_name():
    help_msg = "python BFS.py -i [input_file] -m [mode]"
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
    order = load_order(map_file_name, mode)
    _,_ = build_index(G, vertex_order= order)


    
    


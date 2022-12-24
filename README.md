# PrunedLandmarkLabeling

This repo implements PrunedLandmarkLabeling that calculates the shortest distance in a map, and we make some optimizations for the algorithm.

该repo实现了PrunedLandmarkLabeling，用于计算地图中的最短距离，并对算法的实现进行了一些优化。

## 说明

### Prerequisites
如果需要运行这个仓库的代码，需要以下包:
- numba
- numpy
- networkx
- pandas
### 模块
本仓库分为以下五个模块：
- utils.py：该文件包含了生成order、BFS构建label、计算俩跳计数、输出结到excel表等所有的函数。
- BFS.py: 给定一个order，基于该order进行BFS，注意：此order可以是之前基于degree或者其他centrality得到的order，也可以是user-define的。
- gen_order.py: 选择一种centrality(degree,betweenness or hop_count)，给出节点的order
- script.py: 将所有流程聚在一起的脚本，一键运行
- dataset文件夹： 数据集，包含了几个部分：
  - excel：结果转换为excel表存放在这里面
  - hop_count: 俩跳计数的结果放在里面
  - order：生成的节点order放在里面

### 运行指令(以macau为例)

**script.py**
```bash
python script.py -i [input_file]
e.g: python script.py -i macau
# 完成从基于各种策略生成order，到BFS，到将结果输出到excel的所有流程
```

**gen_order.py**
```bash
python gen_order.py -i [input_file] -m mode
e.g: python gen_order.py -i macau -m degree
# degree可以换成其他策略
```

**BFS.py**
```bash
python BFS.py -i [input_file] -m [mode]
e.g: python BFS.py -i macau -m degree
# degree可以换成其他策略
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/21 5:04
# @Author : {ZM7}
# @File : read_line.py
# @Software: PyCharm


import datetime
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import argparse
from pandas.tseries.offsets import MonthEnd as M
from sklearn.model_selection import train_test_split
path_log = './Dataset/login.txt'
path_pay = './Dataset/pay.txt'
item_path = './Dataset/item_sample_after.txt'
user_path = './Dataset/user_sample_after.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--sample', action='store_true', help='dataset')
parser.add_argument('--method', type=str, default='line', help='line/deepWalk/node2vec')
parser.add_argument('--weight', action='store_true', help='weight')
parser.add_argument('--write', action='store_true', help='write or read')
opt = parser.parse_args()
print(opt)


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def write_seq(opt):
    if opt.sample:
        path = 'sample'
    else:
        path = 'all_data'
    aliases_dict, all_seq = pickle.load(open('./Dataset/' + path + '/all_train.pickle', 'rb'))
    if opt.weight:
        g = build_graph(all_seq)
        fp = open('./Dataset/' + path + '/weight.txt', 'w+')
        for edge in g.edges:
            fp.write(edge[0])
            fp.write(' ')
            fp.write(edge[1])
            fp.write(' ')
            fp.write(g.get_edge_data(edge[0], edge[1])['weight'])
            fp.write('\n')
        fp.close()

    else:
        fp = open('./Dataset/' + path + '/no_weight.txt', 'w+')
        for i in range(len(all_seq)):
            for j in range(len(all_seq[i])-1):
                fp.write(str(all_seq[i][j]))
                fp.write(' ')
                fp.write(str(all_seq[i][j+1]))
                fp.write('\n')
        fp.close()
    return


def read_embedding(opt):
    if opt.sample:
        path = 'sample'
    else:
        path = 'all_data'
    aliases_dict, _ = pickle.load(open('./Dataset/' + path + '/all_train.pickle', 'rb'))
    with open('./Dataset/'+path+'/'+opt.method+'.txt', 'rb') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = np.array(str(line, encoding="utf-8").split(), dtype=int)
                embedding = np.zeros((dim[0]+1, dim[1]))
            else:
                temp = str(line, encoding="utf-8").split()
                embedding[aliases_dict[temp[0]]] = np.array(temp[1:], dtype=float)
        np.save('./Dataset/' + path + '/'+opt.method+'_embedding.npy', embedding)
if __name__ == '__main__':
    if opt.write:
        write_seq(opt)
    else:
        read_embedding(opt)

#python -m openne --method  line --input ./Dataset/all_data/weight.txt  --graph-format edgelist  --output ./Dataset/all_data/line.txt --directed --weighted --representation-size=100 --epochs=40
#python -m openne --method  deepWalk --input ./Dataset/all_data/weight.txt  --graph-format edgelist  --output ./Dataset/all_data/deepwalk.txt --directed -weighted --representation-size=100 --epochs=40


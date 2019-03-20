#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/10 10:34
# @Author : {ZM7}
# @File : data_process_fnn.py
# @Software: PyCharm
import datetime
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
from pandas.tseries.offsets import MonthEnd as M
from sklearn.model_selection import train_test_split
import networkx as nx

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))


#根据序列构建图用于line/deepwalk/node2vec
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

#读取原始数据
def load_data(path_log, path_pay):
    datalist = []
    datapay=[]
    with open(path_log) as f:
        for line in f:
            entry = line.split()
            datalist.append([int(entry[0]), entry[1],entry[2],int(entry[3]),float(entry[4])])

    with open(path_pay) as g:
        for line in g:
            entry = line.split()
            datapay.append([int(entry[0]), entry[1],entry[2],float(entry[3])])
    return (datalist, datapay)

#筛选数据
def data_filter(data):
    datalog, datapay = data
    datalog = pd.DataFrame(datalog, columns=['time', 'user', 'app', 'times', 'duration'])
    datapay = pd.DataFrame(datapay, columns=['time', 'user', 'app', 'value'])
    # 筛掉每天登陆时长大于24小时的
    datalog = datalog[(datalog['duration'] < 3600 * 24)&(datalog['duration'] > 30)]
    data_all = pd.merge(datalog, datapay, on=['user', 'time', 'app'], how='outer').fillna(-1)
    # 判断核心APP阈值
    val_th = np.percentile(data_all[data_all['value'] > 0]['value'].values, 90)  # 充值阈值
    dur_th = np.percentile(data_all[data_all['duration'] > 0]['duration'].values, 90)  # 登陆时间阈值
    # 判断核心app
    mask1 = (data_all['value'] > val_th)
    mask2 = (data_all['duration'] > dur_th)
    data_all['ker'] = (mask1 | mask2).astype(int)
    return data_all


def libsvm_2_coo(libsvm_data, max_index, min_index):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend([i-min_index for i in x])
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=(n, max_index-min_index+1))


def save_sparse(file_name):
    X = []
    D = []
    item = []
    dim = 0
    mi = np.inf
    with open(file_name, 'rb') as f:
        for line in f.readlines():
            X_i = []
            D_i = []
            if len(line.split()) == 1:
                continue
            tem = str(line, encoding="utf-8").split()
            item.append(tem[0])
            for x in tem[1:]:
                if len(x) == 0:
                    continue
                else:
                    X_i.append(int(x.split(':')[0]))
                    D_i.append(float(x.split(':')[1]))
            if len(X_i) > 0:
                dim = max(dim, max(X_i))
                mi = min(mi, min(X_i))
                X.append(X_i)
                D.append(D_i)
    X = libsvm_2_coo(zip(X, D), dim, mi).tocsr()
    item_dic = dict(zip(item, X))
    return item_dic

#读取用户和APP信息保存成稀疏格式
def save_user_item(file_user, file_app):
    user_dic = save_sparse(file_user)
    app_dic = save_sparse(file_app)
    save_pickle(user_dic, './Dataset/user_pro.pickle')
    save_pickle(app_dic, './Dataset/item_pro.pickle')


#生成line所用的序列, 构造边的权重TXT，给每个APP编号
def generate_seq(path_log, path_pay, sample, win=4):
    start = datetime(2016, 7, 1)
    end = datetime(2018, 1, 1)
    data = load_data(path_log, path_pay)
    data = data_filter(data)
    if sample:
        path = 'sample'
    else:
        path = 'all_data'
    user_dic= pickle.load(open('./Dataset/user_pro.pickle', 'rb'))
    app_dic= pickle.load(open('./Dataset/item_pro.pickle', 'rb'))
    data = data[data['user'].isin(user_dic.keys()) & data['app'].isin(app_dic.keys())]
    if sample:
        data = data.sample(frac=0.001, random_state=1)
    train_user, test_user = train_test_split(data['user'].unique(), test_size=0.2, random_state=0)
    train_data = data[data['user'].isin(train_user)]

    train_data = train_data.sort_values(['user', 'time', 'duration']).drop_duplicates(['user', 'time'], keep='last')
    train_data = train_data.reset_index(drop=True)
    train_data['time'] = pd.to_datetime(train_data['time'].astype('str'))

    def sel(df):
        if len(df) > 3:
            return df['app'].values.tolist()
        else:
            del df
    starttime = start
    spl = start+win*M()
    endtime = start+(win+1)*M()
    if endtime > end:
        print('the data is not enough')
        exit(0)

    all_seq = []
    #print('start cutting train set......')
    while endtime < end:
        all_seq.extend(train_data[(train_data['time'] > starttime) &
                                  (train_data['time'] <= endtime)].groupby('user').apply(sel).dropna().values.tolist())
        starttime = starttime + M()
        spl = spl + M()
        endtime = endtime + M()
    all_app = set()
    for i in range(len(all_seq)):
        all_app.update(all_seq[i])
    app_num = len(all_app)
    aliases_dict = dict(zip(all_app, range(1, app_num+1)))      #从1开始编号
    #根据所有序列构建图
    g = build_graph(all_seq)
    fp = open('./Dataset/' + path + '/weight.txt', 'w+')   #保存图每条边和权重
    for edge in g.edges:
        fp.write(edge[0])
        fp.write(' ')
        fp.write(edge[1])
        fp.write(' ')
        fp.write(str(g.get_edge_data(edge[0], edge[1])['weight']))
        fp.write('\n')
    fp.close()
    save_pickle(aliases_dict, './Dataset/' + path + '/aliases_dict.pickle') #每个APP的编号


def read_embedding(opt):
    aliases_dict= pickle.load(open('./Dataset/all_data/aliases_dict.pickle', 'rb'))
    with open('./Dataset/all_data/'+opt.method+'.txt', 'rb') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = np.array(str(line, encoding="utf-8").split(), dtype=int)
                embedding = np.zeros((dim[0]+1, dim[1]))
            else:
                temp = str(line, encoding="utf-8").split()
                embedding[aliases_dict[temp[0]]] = np.array(temp[1:], dtype=float)
        #np.save('./Dataset/all_data/'+opt.method+'_embedding.npy', embedding)
        return embedding


#生成FNN所需的序列和target
def fnn_seq(path_log, path_pay, sample, win=4):
    print('start generating fnn data......   ', datetime.now())
    start = datetime(2016, 7, 1)
    end = datetime(2018, 1, 1)
    data = load_data(path_log, path_pay)
    data = data_filter(data)
    # if sample:
    #     path = 'sample'
    # else:
    #     path = 'all_data'
    user_dic = pickle.load(open('./Dataset/user_pro.pickle', 'rb'))
    app_dic = pickle.load(open('./Dataset/item_pro.pickle', 'rb'))
    data = data[data['user'].isin(user_dic.keys()) & data['app'].isin(app_dic.keys())]
    if sample:
        data = data.sample(frac=0.001, random_state=1)
    train_user, test_user = train_test_split(data['user'].unique(), test_size=0.2, random_state=0)
    train_data = data[data['user'].isin(train_user)]

    train_data = train_data.sort_values(['user', 'time', 'duration']).drop_duplicates(['user', 'time'], keep='last')
    train_data = train_data.reset_index(drop=True)
    train_data['time'] = pd.to_datetime(train_data['time'].astype('str'))

    def sel_seq(df):
        if len(df) > 0:
            return df['app'].values.tolist()
        else:
            return 0

    def sel_tar(df):
        temp = df[df['ker']>0]
        if len(temp) > 0:
            return df['user'].unique().tolist(), temp['app'].unique().tolist()
        else:
            del df

    def sel(df):
        if len(df) > 3:
            return df['app'].values.tolist()
        else:
            del df

    #生成训练集
    starttime = start
    spl = start+win*M()
    endtime = start+(win+1)*M()
    if endtime > end:
        print('the data is not enough')
        exit(0)

    train_seq = []
    all_seq = []
    #print('start cutting train set......')
    while endtime < end:
        #print('start time:', starttime,'----','cut time:', spl, '----', 'end time:', endtime)
        train_tem_seq = train_data[(train_data['time'] > starttime) & (train_data['time'] <= spl)]
        train_tem_ = pd.DataFrame(train_tem_seq.groupby('user').apply(sel).dropna(), columns=['seq'])
        t1 = starttime
        for i in range(win):
            t2 = t1+M()
            temp = pd.DataFrame(train_tem_seq[(train_tem_seq['time'] > t1) & (train_tem_seq['time'] <= t2)]
                             .groupby('user').apply(sel_seq).dropna(), columns=[str(i)])
            train_tem_ = pd.merge(train_tem_, temp, how='left', left_index=True, right_index=True)
            t1 = t2
        train_tem_ = train_tem_.fillna(0)
        train_tem_tar = train_data[(train_data['time'] > spl) & (train_data['time'] <= endtime)]
        train_tem_tar = pd.DataFrame(train_tem_tar.groupby('user').apply(sel_tar).dropna(), columns=['tar'])
        all = train_tem_.merge(train_tem_tar, left_index=True, right_index=True)
        for i in range(len(all)):
            po_set = np.setdiff1d(all.iloc[i]['tar'][1], all.iloc[i]['seq'])
            if len(po_set) > 0:
                for ne in np.unique(all.iloc[i]['seq']):
                    train_seq.append([all.iloc[i][['0', '1', '2', '3']].values.tolist(),
                                      all.iloc[i]['tar'][0][0], ne, 0])   #序列、用户、target_app、标签
                for po in po_set:
                    train_seq.append([all.iloc[i][['0', '1', '2', '3']].values.tolist(),
                                      all.iloc[i]['tar'][0][0], po, 1])
        all_seq.extend(train_data[(train_data['time'] > starttime) &
                                  (train_data['time'] <= endtime)].groupby('user').apply(sel).dropna().values.tolist())
        starttime = starttime + M()
        spl = spl + M()
        endtime = endtime + M()

    all_app = set()
    for i in range(len(all_seq)):
        all_app.update(all_seq[i])

    test_data = data[data['user'].isin(test_user) & data['app'].isin(list(all_app))]
    test_data = test_data.sort_values(['user', 'time', 'duration']).drop_duplicates(['user', 'time'], keep='last')
    test_data = test_data.reset_index(drop=True)
    test_data['time'] = pd.to_datetime(test_data['time'].astype('str'))
    #生成测试集
    test_seq = []

    starttime = start
    spl = start+win*M()
    endtime = start+(win+1)*M()
    #print('start cutting test set......')
    while endtime < end:
        #print('start time:', starttime,'----','cut time:',spl, '----','end time:', endtime)
        test_tem_seq = test_data[(test_data['time'] > starttime) & (test_data['time'] <= spl)]
        test_tem_ = pd.DataFrame(test_tem_seq.groupby('user').apply(sel).dropna(), columns=['seq'])
        t1 = starttime
        for i in range(win):
            t2 = t1 + M()
            temp = pd.DataFrame(test_tem_seq[(test_tem_seq['time'] > t1) & (test_tem_seq['time'] <= t2)]
                             .groupby('user').apply(sel_seq).dropna(), columns=[str(i)])
            t1 = t2
            test_tem_ = pd.merge(test_tem_, temp, how = 'left', left_index=True, right_index=True)
        test_tem_ = test_tem_.fillna(0)
        test_tem_tar = test_data[(test_data['time'] > spl) & (test_data['time'] <= endtime)]
        test_tem_tar = pd.DataFrame(test_tem_tar.groupby('user').apply(sel_tar).dropna(), columns=['tar'])
        all_te = test_tem_.merge(test_tem_tar, left_index=True, right_index=True)
        for i in range(len(all_te)):
            po_set = np.setdiff1d(all_te.iloc[i]['tar'][1], all_te.iloc[i]['seq'])
            if len(po_set) > 0:
                for ne in np.unique(all_te.iloc[i]['seq']):
                    test_seq.append([all_te.iloc[i][['0', '1', '2', '3']].values.tolist(),
                                      all_te.iloc[i]['tar'][0][0], ne, 0])   #序列、用户、target_app、标签
                for po in po_set:
                    test_seq.append([all_te.iloc[i][['0', '1', '2', '3']].values.tolist(),
                                      all_te.iloc[i]['tar'][0][0], po, 1])
        starttime = starttime + M()
        spl = spl + M()
        endtime = endtime + M()
    return train_seq, test_seq
    # save_pickle(train_seq, './Dataset/' + path + '/fnn_train.pickle')
    # save_pickle(test_seq, './Dataset/' + path + '/fnn_test.pickle')






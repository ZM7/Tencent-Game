#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/11/4 11:55
# @Author : {ZM7}
# @File : data_prepare.py
# @Software: PyCharm
import argparse
from data_process_fnn_concat import save_user_item, generate_seq, read_embedding
path_log = './Dataset/login.txt'
path_pay = './Dataset/pay.txt'
item_path = './Dataset/item_sample_after.txt'
user_path = './Dataset/user_sample_after.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--sample', action='store_true', help='dataset')
parser.add_argument('--method', type=str, default='line', help='line/deepWalk/node2vec')
parser.add_argument('--user_item', action='store_true', help='read user and item')
parser.add_argument('--all_seq', action='store_true', help='generate graph seq')
parser.add_argument('--read_embedding', action='store_true', help='read the line embedding')
opt = parser.parse_args()


if __name__ == '__main__':
    if opt.sample:
        print('sample data')
    else:
        print('all data')
    if opt.user_item:
        save_user_item(user_path, item_path)
    if opt.all_seq:
        generate_seq(path_log, path_pay, opt.sample, win=4)
    # if opt.read_embedding:
    #     read_embedding(opt)
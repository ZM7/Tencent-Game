#!/usr/bin/env bash
echo "prepare data"
python -u  data_prepare.py --user_item --all_seq>log &&
echo "line embedding"
CUDA_VISIBLE_DEVICES=0 python -m openne --method  line\
 --input ./Dataset/all_data/weight.txt\
 --graph-format edgelist \
 --output ./Dataset/all_data/line.txt\
 --directed --weighted --representation-size=100 --epochs=40>log && 
CUDA_VISIBLE_DEVICES=0  nohup python -u main.py>test_&
CUDA_VISIBLE_DEVICES=0  nohup python -u main.py --baseline>test_base_&


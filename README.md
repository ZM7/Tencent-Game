# Tencent-Game
Core APP recommendation

1.数据准备
data_prepare.py 
--user_item 读取user和item存储
--all_seq   构建序列line所需的图，生成边和权重，保存在文件weigh.txt中；并且给所有APP编号，保存在aliases_dict.pickle中

2.使用line，得到APP的嵌入表达
line的库可以使用：https://github.com/thunlp/openne
python -m openne --method  line\
 --input ./Dataset/all_data/weight.txt\                #根据序列得到的图的边和权重文件weight.txt
 --graph-format edgelist \
 --output ./Dataset/all_data/line.txt\                 #将line所学的表达存到line.txt
 --directed --weighted --representation-size=100 --epochs=40>log &&            #各种参数

 
3.python main.py 
--baseline       #不加line的嵌入表达
 
直接运行bash demo.sh
将依次执行数据读取，训练line，训练FNN。



其中：
data_process_fnn_concat.py中包含数据切割和数据读取的代码：

load_data(path_log, path_pay):        path_log，path_pay分别为登录数据和支付数据路径
data_filter()：                       
筛选规则：
（1）将每天登录时长大于24小时和小于30秒的数据去掉；
（2）将登录时长或充值金额占所有时长和充值金额前10%的APP为核心APP。
save_user_item() :                   提出筛选后的user和item的稀疏特征保存
generate_seq():                      
1.构建序列line所需的图，生成边和权重，保存在文件weigh.txt中 \\

2.给所有APP编号，保存在aliases_dict.pickle中\\
read_embedding()：                    
读取line生成的嵌入，保存在line_embedding中
fnn_seq()：    
根据登录数据和支付数据，生成fnn所需要的序列和目标。按照user的8:2进行分割，80%user作为训练，20%user作为测试集；
序列按照4:1进行滑动切割。每个月的序列单独保存，用于后续月份的拼接。
滑动窗口为1个月。生成的测试集序列中的app保证都在测试集序列中出现过（程序已经进行了筛选）
训练集和训练集的target都是当月的核心app且不在历史序列（前4个月），负样本为历史序列（前4个月）中的样本。
			   
未使用line的嵌入特征AUC：   0.729
使用了line嵌入特征AUC:      0.868

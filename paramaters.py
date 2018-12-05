#定义超参数
recurrent_cell_size = 128 #GRU cell中的神经元个数

D = 50 #每个词向量的长度

learning_rate = 0.005 #学习率

input_p,output_p = 0.5,0.5 #随机失活的概率

batch_size = 128#每次处理的问题数量

passes = 4 #阅读句子的遍数

ff_hidden_size = 256#前向传播层的大小

weight_decay = 0.00000001 #权重衰减率 正则化

training_iterations_count = 400000#输出结果前训练的次数

display_step = 100#每次有效性检查的训练次数

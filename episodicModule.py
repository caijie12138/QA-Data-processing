import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from inputModule import *
from questionModule import *

#DMN与其他网络最大的不同之处在于，它会多次阅读输入句子，每次只注意句子的fact表示中的一个子集。
size = tf.stack([tf.constant(1),tf.shape(cs)[1], tf.constant(1)])#拼接[1,?,1] 确保当前的记忆会随着事实的维度广播 tf.stack将一组R维张量变为R+1维张量

re_q = tf.tile(tf.reshape(q,[-1,1,recurrent_cell_size]),size) #把问题的维度扩展的和记忆一样
#tf.tile 行复制和列复制

#接着我们把输入的事实通过一个GRU来给予权重（通过相应的注意力常数来给予），从而修改了现有记忆。
#为了避免当上下文比矩阵的长度小的时候向现有记忆里加入不正确的信息，我们创建了一掩盖层，当事实不存在的时候就根本不去注意它（例如，获取了相同的现有记忆）
output_size = 1 #注意力机制的最终输出 

attend_init = tf.random_normal_initializer(stddev = 0.1)#返回一个生成具有正态分布的张量的初始化器。

w_1 = tf.get_variable("attend_w1", [1,recurrent_cell_size*7, recurrent_cell_size], 
                      tf.float32, initializer = attend_init)
w_2 = tf.get_variable("attend_w2", [1,recurrent_cell_size, output_size], 
                      tf.float32, initializer = attend_init)

b_1 = tf.get_variable("attend_b1", [1, recurrent_cell_size], 
                      tf.float32, initializer = attend_init)
b_2 = tf.get_variable("attend_b2", [1, output_size], 
                      tf.float32, initializer = attend_init)

# Regulate all the weights and biases
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_1))#tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_1))
#tf.nn.l2_loss这个函数的作用是利用 L2 范数来计算张量的误差值，
#但是没有开方并且只取 L2 范数的值的一半，具体如下：tf.nn.l2_loss(t, name=None)  output = sum(t ** 2) / 2
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_2))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_2))

#我们通过构建每个事实、现有记忆和最初的问题之间的相似度来计算注意力（需要注意的是，这一方法和通常的注意力是有区别的。通常的注意力只构建事实和现有记忆的相似度）。
#我们把结果送进一个两层的前馈网络来获得每个事实的注意力常数。
def attention(c,mem,existing_facts):
	#c: A [batch_size, maximum_sentence_count, recurrent_cell_size]包含上下文中的所有事实
	#mem: A [batch_size, maximum_sentence_count, recurrent_cell_size]包含当前所有记忆 对所有的fact都有相同的记忆 它应该是所有事实的相同记忆以获得准确的结果
	#existing_facts: A [batch_size, maximum_sentence_count, 1]标识事实的存在性
	with tf.variable_scope("attending") as scope:
		attending = tf.concat([c,mem,re_q,c*re_q,c*mem,(c-re_q)**2,(c-mem)**2],2)#将7个维度的东西都加在第二维度#(?,?,896=128*7)

		m1 = tf.matmul(attending*existing_facts,tf.tile(w_1,tf.stack([tf.shape(attending)[0],1,1])))*existing_facts
		#attending * existing_facts(128,最多句子数，896)  w_1(128，896，128)#print(facts_0s.shape)#(?,?,1)
		bias_1 = b_1 * existing_facts

		tanh = tf.nn.relu(m1+bias_1)

		m2 = tf.matmul(tanh,tf.tile(w_2,tf.stack([tf.shape(attending)[0],1,1])))

		bias_2 = b_2 * existing_facts

		norm_m2 = tf.nn.l2_normalize(m2 + bias_2,-1)
		#这个函数的作用是利用 L2 范数对指定维度 dim 进行标准化。比如，对于一个一维的张量，指定维度 dim = 0，那么计算结果为：output = x / sqrt( max( sum( x ** 2 ) , epsilon ) )
        # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor. 
        #     We make norm_m2 a sparse tensor, then make it dense again after the operation.
		softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]#找出tensor里所有True值的index 
        #print(softmax_idx.shape)#(?, 2)
		softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_idx)
        #print(softmax_gather.shape)#(?,)
		softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
        #print(softmax_shape.shape)#(2,)
		softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
        #print(softmaxable.shape)
		return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)),-1)#(128, 最长句子数, 1)
      	#表示该输入与问题的相似度

facts_0s = tf.cast(tf.count_nonzero(input_sentences_endings[:,:,-1:],-1,keep_dims=True),tf.float32)#print(facts_0s.shape)#(?,?,1)
 #将数据格式化[-1:]取最后一个字符

#接着我们把输入的事实通过一个GRU来给予权重（通过相应的注意力常数来给予），从而修改了现有记忆。
#为了避免当上下文比矩阵的长度小的时候向现有记忆里加入不正确的信息，我们创建了一掩盖层，当事实不存在的时候就根本不去注意它（例如，获取了相同的现有记忆）
with tf.variable_scope("Episodes") as scope:
	attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

	#记忆list 最后一个元素表示当前记忆
	memory = [q]

	attends = []

	for a in range(passes):
		attend_to = attention(cs, tf.tile(tf.reshape(memory[-1],[-1,1,recurrent_cell_size]),size),#A [batch_size, maximum_sentence_count句子数量, recurrent_cell_size128]
                              facts_0s)#我们通过构建每个事实、现有记忆和最初的问题之间的相似度来计算注意力#表示每次阅读对每个时刻（句子）的关注程度 (128, 最长句子数, 1)

		retain = 1-attend_to

		while_valid_index = (lambda state, index: index < tf.shape(cs)[1])#函数返回值为true 或者 false 句子个数

		#接着我们把输入的事实通过一个GRU来给予权重 (通过相应的注意力常数来给予）
		update_state = ( lambda state,index:   ( attend_to[:,index,:]  *  attention_gru(cs[:,index,:],state)[0]  +  retain[:,index,:] * state )   )
		#从而修改了现有记忆
		#tf.while_loop(cond, body, (2, 1, 1))
		memory.append(  tuple(  tf.while_loop(  while_valid_index,   (lambda state,index : ( update_state(state,index),index+1 ))   ,loop_vars = [memory[-1],0])  )[0]   )

		attends.append(attend_to)

		scope.reuse_variables()#保证每次GRU都使用的相同的变量

















import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from episodicModule import *
from paramaters import *

#它使用一个全连接层来对问题和片段记忆模块的输出进行回归来得到最后结果的词向量，以及上下文里和这个词向量距离最接近的词作为我们最后的答案（保证结果是一个实际的词）。
#answer module就是一个简单的GRU decoder，接受上次输出的单词（应该是one-hot向量），以及episodic memory，输出一个单词：
#我们为每个词创建一个得分来计算最近的词，这个得分就是结果的词距离。
#a0 
gold_standard = tf.placeholder(tf.float32,[None,1,D],"answer")#标准答案

a0 = tf.concat([memory[-1],q], -1)#添加到最后一维度 (128,256)

fc_init = tf.random_normal_initializer(stddev = 0.1)#全连接层的权重矩阵生成器

with tf.variable_scope("answer"):
	w_answer = tf.get_variable("weight",[recurrent_cell_size*2,D],initializer = fc_init)#权重 (256,50)

	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_answer)) #正则化

	logit = tf.expand_dims(tf.matmul(a0,w_answer),1)#(128,1,50) tf.matmul(a0,w_answer)(128,50) 最后结果的词向量

	with tf.variable_scope("ending"):
		#input_sentence_endings(128,maximum_sentences_counts,128)

		#我们为每个词创建一个得分来计算最近的词，这个得分就是结果的词距离。
		all_ends = tf.reshape(input_sentences_endings,[-1,2])#(?, 2)128个例子所有句子数量

		range_ends = tf.range(tf.shape(all_ends)[0])#返回一个等差数列

		ends_indices = tf.stack([all_ends[:,0],range_ends], axis=1)#(?,2) r维变成r+1维

		ind = tf.reduce_max(   tf.scatter_nd( ends_indices,  all_ends[:,1]  , [ tf.shape(q)[0]  ,  tf.shape(all_ends)[0] ] )    ,  axis=-1  )
		#tf.scatter_nd 根据indices将updates散布到shape张量。indices指定位置    128                        128个例子所有句子数量
		#沿着tensor的某一维度，计算元素的最大值。
		#(128,)
   
		range_ind = tf.range(tf.shape(ind)[0])#返回一个等差数列 (0,128)

		mask_ends = tf.cast( tf.scatter_nd( tf.stack( [ind , range_ind], axis=1 ), tf.ones_like(range_ind), [tf.reduce_max(ind)+1, tf.shape(ind)[0]] ), bool)
		#格式化为bool类型变量          散布          r维变r+1维度                          该方法用于创建一个所有参数均为1的tensor对象      

		mask = tf.scan(tf.logical_xor,mask_ends, tf.ones_like(range_ind, dtype=bool))
		#给mask_ends和 tf.ones_like(range_ind, dtype=bool)做异或

	logits = -tf.reduce_sum(  tf.square(  context *  tf.transpose(  tf.expand_dims(   tf.cast(mask, tf.float32), -1 )   , [1,0,2]) - logit), axis=-1)
	 	#print(mask.shape)(单词数量, 128)
        #print(tf.transpose(tf.expand_dims(tf.cast(mask, tf.float32),-1),[1,0,2]).shape)(128,单词数量,1)
        #print((context*tf.transpose(tf.expand_dims(tf.cast(mask, tf.float32),-1),[1,0,2])).shape)(128, 单词数量, 50)
        #print(tf.square(context*tf.transpose(tf.expand_dims(tf.cast(mask, tf.float32),-1),[1,0,2])- logit).shape)(128, 单词数量, 50)









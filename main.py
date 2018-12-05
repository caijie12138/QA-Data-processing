import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from dataAnalyse import *

final_train_data = finalize(train_data) #数据处理
final_test_data = finalize(test_data) 
print("Data Analyse done!\n")

tf.reset_default_graph()#清除默认图

from paramaters import *
from inputModule import *
from episodicModule import *
from answerModule import *
from prepareBatch import *


#Trainning

with tf.variable_scope("accuracy"):

	eq = tf.equal(context,gold_standard)#(128,上下文包含单词个数最多的单词个数,50)#tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
		
	corrbool = tf.reduce_all(eq,-1)#对tensor中每行元素求逻辑‘与’(128,上下文包含单词个数最多的单词个数) 50维向量全部相等才为true

	logloc = tf.reduce_max(logits, -1, keep_dims = True)#沿着tensor的某一维度，计算元素的最大值。找每行的最大值 找最相似的单词的位置logits(128,单词数)  降维 (128,1)

	locs = tf.equal(logits, logloc)#(128,单词数) 找到预测单词的最终位置 并标记为true

	correctsbool = tf.reduce_any(tf.logical_and(locs, corrbool), -1)# 只要相同就设置为true 对tensor中各个元素求逻辑‘或’  sess.run(tf.logical_and(True, False)) false

	# 将true和false变为1和0
	corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32), 
                        tf.zeros_like(correctsbool,dtype=tf.float32))

    # 将true和false变为1和0
	corr = tf.where(corrbool, tf.ones_like(corrbool, dtype=tf.float32), 
                        tf.zeros_like(corrbool,dtype=tf.float32))
with tf.variable_scope("loss"):
   
    #  logits是预测单词和正确答案距离的差值.corr标记了正确答案在上下文单词中的位置
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = tf.nn.l2_normalize(logits,-1),labels = corr)
    
    # 添加正则化项, 权重衰减.
	total_loss = tf.reduce_mean(loss) + weight_decay * tf.add_n(#tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))#从一个结合中取出全部变量，是一个列表

# TensorFlow's default implementation of the Adam optimizer works. We can adjust more than 
#  just the learning rate, but it's not necessary to find a very good optimum.
optimizer = tf.train.AdamOptimizer(learning_rate)

# Once we have an optimizer, we ask it to minimize the loss 
#   in order to work towards the proper training.
opt_op = optimizer.minimize(total_loss)

# Initialize variables
init = tf.global_variables_initializer()

# Launch the TensorFlow session
sess = tf.Session()
sess.run(init)

#准备验证集
batch = np.random.randint(final_test_data.shape[0],size=batch_size*10);#随机生成1280个数

batch_data = final_test_data[batch]#从测试集中选择数据用于验证集 观察神经网络的泛化能力 如果使用测试集中的数据作为验证集整个神经网络可能会过拟合

val_set,vconx_words,vcqas = prep_batch(batch_data,True)

def train(iterations,batch_size):
	training_iterations = range(0,iterations,batch_size)

	wordz = []

	for j in training_iterations:

		batch = np.random.randint(final_train_data.shape[0],size=batch_size)#随机选128个问题

		batch_data = final_train_data[batch]

		sess.run(opt_op,feed_dict=prep_batch(batch_data))

		if(j/batch_size)%display_step == 0:

			acc, ccs, tmp_loss, log, con, cor, loc  = sess.run([corrects, cs, total_loss, logit,
                                                                context_placeholder,corr, locs], 
                                                               feed_dict=val_set)

			print("Iter " + str(j/batch_size) + ", Minibatch Loss= ",tmp_loss,
                  "Accuracy= ", np.mean(acc)) 


train(training_iterations_count,batch_size)

run_val = sess.run([corrbool,locs,total_loss,logits,facts_0s,w_1]+attends+[query,cs,question_module_outputs],feed_dict = val_set)

max_limit = batch_size*10

r = run_val[0]

u = run_val[1]

n = run_val[2]

prepdict_answerind = np.argmax(u,axis = 1)

standard_answerind = np.argmax(r,axis = 1)

for i,x,context_words,cqa in list(zip(prepdict_answerind,standard_answerind,vconx_words,vcqas))[:max_limit]:

	print("上下文："," ".join(context_words))

	print("问题："," ".join(cqa[3]))

	print("预测答案：",context_words[i])

	print("标准答案：",context_words[x],["正确","错误"][i!=x])

	print()


























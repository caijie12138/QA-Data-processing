import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from paramaters import *

#contex:A[batch_size,问题对应上下文的单词最大数，每个单词的向量长度]
context = tf.placeholder(tf.float32,[None,None,D],"context")#(1280, 614, 50)

context_placeholder = context
#input_sentence_ending:A[batch_size,最大句子数量maximum_sentence_count，2]包含每个句子的结束位置信息 
input_sentences_endings = tf.placeholder(tf.int32,[None,None,2],"sentence")#(1280,102,2)第三维表示第几个问题中的第几句话的长度 找到包含句子最多的问题对应的上下文句子数量

input_gru = tf.contrib.rnn.GRUCell(batch_size)#定义GRUcell

gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru,input_p,output_p)#Dropout

input_module_outputs,_ = tf.nn.dynamic_rnn(gru_drop,context,dtype = tf.float32,scope = "input_module")#(128,614,128)

cs = tf.gather_nd(input_module_outputs,input_sentences_endings)#shape(128,?,128)  input_sentence_ending当做索引，从input_module_outputs寻找元素
#A [batch_size, maximum_sentence_count, recurrent_cell_size]#<128,上下文句子个数,128>


'''
输入模块接受 TI个输入单词，输出TC个“事实”的表示。如果输出是一系列词语，那么有TC=TI；
如果输出是一系列句子，那么约定TC表示句子的数量，TI表示句子中单词的数量。
我们使用简单的GRU读入句子，得到隐藏状态ht=GRU(xt,ht−1)，其中xt=L[wt]，L是embedding matrix，wt是时刻t的词语。
输入模块是我们的动态记忆网络用来得到答案的4个模块的第一个。
它包括一个带有门循环单元GRU的输入通道，让数据通过来收集证据片段。
每个片段的证据或是事实都对应这上下文的单个句子，并由这个时间片的输出所代表。
这就要求一些非TensorFlow的预处理，从而能获取句子的结尾并把这个信息送给TensorFlow来用于后面的模块。
 '''

 #tensorflow 的dynamic_rnn方法，我们用一个小例子来说明其用法，假设你的RNN的输入input是[2,20,128]，
#其中2是batch_size,20是文本最大长度，128是embedding_size，可以看出，有两个example，
#我们假设第二个文本长度只有13，剩下的7个是使用0-padding方法填充的。
#dynamic返回的是两个参数：outputs,last_states，其中outputs是[2,20,128]，也就是每一个迭代隐状态的输出,last_states是由(c,h)组成的tuple，均为[batch,128]。

#到这里并没有什么不同，但是dynamic有个参数：sequence_length，这个参数用来指定每个example的长度
#，比如上面的例子中，我们令 sequence_length为[20,13]，
#表示第一个example有效长度为20，第二个example有效长度为13，当我们传入这个参数的时候，
#对于第二个example，TensorFlow对于13以后的padding就不计算了，其last_states将重复第13步的last_states直至第20步，而outputs中超过13步的结果将会被置零。



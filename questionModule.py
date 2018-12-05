import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from inputModule import *
from paramaters import *

#query：A[batch_size,最长问题长度,D]
query = tf.placeholder(tf.float32,[None,None,D],"query")#(1280,8,50)
#input_query_lengths: A [batch_size, 2]
input_query_lengths = tf.placeholder(tf.int32,[None,2],"query_length")##print(querylengths)(0,4)(1,6)...(1279,6) 问题序号加上问题长度 (1280, 2)

question_module_outputs,_ = tf.nn.dynamic_rnn(gru_drop,query,dtype = tf.float32,scope=tf.VariableScope(True,"input_module"))#(128, 最长问题长度, 128)

q = tf.gather_nd(question_module_outputs,input_query_lengths)#(128, 128)



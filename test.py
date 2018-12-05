import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from dataAnalyse import *
from paramaters import *
from inputModule import *
from episodicModule import *
from answerModule import *
from prepareBatch import *

sess = tf.Session()
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

	print("预测答案：",vconx_words[i])

	print("标准答案：",vconx_words[e],["正确","错误"][i!=x])

	print()

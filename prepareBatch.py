import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools
from inputModule import *
from questionModule import *
from answerModule import *
 
#准备数据

def prep_batch(batch_data,more_data=False):

	context_vec, sentence_ends, questionvs, spt, context_words, cqas, answervs,_ = zip(*batch_data) #(1280,8)

	ends = list(sentence_ends)#长度为1280 包含每个问题前上下文的每个句子的结束位置

	maxend = max(map(len,ends))#map() 会根据提供的函数对指定序列做映射  找到最多的句子数量 包含断句点最多的

	aends = np.zeros((len(ends),maxend))#(1280,102)

	for index, i in enumerate(ends):
		for indexj, x in enumerate(i):
			aends[index, indexj] = x-1#aends表示第几个问题中的第几句话的结束位置

	new_ends = np.zeros(aends.shape+(2,))#扩展维度 (1280,102,2)

	for index, x in np.ndenumerate(aends):#array coordinates and values
		new_ends[index+(0,)] = index[0]#问题序号
		new_ends[index+(1,)] = x#上下文句子结束位置

	contexts = list(context_vec)#list每个元素都是上下文向量

	max_context_length = max([len(x) for x in contexts])#所有的上下文句子数量

	contextsize = list(np.array(contexts[0]).shape)#(100,50)单词数量 词向量维度 
    #['bill', 'travelled', 'to', 'the', 'hallway', '.', 'bill', 'went', 'back', 'to', 'the', 'garden', '.', 
    #'fred', 'went', 'to', 'the', 'office', '.', 'fred', 'travelled', 'to', 'the', 'kitchen', '.', 'jeff', 
    #'travelled', 'to', 'the', 'bathroom', '.', 'fred', 'journeyed', 'to', 'the', 'bedroom', '.', 
    #'fred', 'picked', 'up', 'the', 'milk', 'there', '.', 'mary', 'travelled', 'to', 'the', 'kitchen', '.', 
    #'bill', 'went', 'back', 'to', 'the', 'hallway', '.', 'fred', 'went', 'back', 'to', 'the', 'office', '.',
    # 'fred', 'left', 'the', 'milk', '.', 'fred', 'got', 'the', 'milk', 'there', '.', 'fred', 'went', 'to', 'the', 
    #'bathroom', '.', 'mary', 'travelled', 'to', 'the', 'office', '.', 'fred', 'gave', 'the', 'milk', 'to', 'jeff', '.', 
    #'jeff', 'moved', 'to', 'the', 'office', '.']
	contextsize[0] = max_context_length

	final_contexts = np.zeros([len(contexts)]+contextsize)

	contexts = [np.array(x) for x in contexts]#shape和上方的contexts一样，一个list中含有1280个(上下文单词数量,50)，只是转换为了array 为了遍历

	for i, context in enumerate(contexts):
		final_contexts[i,0:len(context),:] = context#(1280,最长上下文,50)

	max_query_length = max(len(x) for x in questionvs)

	querysize = list(np.array(questionvs[0]).shape)#(7,50)

	querysize[:1] = [len(questionvs),max_query_length]

	queries = np.zeros(querysize)#(1280,8,50)

	querylengths = np.array(list(zip(range(len(questionvs)),[len(q)-1 for q in questionvs])))#print(querylengths)(0,4)(1,6)...(1279,6) 问题序号加上长度

	questions = [np.array(q) for q in questionvs]#(7,50)

	for i, question in enumerate(questions):
		queries[i,0:len(question),:] = question

	data = {context_placeholder: final_contexts, input_sentences_endings: new_ends, query:queries, input_query_lengths:querylengths, gold_standard: answervs}

	return (data, context_words, cqas) if more_data else data














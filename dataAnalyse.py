import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
import re
import itertools

glove_file = "data/glove.6B.50d.txt"#词向量文件
train_file = "data/qa5_three-arg-relations_train.txt"#训练数据集合
test_file = "data/qa5_three-arg-relations_test.txt"#测试数据集合

#GloVe数据集合中是很多单词的词向量表示 每个单词通过50维向量来表示
def get_glove():#400000个单词加符号
	glove_map = {}
	with open(glove_file,"r",encoding="utf8") as glove:
		for line in glove:
			name,vector = tuple(line.split(" ",1))#这里的1表示分割次数，意思就是首次遇到空格就分割
			glove_map[name] = np.fromstring(vector,sep=" ")#根据空格分开的数字创建矩阵
	return glove_map

def gloveToWordVector(glove_map):
	wv = []
	for item in glove_map.items():
		wv.append(item[1])
	s = np.vstack(wv)#变成一列
	v = np.var(s,0)#方差
	m = np.mean(s,0)#均值   目的找不到单词的情况下给出一个同分布的单词向量
	RS = np.random.RandomState()#伪随机数生成器
	return v,m,RS

glove_map = get_glove()
#print(len(glove_map))400000
v,m,RS = gloveToWordVector(glove_map)#数据的分布 方差 均值和随机数生成器

def fill_unknow(unknow):#fill_unkNOW函数会在我们需要时给出一个新的词向量
	glove_map[unknow] = RS.multivariate_normal(m,np.diag(v))
	return glove_map[unknow]

def sentence2sequence(sentence):#将句子转化为数字
	tokens = sentence.strip('"(),-').split(" ")#先使用strip方法移除首尾指定字符 再将空格分开的单词存入token
	rows = []#词向量集合
	words = []#单词集合
	for token in tokens:#贪心搜索策略
		i = len(token)
		while len(token) > 0:
			word = token[:i]
			if word in glove_map:
				rows.append(glove_map[word])
				words.append(word)
				token = token[i:]
				i = len(token)
				continue;
			else:
				i = i-1 #如果不在集合中，便把token长度减一 寻找最相似的单词
			if i == 0:
				rows.append(fill_unknow(token))#相似的也不存在的话就只能随机生成了
				words.append(token)
				break;
	return np.array(rows),words #一个是单词集合 一个是向量集合


def contextualize(set_file):
	'''例子
	1 Bill travelled to the office.
	2 Bill picked up the football there.
	3 Bill went to the bedroom.
	4 Bill gave the football to Fred.
	5 What did Bill give to Fred? 	football	4
	6 Fred handed the football to Bill.
	7 Jeff went back to the office.
	8 Who received the football? 	Bill	6
	9 Bill travelled to the office.
	10 Bill got the milk there.
	11 Who received the football? 	Bill	6
	12 Fred travelled to the garden.
	13 Fred went to the hallway.
	14 Bill journeyed to the bedroom.
	15 Jeff moved to the hallway.
	16 Jeff journeyed to the bathroom.
	17 Bill journeyed to the office.
	18 Fred travelled to the bathroom.
	19 Mary journeyed to the kitchen.
	20 Jeff took the apple there.
	21 Jeff gave the apple to Fred.
	22 Who did Jeff give the apple to? 	Fred	21
	23 Bill went back to the bathroom.
	24 Bill left the milk.
	25 Who received the apple? 	Fred	21
	'''
	data = []
	context = []
	with open(set_file,"r",encoding='utf8') as train:
		for line in train:
			l,ine = tuple(line.split(" ",1))#1代表分割次数 遇到第一个空格切分 并将其转换为tuple 元素不可变
			if l is "1":#新的文本总是从1开始计数
				context = []
			if "\t" in ine:#以分隔符划分问题 答案 答案对应的句子的序号
				question,answer,support = tuple(ine.split("\t"))
				data.append((tuple(zip(*context))+sentence2sequence(question)+sentence2sequence(answer)+([int(s) for s in support.split()],)))
			else:
				context.append(sentence2sequence(ine[:-1]))#去掉最后一位
	return data

train_data = contextualize(train_file)
#print(len(train_data)) 10000
test_data = contextualize(test_file)
#print(len(test_data)) #1000

final_train_data = []

def finalize(data):
	final_data = []
	for cqas in train_data:
		contextvs, contextws, qvs, qws, avs, aws, spt = cqas
		#上下文向量  上下文问题  问题向量 问题 答案向量 答案 答案在的句子序号
		lengths = itertools.accumulate(len(cves)for cves in contextvs)
		context_vec = np.concatenate(contextvs)	#每次拼接一部分
		context_words = sum(contextws,[])	#每次拼接一部分
		"""
		['bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'picked', 'up', 'the', 'football', 'there', '.', 'bill', 'went', 'to', 'the', 'bedroom', '.', 'bill', 'gave', 'the', 'football', 'to', 'fred', '.', 'fred', 'handed', 'the', 'football', 'to', 'bill', '.', 'jeff', 'went', 'back', 'to', 'the', 'office', '.']
		['bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'picked', 'up', 'the', 'football', 'there', '.', 'bill', 'went', 'to', 'the', 'bedroom', '.', 'bill', 'gave', 'the', 'football', 'to', 'fred', '.', 'fred', 'handed', 'the', 'football', 'to', 'bill', '.', 'jeff', 'went', 'back', 'to', 'the', 'office', '.', 'bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'got', 'the', 'milk', 'there', '.']
		['bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'picked', 'up', 'the', 'football', 'there', '.', 'bill', 'went', 'to', 'the', 'bedroom', '.', 'bill', 'gave', 'the', 'football', 'to', 'fred', '.', 'fred', 'handed', 'the', 'football', 'to', 'bill', '.', 'jeff', 'went', 'back', 'to', 'the', 'office', '.', 'bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'got', 'the', 'milk', 'there', '.', 'fred', 'travelled', 'to', 'the', 'garden', '.', 'fred', 'went', 'to', 'the', 'hallway', '.', 'bill', 'journeyed', 'to', 'the', 'bedroom', '.', 'jeff', 'moved', 'to', 'the', 'hallway', '.', 'jeff', 'journeyed', 'to', 'the', 'bathroom', '.', 'bill', 'journeyed', 'to', 'the', 'office', '.', 'fred', 'travelled', 'to', 'the', 'bathroom', '.', 'mary', 'journeyed', 'to', 'the', 'kitchen', '.', 'jeff', 'took', 'the', 'apple', 'there', '.', 'jeff', 'gave', 'the', 'apple', 'to', 'fred', '.']
		['bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'picked', 'up', 'the', 'football', 'there', '.', 'bill', 'went', 'to', 'the', 'bedroom', '.', 'bill', 'gave', 'the', 'football', 'to', 'fred', '.', 'fred', 'handed', 'the', 'football', 'to', 'bill', '.', 'jeff', 'went', 'back', 'to', 'the', 'office', '.', 'bill', 'travelled', 'to', 'the', 'office', '.', 'bill', 'got', 'the', 'milk', 'there', '.', 'fred', 'travelled', 'to', 'the', 'garden', '.', 'fred', 'went', 'to', 'the', 'hallway', '.', 'bill', 'journeyed', 'to', 'the', 'bedroom', '.', 'jeff', 'moved', 'to', 'the', 'hallway', '.', 'jeff', 'journeyed', 'to', 'the', 'bathroom', '.', 'bill', 'journeyed', 'to', 'the', 'office', '.', 'fred', 'travelled', 'to', 'the', 'bathroom', '.', 'mary', 'journeyed', 'to', 'the', 'kitchen', '.', 'jeff', 'took', 'the', 'apple', 'there', '.', 'jeff', 'gave', 'the', 'apple', 'to', 'fred', '.', 'bill', 'went', 'back', 'to', 'the', 'bathroom', '.', 'bill', 'left', 'the', 'milk', '.']
		"""
		sentence_ends = np.array(list(lengths)) #list 包含每个句子的结尾
		final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws))
	return np.array(final_data)
							#[0][0](26,50)26个单词包含标点符号 每个50维向量 
                            #[0][1][ 6 13 19 26]每个句子结束的位置
                            #[0][2](7,50)问题单词向量
                            #[0][3] 答案出现的句子位置
                            #[0][4] 上下文的单词 
                            #[0][5] 7个维度的所有#上下文向量  上下文问题  问题向量 问题 答案向量 答案 答案在的句子序号
                            #[0][6] 答案单词向量
                            #[0][7] 答案单词
   	
     







#coding:utf-8

import random
import csv
from print_r import print_r
import numpy as np

#過去２週間のデータから、今日の高値より、翌日の日の安値が高いもの
#5(始,終,高,低,出来高)x14(２週間)x学習件数

#ラベルと教師データを作る
def makedata(train,test,validation):

	#ファイル読み込み	
	f = open('kabuka-201407-201602_sort.csv', 'rb')
	kabuka = csv.reader(f)

	last16daydata = []

	day15 = []
	day16 = []

	day = 0
	meigara = ""

	supervied_count = 0
	
	trainarray = np.array([])
	testarray = np.array([])
	validationarray = np.array([])
	
	ii = 1
	#データを作成
	#14日分の5次元データを作成
	for nowraw in kabuka:

		ii += 1
		#if ii> 1000:
		#	break

		#銘柄変更したらまたクリア
		if meigara != nowraw[1]:
			meigara = nowraw[1]
			day = 0
			supervised = []
			day15 = []
			day16 = []

		day += 1

		#5次元のデータを作成
		now = [nowraw[2],nowraw[3],nowraw[4],nowraw[5],nowraw[6]]

		#先頭に追加
		supervised.insert(0,now)

		#17日以下は行わない
		if day < 17 :
			continue;

		day15 = supervised[1]
		day16 = supervised[0]

		#末尾削除
		supervised.pop()

		#label作成
		label = [0., 1.]
		if day16[0] > day15[0]:
			label = [1., 0.]

		dbg = ""
		for d in supervised[2:]:
			dbg += str(d[0]) +  ","


		#75%が訓練用 15%が正解データ、残り10%が検証用のデータ
		rand = random.randint(1,100)	
		superviseddata = np.array(supervised[2:])
		supervisedlabel = np.array(label)

		#print superviseddata

		if rand  < 75:
			train.add(superviseddata, supervisedlabel)
		elif rand < 90:
			test.add(superviseddata, supervisedlabel)
		else:
			validation.add(superviseddata, supervisedlabel)

		#１行追加
		#2日前-16日分のデータを作成
		#https://hydrocul.github.io/wiki/programming_languages_diff/list/sub-list.html
		
	#データ形式の形を整える
	train.format()
	test.format()
	validation.format()

#データセットのクラス
class DataSet(object):

	def __init__(self):

		#株価ちょくせつじゃなくて、1.0 - 0.0　で丸めたりとか、
		#したいのだけど、、、まぁあとで考える。
		#それに、移動平均値とか、出したくもなるのかも、、、わからない。

		self._kabukas = np.array([])
		self._labels = np.array([])
		self._num_examples = 0
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def add(self, kabuka, label):

		self._kabukas = np.append(self._kabukas, kabuka)
		self._labels = np.append(self._labels, label)
		self._num_examples += 1 

	def format(self):
		
		self._kabukas = self._kabukas.reshape((self._num_examples, 14 * 5))
		self._labels = self._labels.reshape((self._num_examples, 2))

	@property
	def kabukas(self):
		return self._kabukas

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):	

		start = self._index_in_epoch
		end = self._index_in_epoch + batch_size
		self._index_in_epoch += batch_size

		#データをバッチサイズがこえているので、らんだに詰め物したので埋める
		if self._index_in_epoch > self._num_examples:
			
			# Finished epoch
			self._epochs_completed += 1

			# Shuffle the data
			kabukarand = np.arange(batch_size * 14 * 5)
			np.random.shuffle(kabukarand)
			labelrand = np.arange(batch_size * 2)
			np.random.shuffle(labelrand)
			
			self._num_examples = batch_size
			self._kabukas = kabukarand
			self._labels = labelrand
			self.format()
      
			start = 0
			self._index_in_epoch = batch_size

			assert batch_size <= self._num_examples
			end = self._index_in_epoch

		return self._kabukas[start:end], self._labels[start:end]		

def read_data_sets():

	#つめこみようのガワclass作成
	class DataSets(object):
		pass

	datasets = DataSets()

	datasets.train = DataSet()
	datasets.test = DataSet()
	datasets.validation = DataSet()

	makedata(datasets.train, datasets.test, datasets.validation)


	return datasets

read_data_sets()


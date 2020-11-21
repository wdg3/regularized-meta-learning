import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.feather as feather
import os
import random
from functools import partial

class DataGenerator(object):
	def __init__(self, N, K, test_N, test_K):
		self.N = N
		self.K = K
		self.test_N = test_N
		self.test_K = test_K
		self.c_length = 6
		self.c_dim = 145
		self.tolerance = 0.075
		self.dim_output = self.N

		data_path = '../data/'
		assert(os.path.isdir(data_path))

		print('Loading data...', end='', flush=True)
		self.daily = feather.read_feather(data_path + 'daily.dat')
		#self.quarterly = feather.read_feather(data_path + 'quarterly.dat')
		self.combined = feather.read_feather(data_path + 'combined.dat')
		self.labels = feather.read_feather(data_path + 'labels.dat')
		print('done.')
		tickers = list(self.daily['ticker'].unique())

		print('Removing tickers with insufficient data for K=' + str(self.K) + '...', end='', flush=True)
		random.seed = 0
		np.random.seed(0)
		np.random.shuffle(tickers)
		copy = tickers.copy()
		for ticker in copy:
			if len(self.combined[self.combined['ticker']==ticker]) < (self.c_length + self.K * self.N):
				tickers.remove(ticker)
		self.labels['0'] = self.labels['0'].fillna(value=0)

		print('done.')

		table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
		df = table[0]
		symbols = df['Symbol']
		sp = list(symbols[symbols.isin(tickers)])
		self.num_test = int(len(sp) / 2)
		self.num_val = len(sp) - self.num_test
		random.seed=0
		np.random.seed(0)
		np.random.shuffle(sp)

		self.metatrain_tickers = [t for t in tickers if t not in sp]
		self.metatest_tickers = sp[:self.num_test]
		self.metaval_tickers = sp[self.num_test:]
		print(len(self.metatrain_tickers))
		print(len(self.metaval_tickers))
		print(len(self.metatest_tickers))
		#self.num_test = len(tickers) - num_train - num_val
		#self.metatrain_tickers = tickers[:num_train]
		#self.metaval_tickers = tickers[num_train:num_train + num_val]
		#self.metatest_tickers = tickers[num_train + num_val:]

	def get_subset(self, ticker, which):
		t = self.combined[self.combined['ticker'] == ticker].iloc[self.c_length:]
		l = self.labels[self.combined['ticker'] == ticker].iloc[self.c_length:]
		if which == 1:
			return t[l['0'] > self.tolerance]
		elif which == -1:
			return t[l['0'] < -self.tolerance]
		else:
			return t[(l['0'] <= self.tolerance) & (l['0'] >= -self.tolerance)]

	def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
		if batch_type == 'train':
			tickers = self.metatrain_tickers
			num_classes = self.N
			num_samples_per_class = self.K
		elif batch_type == 'val':
			tickers = self.metaval_tickers
			num_classes = self.N
			num_samples_per_class = self.K
		else:
			tickers = self.metatest_tickers
			num_classes = self.test_N
			num_samples_per_class = self.test_K

		sampled_tickers = random.sample(tickers, batch_size)
		#if num_classes != 3:
		#	raise(Exception('Only N=3 is currently supported.'))

		#out_dtype = ([tf.float32]*self.combined.shape[1]*num_classes*num_samples_per_class, [tf.int32]*num_classes*num_samples_per_class)
		data_partial = partial(self.get_datapoints, n_classes = num_classes, n_samples = num_samples_per_class)
		#(data, labels) = tf.map_fn(data_partial, elems=tf.constant(sampled_tickers),
		#						   dtype=out_dtype,
		#						   parallel_iterations=batch_size)
		#print(data.shape, labels.shape)
		data = np.zeros((batch_size, num_classes, num_samples_per_class, self.c_length, self.c_dim))
		labels = np.zeros((batch_size, num_classes, num_samples_per_class))
		for i in range(batch_size):
			(data[i, :, :, :, :], labels[i, :, :]) = data_partial(sampled_tickers[i])
		labels = labels.reshape((batch_size, num_classes, num_samples_per_class, 1))
		data = np.apply_along_axis(lambda x: x - np.mean(x), -1, data)
		data = np.apply_along_axis(lambda x: x / np.std(x), -1, data)


		return (data, labels)

	def get_datapoints(self, ticker, n_classes, n_samples, shuffle=True, test=False):
		t = self.combined[self.combined['ticker'] == ticker]
		l = self.labels[self.combined['ticker'] == ticker]
		curr = t.iloc[self.c_length:]
		curr_labels = l.iloc[self.c_length:]

		data = np.zeros((n_classes, n_samples, self.c_length, self.c_dim))
		labels = np.zeros((n_classes, n_samples))

		for i in range(n_classes):
			if (test):
				n_samples = len(curr)
			idxs = random.sample(range(len(curr)), n_samples)
			idxs.sort()
			for j in range(n_samples):
				k = idxs[j]
				index = int(curr.iloc[k].name)
				start = index - self.c_length + 1
				point = (t.loc[start:index]).drop(['ticker', 'date', 'calendardate', 'datekey', 'reportperiod'], axis=1)
				data[i, j, :, :] = np.array(point.fillna(value=0))
				labels[i, j] = float(l.loc[index]['0'])

		return (data, labels)

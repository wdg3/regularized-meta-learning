#########################
# data_processing.py #
#########################
# Implements DataGenerator and batch sampling
# functionality for MAML. Originally based on
# functionality in CS330 Homework 2, Fall 2020.

# Written by Will Geoghegan for CS330
# final project, Fall 2020. Based on work by
# CS330 course staff.

import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.feather as feather
import os
import random
from functools import partial

# A DataGenerator samples batches of tasks from our task distribution,
# with each task containing K samples. We experimented with N-way
# classification based on the size and direction of the price change
# before settling on regression, but N-way specification remains for
# easy extension back to classification
class DataGenerator(object):
	def __init__(self, N, K, test_N, test_K, demo):
		self.N = N
		self.K = K
		self.test_N = test_N
		self.test_K = test_K
		self.c_length = 6 # time dimension
		self.c_dim = 145
		self.tolerance = 0.075
		self.dim_output = self.N

		data_path = '../data/'
		assert(os.path.isdir(data_path))

		print('Loading data...', end='', flush=True)
		#self.daily = feather.read_feather(data_path + 'daily.dat')
		#self.quarterly = feather.read_feather(data_path + 'quarterly.dat')
		if demo == False:
			self.combined = feather.read_feather(data_path + 'combined.dat')
			self.labels = feather.read_feather(data_path + 'labels.dat')
		else:
			self.combined = feather.read_feather(data_path + 'combined_demo.dat')
			self.labels = feather.read_feather(data_path + 'labels_demo.dat')
		print('done.')
		tickers = list(self.combined['ticker'].unique())

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

		# Validation and test sets are drawn from large companies that are highly liquid
		# and desirable to trade in
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

	# This function was written for the classification setting.
	def get_subset(self, ticker, which):
		t = self.combined[self.combined['ticker'] == ticker].iloc[self.c_length:]
		l = self.labels[self.combined['ticker'] == ticker].iloc[self.c_length:]
		if which == 1:
			return t[l['0'] > self.tolerance]
		elif which == -1:
			return t[l['0'] < -self.tolerance]
		else:
			return t[(l['0'] <= self.tolerance) & (l['0'] >= -self.tolerance)]

	# This function samples a batch of the specified size from the specified task
	# partition. Also normalizes the batch by input dimension. Based on
	# the equivalent function in CS330 Homework 2.
	def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
		# Select the correct partition.
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
		data_partial = partial(self.get_datapoints, n_classes = num_classes, n_samples = num_samples_per_class)

		data = np.zeros((batch_size, num_classes, num_samples_per_class, self.c_length, self.c_dim))
		labels = np.zeros((batch_size, num_classes, num_samples_per_class))
		for i in range(batch_size):
			(data[i, :, :, :, :], labels[i, :, :]) = data_partial(sampled_tickers[i])
		labels = labels.reshape((batch_size, num_classes, num_samples_per_class, 1))
		
		# Normalize
		data = np.apply_along_axis(lambda x: x - np.mean(x), -1, data)
		data = np.apply_along_axis(lambda x: x / np.std(x), -1, data)

		return (data, labels)

	# This function selects the appropriate datapoints for a single task in a batch.
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

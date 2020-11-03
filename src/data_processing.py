import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.feather as feather
import os
import random

class DataGenerator(object):
	def __init__(self, N, K, test_N, test_K, t):
		self.N = N
		self.K = K
		self.test_N = test_N
		self.test_K = test_K
		self.d_length = 15
		self.q_length = 8
		self.d_dim = 14
		self.q_dim = 131
		self.time_horizon = t
		self.tolerance = 0.025
		self.dim_output = self.N
		self.dim_input_d = self.d_length * 14
		self.dim_input_q = self.q_length * 131

		data_path = '../data/'
		assert(os.path.isdir(data_path))

		print('Loading data...', end='', flush=True)
		self.daily = feather.read_feather(data_path + 'daily.dat')
		self.quarterly = feather.read_feather(data_path + 'quarterly.dat')
		print('done.')
		tickers = list(self.daily['ticker'].unique())

		random.seed = 123
		random.shuffle(tickers)

		num_train = 2500
		num_val = 250
		self.metatrain_tickers = tickers[:num_train]
		self.metaval_tickers = tickers[num_train:num_train + num_val]
		self.metatest_tickers = tickers[num_train + num_val:]

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

		sampled_tickers = random.sample(tickers, batch_size*10)
		if num_classes != 3:
			raise(Exception('Only N=3 is currently supported.'))

		thresholds = [-self.tolerance, self.tolerance]
		all_d = []
		all_q = []
		all_labels = []
		i = 0
		j = 0
		while i < batch_size:
			ticker = sampled_tickers[j]
			d_data, q_data, labels, success = self.get_datapoints(ticker, thresholds, num_classes, num_samples_per_class)
			if success:
				all_d.append(d_data)
				all_q.append(q_data)
				all_labels.append(tf.one_hot(labels, num_classes, axis=-1))
				i += 1
			j+= 1


		all_d = np.apply_along_axis(lambda x: x - np.mean(x), -1, np.array(all_d))
		all_d = np.apply_along_axis(lambda x: x / np.std(x), -1, all_d)
		all_q = np.apply_along_axis(lambda x: x - np.mean(x), -1, np.array(all_q))
		all_q = np.apply_along_axis(lambda x: x / np.std(x), -1, all_q)

		return np.array(all_d), np.array(all_q), np.array(all_labels)

	def get_datapoints(self, ticker, thresholds, n_classes, n_samples, shuffle=True):
		curr_d = self.daily[self.daily['ticker'] == ticker]
		curr_q = self.quarterly[self.quarterly['ticker'] == ticker]
		start_date = curr_q.iloc[self.q_length - 1]['datekey']
		d = curr_d[curr_d['date'] >= start_date]
		d = d.iloc[:-self.time_horizon]
		d = d.reset_index(drop=True)
		q = curr_q.reset_index(drop=True)
		y = curr_d[curr_d['date'] >= start_date]
		y = y.iloc[self.time_horizon:]['close']
		y = y.reset_index(drop=True)

		y_up = y * (1 + thresholds[1])
		y_down = y * (1 + thresholds[0])

		d_buckets = []
		d_buckets.append(d[d['close'] < y_down])
		d_buckets.append(d[(d['close'] >= y_down) & (d['close'] < y_up)])
		d_buckets.append(d[d['close'] >= y_up])
		
		d_data = np.zeros((n_classes, n_samples, self.d_length, self.d_dim))
		q_data = np.zeros((n_classes, n_samples, self.q_length, self.q_dim))

		labels = np.zeros((n_classes, n_samples))
		for i in range(n_classes):
			bucket = d_buckets[i]
			try:
				dates = random.sample(list(bucket['date'].iloc[self.d_length:]), n_samples)
			except ValueError:
				return [],[],[],False

			ls = bucket[bucket['date'].isin(dates)].index
			for j in range(n_samples):
				l = ls[j]
				point = d.loc[l - self.d_length + 1:l]
				point = point.drop(['ticker', 'date'], axis=1)
				d_data[i, j, :, :] = np.array(point)

				date = d.loc[l]['date']
				point = q[q['datekey'] < date].iloc[-self.q_length:]
				point = point.drop(['ticker', 'calendardate', 'datekey', 'reportperiod'], axis=1)
				q_data[i, j, :, :] = np.array(point)

			labels[i, :] = np.ones(n_samples) * i
		
		return np.array(d_data), np.array(q_data), np.array(labels), True


generator = DataGenerator(3,5,3,5, 20)
#generator.sample_batch('train', 16)

# Normalize numeric features to zero-mean and one-stddev.
"""print('Normalizing daily data...', end='', flush=True)
for i in range(2, len(daily.columns)):
	col = daily[daily.columns[i]]
	col = col - np.mean(col)
	daily[daily.columns[i]] = col / np.std(col)
print('done.')
print('Normalizing quarterly data...', end='', flush=True)
for i in range(4, len(quarterly.columns)):
	col = quarterly[quarterly.columns[i]]
	col = col - np.mean(col)
	quarterly[quarterly.columns[i]] = col / np.std(col)
print('done.')"""

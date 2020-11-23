#########################
# data_preprocessing.py #
#########################
# Converts csv-level financial data
# into pandas-based binaries for quick
# data pipelining during model training.

# Written by Will Geoghegan for CS330
# final project, Fall 2020.

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import os

path = '../data/'
assert os.path.isdir(path)

f_names = ['SEP', 'DAILY', 'SF1', 'SF3A']

dfs = {}
for f_name in f_names:
	print('Reading ' + path + f_name + '.csv...', end='', flush=True)
	dfs[f_name] = pd.read_csv(path + f_name + '.csv')
	print('done.')

# SEP: Daily price data. DAILY: Daily price metrics.
# SF1: Quarterly financial report data. SF3A: Quarterly institutional investment data.
print('Merging daily data...', end='', flush=True)
daily = pd.merge(dfs['SEP'], dfs['DAILY'], on=['ticker', 'date'], how='inner')
print('done.')
print('Merging quarterly data...', end='', flush=True)
quarterly = pd.merge(dfs['SF1'], dfs['SF3A'], on=['ticker', 'calendardate'], how='inner')
print('done.')

# A few NaN values in the quarterly data can be zero-filled but if there are too
# many we drop the row.
print('Handling null values...', end='', flush=True)
daily = daily.dropna()
quarterly = quarterly.dropna(thresh=130)
quarterly = quarterly.fillna(0)
print('done.')

# Data is sorted by date then ticker by default. Switch the order.
print('Sorting daily data...', end='', flush=True)
daily = daily.sort_values(by=['ticker', 'date'])
print('done.')
print('Sorting quarterly data...', end='', flush=True)
quarterly = quarterly.sort_values(by=['ticker', 'calendardate'])
print('done.')

# Quarterly data has both "as-reported" and "most recent report" dimensions.
# We are interested in "as-reported".
dim_filter = quarterly['dimension'] == 'ARQ'
quarterly = quarterly[dim_filter]

print('Dropping unneeded columns...', end='', flush=True)
quarterly = quarterly.drop(['dimension'], axis=1)
daily = daily.drop(['lastupdated_x', 'lastupdated_y'], axis=1)
quarterly = quarterly.drop(['lastupdated', 'name'], axis=1)
print('done.')

# Drop tickers for which we don't have enough data.
print('Dropping tickers without enough data...', end='', flush=True)
counts = daily['ticker'].value_counts()
t1 = [t for t in daily['ticker'].unique() if counts[t] > int(3*255)]
daily = daily[daily['ticker'].isin(t1)]
counts = quarterly['ticker'].value_counts()
t2 = [t for t in quarterly['ticker'].unique() if counts[t] >= 12]
quarterly = quarterly[quarterly['ticker'].isin(t2)]
print('done.')

# Drop tickers that are not shared between datasets.
print('Dropping unshared tickers...', end='', flush=True)
t1 = daily['ticker'].unique()
t2 = quarterly['ticker'].unique()
daily = daily[daily['ticker'].isin(t2)]
quarterly = quarterly[quarterly['ticker'].isin(t1)]
print('done.')

# Merge quarterly and daily data.
print('Combining datasets...', end='', flush=True)
combined = pd.merge(quarterly, daily, left_on=['ticker', 'datekey'], right_on=['ticker', 'date'], how='left')
print('done.')

# Calculate percentage change in following datapoint as y label.
print('Calculating regression/classification labels...', end='', flush=True)
labels = pd.Series(np.zeros(len(combined)), dtype='float64')
for i in range(len(combined) - 1):
	d = combined.iloc[i]
	n = combined.iloc[i + 1]
	if d['ticker'] == n['ticker']:
		result = (float(d['close']) / float(n['close'])) - 1
	else:
		result = 0
	labels.at[i] = result
print('done.')

print(str(daily.shape[0]) + ' daily data points\n' + str(daily.shape[1] - 2) + ' daily features\n' +
	str(quarterly.shape[0]) + ' quarterly data points\n' + str(quarterly.shape[1] - 4) + ' quarterly features\n' + 
	str(len(labels)) + ' label data points\n' +
	str(combined.shape[0]) + ' combined data points\n' + str(combined.shape[1] - 4) + ' combined features')

# Save everything
print('Converting and saving...', end='', flush=True)
feather.write_feather(daily, '../data/daily.dat')
feather.write_feather(quarterly, '../data/quarterly.dat')
feather.write_feather(combined, '../data/combined.dat')
feather.write_feather(pd.DataFrame(labels), '../data/labels.dat')
print('done.')
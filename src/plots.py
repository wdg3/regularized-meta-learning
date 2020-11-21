import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Helvetica Neue"
matplotlib.rcParams['font.family'] = "sans-serif"

colors = ["#404040", "#a040a0", "#ffaa00", "#0098db", "#009b40", "#8c1515", "#ff8888"]

axes = []
accs = []
mses = []
for prefix in ['ff', 'meta_ff', 'meta_conv', 'meta_reg', 'meta_reg_learn', 'meta_reg_learn_multi', 'meta_reg_learn_multi_ff']:
	accs.append(pd.read_csv('../data/'+prefix+'_acc.csv'))
	mses.append(pd.read_csv('../data/'+prefix+'_mse.csv'))
	axes.append(list(accs[-1]['Step']))

for i in range(7):
	ys = accs[i]['Value']
	smooth = [np.mean(ys[i-15:i]) for i in range(15, len(ys))]
	plt.plot(axes[i], ys, color=colors[i] + "20", label='_nolegend_')
	plt.plot(axes[i][15:], smooth, color=colors[i])
#ys = accs[-1]['Value']
#smooth = [np.mean(ys[i-15:i]) for i in range(15, len(ys))]
#plt.plot(x_short, ys, color=colors[-1] + "40", label='_nolegend_')
#plt.plot(x_short[15:], smooth, color=colors[-1])

plt.legend(['Supervised DNN', 'MAML-FF', 'MAML-TC', 'MR-MAML-TC', 'LMR-MAML-TC', 'Multi-LMR-MAML-TC', 'Multi-LMR-MAML-FF'])
plt.title("Meta-validation Accuracy Curves")
plt.xlabel("Iteration")
plt.ylabel("Meta-val. accuracy (smoothed)")
plt.grid()
plt.show()
plt.clf()

for i in range(7):
	ys = mses[i]['Value']
	smooth = [np.mean(np.log(ys[i-15:i])) for i in range(15, len(ys))]
	plt.plot(axes[i], np.log(ys), color=colors[i] +"20", label='_nolegend_')
	plt.plot(axes[i][15:], smooth,color=colors[i])

plt.legend(['Supervised DNN', 'MAML-FF', 'MAML-TC', 'MR-MAML-TC', 'LMR-MAML-TC', 'Multi-LMR-MAML-TC', 'Multi-LMR-MAML-FF'])
plt.title("Meta-validation MSE Curves")
plt.xlabel("Iteration")
plt.ylabel("Log MSE (smoothed)")
plt.ylim(-4, 2)
plt.grid()
plt.show()
plt.clf()

#y_train = y_train + list(pd.read_csv('train2.csv')['Value'])

#smooth_test = [np.mean(y_test[i-10:i]) for i in range(10, len(y_test))]
#smooth_train = [np.mean(y_train[i-10:i]) for i in range(10, len(y_train))]
#lt.plot(x_axis[5:], smooth_test, linestyle='--')
#plt.plot(x_axis[10:], smooth_test, color='#DE354C')
#plt.plot(x_axis[5:], smooth_train, linestyle='--')
#plt.plot(x_axis[10:], smooth_train, color='#283747')
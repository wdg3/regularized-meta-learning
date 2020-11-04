import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('test_1.csv')
x_axis = list(df['Step'])
x_axis = x_axis + (list(pd.read_csv('test_2.csv')['Step']))
y_test = list(df['Value'])
y_test = y_test + list(pd.read_csv('test_2.csv')['Value'])
y_train = list(pd.read_csv('train1.csv')['Value'])
y_train = y_train + list(pd.read_csv('train2.csv')['Value'])

smooth_test = [np.mean(y_test[i-10:i]) for i in range(10, len(y_test))]
smooth_train = [np.mean(y_train[i-10:i]) for i in range(10, len(y_train))]
#lt.plot(x_axis[5:], smooth_test, linestyle='--')
plt.plot(x_axis[10:], smooth_test, color='#DE354C')
#plt.plot(x_axis[5:], smooth_train, linestyle='--')
plt.plot(x_axis[10:], smooth_train, color='#283747')
plt.legend(["smoothed post-loop test acc", "smoothed pre-loop train acc"])
plt.title("Meta-validation accuracy over iterations")
plt.xlabel("Iteration")
plt.ylabel("Meta-val. accuracy")
plt.grid()
plt.show()
plt.clf()
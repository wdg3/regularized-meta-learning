from data_processing import *
import numpy as np
import tensorflow as tf
from functools import partial
import datetime

seed = 123

def cross_entropy_loss(pred, label):
	return tf.keras.losses.MSE(label, pred)

def accuracy(labels, predictions):
	return tf.reduce_mean(tf.cast(tf.equal(tf.sign(labels), tf.sign(predictions)), dtype=tf.float32))

def conv_block(inp, weight, bweight, bn, activation=tf.nn.relu):
	stride, no_stride = [1,2,2,1], [1,1,1,1]
	output = tf.nn.conv2d(input=inp, filters=weight, strides=no_stride, padding='SAME') + bweight
	normed = bn(output)
	normed = activation(normed)
	return normed

def weight_block(inp, weight, bweight, bn, activation=tf.nn.relu):
	output = tf.matmul(inp, weight) + bweight
	normed = bn(output)
	normed = activation(normed)
	return normed

class ModelLayers(tf.keras.layers.Layer):
	def __init__(self, dim_input_time, dim_input, dim_hidden, dim_output, conv):
		super(ModelLayers, self).__init__()
		self.dim_input_time = dim_input_time
		self.dim_input = dim_input
		self.dim_output = dim_output
		self.dim_hidden = dim_hidden
		self.conv = conv

		weights = {}

		dtype = tf.float32
		weight_initializer = tf.keras.initializers.GlorotUniform()

		if self.conv:
			weights['w1'] = tf.Variable(weight_initializer(shape=[self.dim_input, self.dim_input_time, 1, self.dim_hidden]),name='w1', dtype=dtype)
			weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
			self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
			weights['w2'] = tf.Variable(weight_initializer(shape=[self.dim_input, self.dim_input_time, self.dim_hidden, self.dim_hidden]),name='w2', dtype=dtype)
			weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
			self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
			weights['w3'] = tf.Variable(weight_initializer(shape=[self.dim_input, self.dim_input_time, self.dim_hidden, self.dim_hidden]),name='w3', dtype=dtype)
			weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
			self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
			weights['w4'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w4', dtype=dtype)
			weights['b4'] = tf.Variable(tf.zeros([self.dim_output]), name='b4')
		else:
			weights['w1'] = tf.Variable(weight_initializer(shape=[self.dim_input * self.dim_input_time, self.dim_hidden]),name='w1', dtype=dtype)
			weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
			self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
			weights['w2'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_hidden]),name='w2', dtype=dtype)
			weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
			self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
			weights['w3'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w3', dtype=dtype)
			weights['b3'] = tf.Variable(tf.zeros([self.dim_output]), name='b3')

		self.ff_weights = weights

	def call(self, inp, weights):
		if self.conv:
			inp = tf.reshape(inp, [inp.shape[0], inp.shape[1], inp.shape[2], 1])
			hidden1 = conv_block(inp, weights['w1'], weights['b1'], self.bn1)#, weights['b1'], self.bn1)
			hidden2 = conv_block(hidden1, weights['w2'], weights['b2'], self.bn2)#, weights['b2'], self.bn2)
			hidden3 = conv_block(hidden2, weights['w3'], weights['b3'], self.bn3)
			hidden3 = tf.reduce_mean(input_tensor=hidden3, axis=[1,2])
			return tf.matmul(hidden3, weights['w4']) + weights['b4']
		else:
			inp = tf.reshape(inp, [inp.shape[0], inp.shape[1] * inp.shape[2]])
			hidden1 = weight_block(inp, weights['w1'], weights['b1'], self.bn1)#, weights['b1'], self.bn1)
			hidden3 = weight_block(hidden1, weights['w2'], weights['b2'], self.bn2)#, weights['b2'], self.bn2)
			return tf.matmul(hidden3, weights['w3']) + weights['b3']

class MAML(tf.keras.Model):
	def __init__(self, dim_input_time=1, dim_input=1, dim_output=1, num_inner_updates=1,
				 inner_update_lr=0.4, num_units=32, k_shot=5, learn_inner_update_lr=False,
				 conv = True, meta_reg=False):
		super(MAML, self).__init__()
		self.dim_input_time = dim_input_time
		self.dim_input = dim_input
		self.dim_output = dim_output
		self.inner_update_lr = inner_update_lr
		self.loss_func = partial(cross_entropy_loss)
		self.dim_hidden = num_units
		self.meta_reg = meta_reg

		losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
		accuracies_tr_pre, accuracies_ts = [], []

		outputs_ts = [[]] * num_inner_updates
		losses_ts_post = [[]] * num_inner_updates
		accuracies_ts = [[]] * num_inner_updates

		tf.random.set_seed(seed)

		self.ff_layers = ModelLayers(self.dim_input_time, self.dim_input, self.dim_hidden, self.dim_output, conv)

		if self.meta_reg=='gaussian':
			self.meta_reg_dict = {}
			for (key, val) in self.ff_layers.ff_weights.items():
				if 'b' not in key:
					self.meta_reg_dict[key + '_std'] = tf.Variable(1., constraint = lambda x: tf.clip_by_value(x, 0, 100), name='meta_reg_std_%s' % key)
					self.meta_reg_dict[key + '_mean'] = tf.Variable(0., name='meta_reg_mean_%s' % key)
			self.dict_copy = self.ff_layers.ff_weights.copy()
		elif self.meta_reg == 'l2':
			self.meta_reg_dict = {}
			for (key, val) in self.ff_layers.ff_weights.items():
				if 'b' not in key:
					self.meta_reg_dict[key] = [tf.Variable(1.0, name = 'meta_reg_%s' % key, constraint= tf.keras.constraints.non_neg())]

		self.learn_inner_update_lr = learn_inner_update_lr
		if self.learn_inner_update_lr:
			self.inner_update_lr_dict = {}
			for key in self.ff_layers.ff_weights.keys():
				self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in range(num_inner_updates)]

	def call(self, inp, meta_batch_size=16, num_inner_updates=1):
		def task_inner_loop(inp, reuse=True, meta_batch_size=16, num_inner_updates=1):
			input_tr, input_ts, label_tr, label_ts = inp
			weights = self.ff_layers.ff_weights
			task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None
			task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

			closes = []
			temp_weights = dict(weights).copy()
			print
			if self.learn_inner_update_lr:
				temp_lrs = self.inner_update_lr_dict.copy()
			task_output_tr_pre = self.ff_layers(input_tr, temp_weights)
			task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
			if num_inner_updates == 0:
				task_outputs_ts=[self.ff_layers(input_ts, temp_weights)]
				task_losses_ts=[self.loss_func(task_outputs_ts[0], label_ts)]

			else:
				with tf.GradientTape(persistent=True) as inner_tape:
					for i in range(num_inner_updates):
						out = self.ff_layers(input_tr, temp_weights)
						loss = self.loss_func(out, label_tr)
						#if self.meta_reg == 'gaussian':
						#	dl_dw = inner_tape.gradient(loss, self.dict_copy)
						#	self.dict_copy = temp_weights.copy()
						#else:
						dl_dw = inner_tape.gradient(loss, temp_weights)
						#dl_dw = {(key, tf.clip_by_value(val, -1.0, 1.0)) for (key, val) in dl_dw.items()}
						if self.learn_inner_update_lr:
							temp_weights = dict(zip(temp_weights.keys(), [temp_weights[key] - temp_lrs[key][i] * tf.clip_by_value(dl_dw[key], -1.0, 1.0) for key in temp_weights.keys()]))
						else:
							temp_weights = dict(zip(temp_weights.keys(), [temp_weights[key] - self.inner_update_lr * tf.clip_by_value(dl_dw[key], -1.0, 1.0) for key in temp_weights.keys()]))

						task_outputs_ts.append(self.ff_layers(input_ts, temp_weights))
						task_losses_ts.append(self.loss_func(task_outputs_ts[i], label_ts))

			task_accuracy_tr_pre = accuracy(label_tr, task_output_tr_pre)
			for j in range(num_inner_updates):
				task_accuracies_ts.append(accuracy(label_ts, task_outputs_ts[j]))
				labs = label_ts
				preds = task_outputs_ts[j]
				close = tf.abs(labs - preds) < .025
				closes.append(np.mean(close))
			if num_inner_updates == 0:
				task_accuracies_ts = [task_accuracy_tr_pre]
				closes = [tf.abs(label_ts - task_outputs_ts[0]) < .025]
			close = np.mean(closes)

			task_output = ([task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts], close)

			return task_output

		standin = max(num_inner_updates, 1)
		d_tr, d_ts, label_tr, label_ts = inp
		unused = task_inner_loop((d_tr[0], d_ts[0], label_tr[0], label_ts[0]),
								  False, meta_batch_size, num_inner_updates)
		out_dtype = [tf.float32, [tf.float32]*standin, tf.float32, [tf.float32]*standin]
		out_dtype.extend([tf.float32, [tf.float32]*standin])
		out_dtype = (out_dtype, tf.float32)
		task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
		result = tf.map_fn(task_inner_loop_partial, elems=(d_tr, d_ts, label_tr, label_ts),
			dtype=out_dtype, parallel_iterations=meta_batch_size)

		return result

def outer_train_step(inp, model, optim, meta_batch_size=16, num_inner_updates=1):
	with tf.GradientTape(persistent=True) as outer_tape:
		if model.meta_reg == 'gaussian':
			reg_vars = [var for (key, var) in model.ff_layers.ff_weights.items()]
			for (key, val) in model.ff_layers.ff_weights.items():
				if 'b' not in key:
					mean = model.meta_reg_dict[key + '_mean']
					std = model.meta_reg_dict[key + '_std']
					noise = tf.random.normal(val.shape, mean, std)

					val = val + noise
					model.ff_layers.ff_weights[key] = val
		elif model.meta_reg == 'l2':
			penalty = tf.reduce_sum([model.meta_reg_dict[key][0] * tf.nn.l2_loss(var) for (key, var) in model.ff_layers.ff_weights.items() if 'b' not in key])
		(result, close) = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
		outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result
		
		if model.meta_reg == 'l2':
			total_losses_ts = [loss_ts + penalty for loss_ts in losses_ts]
		else:
			total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

	gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
	optim.apply_gradients(zip(gradients, model.trainable_variables))

	total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
	total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
	total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

	return (outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts), close

def outer_eval_step(inp, model, meta_batch_size=16, num_inner_updates=1):
	(result, close) = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

	outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

	total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
	total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

	total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
	total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

	return (outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts), close

def meta_train_fn(model, exp_string, data_generator, n_way=3, meta_train_iterations=5000, meta_batch_size=16,
				  log=True, logdir='../logs/', k_shot=5, num_inner_updates=1, meta_lr=0.001,
				  resume_itr = 0):
	SUMMARY_INTERVAL = 10
	SAVE_INTERVAL = 100
	PRINT_INTERVAL = 10
	TEST_PRINT_INTERVAL = PRINT_INTERVAL# * 5

	pre_accuracies, post_accuracies = [], []
	pre_losses, post_losses = [], []

	num_classes = data_generator.N

	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

	train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
	test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

	train_log_dir = logdir + '/gradient_tape/' + exp_string + '/train'
	test_log_dir = logdir + '/gradient_tape/' + exp_string + '/test'
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	test_summary_writer = tf.summary.create_file_writer(test_log_dir)

	for itr in range(resume_itr, meta_train_iterations):
		if (itr%10) != 0:
			print('Iteration %d' % itr)
		(data,  labs) = data_generator.sample_batch('meta_train', meta_batch_size, shuffle=True)
		d_tr, d_ts = data[:, :, :k_shot, :, :], data[:, :, k_shot:, :, :]
		d_tr = tf.reshape(d_tr, [meta_batch_size, n_way * k_shot, d_tr.shape[-2], d_tr.shape[-1]])
		d_ts = tf.reshape(d_ts, [meta_batch_size, n_way * k_shot, d_ts.shape[-2], d_ts.shape[-1]])
		label_tr, label_ts = labs[:, :, :k_shot, :], labs[:, :, :k_shot, :]
		label_tr = tf.reshape(label_tr, [meta_batch_size, n_way * k_shot, label_tr.shape[-1]])
		label_ts = tf.reshape(label_ts, [meta_batch_size, n_way * k_shot, label_ts.shape[-1]])
		inp = (d_tr, d_ts, label_tr, label_ts)
		(result, close) = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

		if itr % SUMMARY_INTERVAL == 0:
			pre_accuracies.append(result[-2])
			post_accuracies.append(result[-1][-1])
			pre_losses.append(result[2])
			post_losses.append(result[3][-1])

		if (itr!=-1) and itr % PRINT_INTERVAL == 0:
			print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
			loss_str = '"""""""""""": pre-inner-loop train loss: %.5f, post-inner-loop test loss %.5f' % (np.mean(pre_losses), np.mean(post_losses))
			print(print_str)
			print(loss_str)
			print(np.mean(label_ts > 0))
			print()

			pre_accuracies, post_accuracies = [], []
			pre_losses, post_losses = [], []

		if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
			(data,  labs) = data_generator.sample_batch('meta_val', meta_batch_size, shuffle=True)
			d_tr, d_ts = data[:, :, :k_shot, :, :], data[:, :, k_shot:, :, :]
			d_tr = tf.reshape(d_tr, [meta_batch_size, n_way * k_shot, d_tr.shape[-2], d_tr.shape[-1]])
			d_ts = tf.reshape(d_ts, [meta_batch_size, n_way * k_shot, d_ts.shape[-2], d_ts.shape[-1]])
			label_tr, label_ts = labs[:, :, :k_shot, :], labs[:, :, :k_shot, :]
			label_tr = tf.reshape(label_tr, [meta_batch_size, n_way * k_shot, label_tr.shape[-1]])
			label_ts = tf.reshape(label_ts, [meta_batch_size, n_way * k_shot, label_ts.shape[-1]])

			inp = (d_tr, d_ts, label_tr, label_ts)
			(result, close) = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

			with train_summary_writer.as_default():
				tf.summary.scalar('accuracy', result[-2], step=itr)
				tf.summary.scalar('mse', result[2], step=itr)
				if num_inner_updates == 0:
					tf.summary.scalar('close', np.mean(close), step=itr)
			if num_inner_updates > 0:
				with test_summary_writer.as_default():
					tf.summary.scalar('accuracy', result[-1][-1], step=itr)
					tf.summary.scalar('mse', result[3][-1], step=itr)
					tf.summary.scalar('close', np.mean(close), step=itr)
					means = []
					stds = []
					if model.meta_reg == 'gaussian':
						for (key, val) in model.meta_reg_dict.items():
							if 'mean' in key:
								means += list(val)
							elif 'std' in key:
								stds += list(val)
						tf.summary.histogram('mean', means, step=itr)
						tf.summary.histogram('std', stds, step=itr)
					elif model.meta_reg == 'l2':
						tf.summary.histogram('beta', [val[0] for (key, val) in model.meta_reg_dict.items()], step=itr)
					tf.summary.histogram('alpha', [val[0] for (key, val) in model.inner_update_lr_dict.items()], step=itr)

			train_accuracy.reset_states()
			test_accuracy.reset_states()

			print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))
			if model.learn_inner_update_lr:
				lrs = [var.numpy() for var in model.trainable_variables if 'inner_update_lr' in var.name]
				print('Variable learning rate mean and stdde')
				print(np.mean(lrs), np.std(lrs))

		if (itr != 0) and itr % SAVE_INTERVAL == 0:
			model_file = logdir + exp_string + '/model' + str(itr)
			print('Saving to ', model_file)
			model.save_weights(model_file)

def meta_test_fn(model, data_generator, n_way = 3, meta_batch_size=16, k_shot=5, num_inner_updates=1):
	num_classes = data_generator.N
	NUM_META_TEST_POINTS = data_generator.num_test

	meta_test_accuracies=[]
	meta_test_outs = []
	closes = []
	meta_pre_losses = []
	meta_post_losses = []
	label_pos = []
	label_neg = []
	mini = 1000
	maxi = -1000
	
	for _ in range(1):
		(data,  labs) = data_generator.sample_batch('meta_test', NUM_META_TEST_POINTS, shuffle=True)
		d_tr, d_ts = data[:, :, :k_shot, :, :], data[:, :, k_shot:, :, :]
		d_tr = tf.reshape(d_tr, [NUM_META_TEST_POINTS, n_way * k_shot, d_tr.shape[-2], d_tr.shape[-1]])
		d_ts = tf.reshape(d_ts, [NUM_META_TEST_POINTS, n_way * k_shot, d_ts.shape[-2], d_ts.shape[-1]])
		label_tr, label_ts = labs[:, :, :k_shot, :], labs[:, :, :k_shot, :]
		label_tr = tf.reshape(label_tr, [NUM_META_TEST_POINTS, n_way * k_shot, label_tr.shape[-1]])
		label_ts = tf.reshape(label_ts, [NUM_META_TEST_POINTS, n_way * k_shot, label_ts.shape[-1]])

		for j in range(NUM_META_TEST_POINTS):
			inp = (np.array([d_tr[j,:,:,:]]), np.array([d_ts[j,:,:,:]]), np.array([label_tr[j,:,:]]), np.array([label_ts[j,:,:]]))
			(result, close) = outer_eval_step(inp, model, meta_batch_size=1, num_inner_updates=num_inner_updates)

			meta_test_accuracies.append(result[-1][-1])
			meta_test_outs.append(result[1])
			meta_pre_losses.append(result[2])
			meta_post_losses.append(result[3])
			label_pos.append(np.mean(label_ts > 0))
			label_neg.append(np.mean(label_ts < 0))
			if np.min(label_ts) < mini:
				mini = np.min(label_ts)
			if np.max(label_ts) > maxi:
				maxi = np.max(label_ts)

			closes.append(close)

	meta_test_accuracies = np.array(meta_test_accuracies)
	means = np.mean(meta_test_accuracies)
	stds = np.std(meta_test_accuracies)
	ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

	returns = []
	actuals = []
	preds = []
	for ticker in data_generator.metatest_tickers:
		print(ticker)
		samples = data_generator.combined[data_generator.combined['ticker'] == ticker].iloc[data_generator.c_length:]
		num_samples = len(samples)
		data = np.zeros((1, n_way, num_samples, data_generator.c_length, data_generator.c_dim))
		labels = np.zeros((1, n_way, num_samples))
		(data[0, :, :, :, :], labels[0, :, :]) = data_generator.get_datapoints(ticker, n_way, num_samples, test=True)
		labels = labels.reshape((1, n_way, num_samples, 1))
		data = np.apply_along_axis(lambda x: x - np.mean(x), -1, data)
		data = np.apply_along_axis(lambda x: x / np.std(x), -1, data)
		ret, actual, pred = get_return(model, data, labels, num_inner_updates)
		returns.append(ret)
		actuals.append(actual)
		preds.append(pred)

	total_weight = np.sum([abs(pred) for pred in preds])
	weighted_returns = []
	for i in range(len(returns)):
		weighted_returns.append((abs(preds[i])/total_weight)*returns[i])
	print('Mean meta-test accuracy and stddev')
	print((means, stds))
	print('Mean meta-test loss and stddev')
	print((np.mean(meta_post_losses), np.std(meta_post_losses)))
	print(np.mean(closes))
	print(np.mean(returns))
	print(np.sum(weighted_returns))
	print(np.mean(actuals))
	print(data_generator.metatest_tickers)

def get_return(model, data, labels, num_inner_updates):
	d_tr, d_ts = data[:, :, -2, :, :], data[:, :, -2, :, :]
	d_tr = tf.reshape(d_tr, [1, len(data) - 2, d_tr.shape[-2], d_tr.shape[-1]])
	d_ts = tf.reshape(d_ts, [1, 1, d_ts.shape[-2], d_ts.shape[-1]])
	label_tr, label_ts = labels[:, :, -3, :], labels[:, :, -2, :]
	label_tr = tf.reshape(label_tr, [1, len(labels) - 2, label_tr.shape[-1]])
	label_ts = tf.reshape(label_ts, [1, 1, label_ts.shape[-1]])

	inp = (d_tr, d_ts, label_tr, label_ts)
	(result, close) = outer_eval_step(inp, model, meta_batch_size=1, num_inner_updates=num_inner_updates)
	out = np.mean(result[1])
	if np.sign(out) == np.sign(label_ts) or np.sign(label_ts) == 0:
		return np.abs(label_ts),label_ts, out
	else:
		return -np.abs(label_ts),label_ts, out

def run_model(n_way = 3, k_shot = 5, meta_batch_size = 16, meta_lr = 0.01,
			  inner_update_lr = 0.4, num_units = 64, num_inner_updates = 1,
			  learn_inner_update_lr = False, log=True, logdir = '../logs/',
			  data_path = '../data/', meta_train = True, meta_train_iterations = 5000,
			  meta_train_k_shot = -1, meta_train_inner_update_lr=-1,
			  time_horizon = 40, resume=False, resume_itr=0, conv=True, meta_reg=False):

	data_generator = DataGenerator(n_way, k_shot * 2, n_way, k_shot *2)

	dim_output = data_generator.dim_output
	dim_input_time = data_generator.c_length
	dim_input = data_generator.c_dim

	model = MAML(dim_input_time, dim_input, dim_output,
				 num_inner_updates = num_inner_updates,
				 inner_update_lr = inner_update_lr,
				 k_shot = k_shot,
				 num_units = num_units,
				 learn_inner_update_lr = learn_inner_update_lr,
				 conv = conv, meta_reg = meta_reg)

	if meta_train_k_shot == -1:
		meta_train_k_shot = k_shot
	if meta_train_inner_update_lr == -1:
		meta_train_inner_update_lr = inner_update_lr

	exp_string = 'maml_cls_.clipped_.meta_reg_learned_' + str(meta_reg) + '.conv_' + str(conv) + '.n_way_' + str(n_way) + '.mbs_' + str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot)\
				 + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr)\
				 + '.learn_inner_update_lr' + str(learn_inner_update_lr)\
				 + '.dim_hidden' + str(num_units)
	#exp_string = 'maml_cls_.meta_reg_learned_True.conv_True.n_way_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.1.learn_inner_update_lrTrue.dim_hidden32'
	#exp_string ='maml_cls.meta_reg_Trueconv_True.n_way_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.04.learn_inner_update_lrTrue.dim_hidden32'
	#exp_string = 'maml_cls_conv_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.04.learn_inner_update_lrTrue.dim_hidden32'
	#exp_string = 'maml_cls_1.mbs_32.k_shot_3.inner_numstep_1.inner_updatelr_0.04.learn_inner_update_lrTrue.dim_hidden64'
	#exp_string = 'maml_cls.conv_False1.mbs_32.k_shot_3.inner_numstep_0.inner_updatelr_0.04.learn_inner_update_lrFalse.dim_hidden32'

	if meta_train:
		if resume:
			model_file = tf.train.latest_checkpoint(logdir + exp_string)
			model.load_weights(model_file)
		meta_train_fn(model, exp_string, data_generator, n_way,
					  meta_train_iterations, meta_batch_size, log, logdir,
					  k_shot, num_inner_updates, meta_lr, resume_itr)
		meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)

	else:
		meta_batch_size = 1
		print(exp_string)
		model_file = tf.train.latest_checkpoint(logdir + exp_string)
		print('Restoring model weights from ', model_file)
		model.load_weights(model_file)

		meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)

run_model(meta_reg = 'l2', conv = False, num_units = 32, meta_train=False, learn_inner_update_lr=True, meta_batch_size=32, inner_update_lr=0.1, n_way = 1, k_shot = 3, meta_train_iterations=10000, num_inner_updates=1)